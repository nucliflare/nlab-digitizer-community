"""Read published GitHub releases without blocking the Qt event loop."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import quote

from PySide6.QtCore import QObject, QTimer, QUrl, Signal, Slot
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

from nlab import __version__

GITHUB_REPOSITORY_URL = "https://github.com/nucliflare/nlab-digitizer-community"
LATEST_RELEASE_API = (
    "https://api.github.com/repos/nucliflare/nlab-digitizer-community/releases/latest"
)
_MAX_RESPONSE_BYTES = 1024 * 1024
_TIMEOUT_MS = 15_000
_VERSION_RE = re.compile(
    r"[vV]?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:\+[0-9A-Za-z.-]+)?"
)


def release_version(tag: str) -> tuple[int, int, int]:
    """Compare stable MAJOR.MINOR.PATCH tags numerically, ignoring build metadata."""
    match = _VERSION_RE.fullmatch(tag)
    if match is None:
        raise ValueError(f"Unsupported release version: {tag!r}.")
    return int(match[1]), int(match[2]), int(match[3])


@dataclass(frozen=True)
class GitHubRelease:
    tag: str
    version: tuple[int, int, int]
    url: str


def parse_latest_release(payload: bytes) -> GitHubRelease:
    """Validate GitHub's latest full release, excluding drafts and prereleases.

    Source: https://docs.github.com/en/rest/releases/releases#get-the-latest-release
    A published release is distinct from a Git tag or a completed CI build.
    """
    if len(payload) > _MAX_RESPONSE_BYTES:
        raise ValueError("GitHub's release response was too large.")
    try:
        document = json.loads(payload)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("GitHub returned an invalid release response.") from exc
    if not isinstance(document, dict):
        raise ValueError("GitHub returned an invalid release response.")
    if document.get("draft") is not False or document.get("prerelease") is not False:
        raise ValueError("GitHub did not return a published stable release.")
    tag = document.get("tag_name")
    if not isinstance(tag, str) or len(tag) > 128:
        raise ValueError("GitHub's release is missing a valid version tag.")
    return GitHubRelease(
        tag=tag,
        version=release_version(tag),
        url=f"{GITHUB_REPOSITORY_URL}/releases/tag/{quote(tag, safe='')}",
    )


class GitHubReleaseChecker(QObject):
    """One bounded, anonymous HTTPS request, initiated only by the user."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._manager = QNetworkAccessManager(self)
        self._reply: QNetworkReply | None = None
        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.setInterval(_TIMEOUT_MS)
        self._timeout.timeout.connect(self._on_timeout)

    @Slot()
    def check(self) -> None:
        if self._reply is not None:
            return
        request = QNetworkRequest(QUrl(LATEST_RELEASE_API))
        request.setRawHeader(b"Accept", b"application/vnd.github+json")
        request.setRawHeader(b"User-Agent", f"nlab/{__version__}".encode("ascii"))
        request.setAttribute(
            QNetworkRequest.Attribute.RedirectPolicyAttribute,
            QNetworkRequest.RedirectPolicy.SameOriginRedirectPolicy,
        )
        reply = self._manager.get(request)
        self._reply = reply
        reply.setReadBufferSize(_MAX_RESPONSE_BYTES + 1)
        reply.readyRead.connect(self._on_ready_read)
        reply.finished.connect(self._on_finished)
        self._timeout.start()

    @Slot()
    def cancel(self) -> None:
        self._timeout.stop()
        reply, self._reply = self._reply, None
        if reply is not None:
            reply.abort()

    @Slot()
    def _on_timeout(self) -> None:
        self.cancel()
        self.failed.emit("The GitHub update check timed out. Please try again.")

    @Slot()
    def _on_ready_read(self) -> None:
        if self._reply is not None and self._reply.bytesAvailable() > _MAX_RESPONSE_BYTES:
            self.cancel()
            self.failed.emit("GitHub's release response was too large.")

    @Slot()
    def _on_finished(self) -> None:
        reply = self.sender()
        if not isinstance(reply, QNetworkReply):
            return
        if reply is not self._reply:
            reply.deleteLater()
            return
        self._reply = None
        self._timeout.stop()
        try:
            status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
            if status == 404:
                self.failed.emit("No published GitHub release was found for this repository.")
            elif status in (403, 429):
                self.failed.emit(
                    "GitHub refused the update check or its rate limit was reached. "
                    "Please try again later."
                )
            elif reply.error() != QNetworkReply.NetworkError.NoError:
                self.failed.emit(f"Could not contact GitHub: {reply.errorString()}")
            elif status != 200:
                self.failed.emit(f"GitHub returned HTTP {status} while checking for updates.")
            else:
                self.succeeded.emit(parse_latest_release(bytes(reply.readAll().data())))
        except ValueError as exc:
            self.failed.emit(str(exc))
        finally:
            reply.deleteLater()
