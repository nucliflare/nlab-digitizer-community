from __future__ import annotations

import json
from typing import cast
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QIODevice, QObject, QTimer, QUrl, Signal
from PySide6.QtNetwork import QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import QMainWindow
from pytestqt.qtbot import QtBot

from nlab import app as app_module
from nlab.app import MainAppWindow
from nlab.utils import github_releases
from nlab.utils.github_releases import (
    GITHUB_REPOSITORY_URL,
    LATEST_RELEASE_API,
    GitHubRelease,
    GitHubReleaseChecker,
    parse_latest_release,
    release_version,
)
from nlab.views import about_dialog


def _payload(tag: str = "v0.7.10", **changes: object) -> bytes:
    return json.dumps(
        {
            "tag_name": tag,
            "draft": False,
            "prerelease": False,
            **changes,
        }
    ).encode()


def test_versions_are_compared_numerically_and_ignore_build_metadata() -> None:
    assert release_version("v0.7.10") > release_version("0.7.9")
    assert release_version("v1.0.0") > release_version("0.99.99")
    assert release_version("0.7.2+local") == release_version("v0.7.2")


def test_release_url_stays_on_project_github_page() -> None:
    release = parse_latest_release(_payload(html_url="file:///unexpected"))
    assert release.tag == "v0.7.10"
    assert release.version == (0, 7, 10)
    assert release.url == f"{GITHUB_REPOSITORY_URL}/releases/tag/v0.7.10"


@pytest.mark.parametrize(
    "payload",
    [
        b"not JSON",
        b"[]",
        b"{}",
        _payload(draft=True),
        _payload(prerelease=True),
        _payload("nightly"),
        _payload("v0.8.0-rc1"),
        _payload(tag_name=None),
    ],
)
def test_invalid_or_unstable_release_is_not_offered(payload: bytes) -> None:
    with pytest.raises(ValueError):
        parse_latest_release(payload)


class _Reply(QNetworkReply):
    def __init__(
        self,
        payload: bytes,
        status: int | None = 200,
        error: QNetworkReply.NetworkError = QNetworkReply.NetworkError.NoError,
    ) -> None:
        super().__init__()
        self.payload = payload
        self.aborted = False
        self.setOpenMode(QIODevice.OpenModeFlag.ReadOnly)
        if status is not None:
            self.setAttribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute, status)
        if error != QNetworkReply.NetworkError.NoError:
            self.setError(error, "test network failure")

    def bytesAvailable(self) -> int:  # noqa: N802 - Qt virtual
        return len(self.payload) + super().bytesAvailable()

    def readData(self, maxlen: int) -> bytes:  # noqa: N802 - Qt virtual
        data, self.payload = self.payload[:maxlen], self.payload[maxlen:]
        return data

    def complete(self) -> None:
        self.setFinished(True)
        self.finished.emit()

    def abort(self) -> None:
        self.aborted = True
        self.setError(QNetworkReply.NetworkError.OperationCanceledError, "cancelled")
        self.complete()


def _checker(monkeypatch: pytest.MonkeyPatch, reply: _Reply) -> tuple[GitHubReleaseChecker, Mock]:
    manager = Mock()
    manager.get.return_value = reply
    monkeypatch.setattr(github_releases, "QNetworkAccessManager", lambda _parent: manager)
    return GitHubReleaseChecker(), manager


def test_check_is_asynchronous_and_prevents_duplicate_requests(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply = _Reply(_payload())
    checker, manager = _checker(monkeypatch, reply)
    releases: list[GitHubRelease] = []
    checker.succeeded.connect(releases.append)
    checker.check()
    checker.check()
    manager.get.assert_called_once()
    request = manager.get.call_args.args[0]
    assert request.url().toString() == LATEST_RELEASE_API
    assert request.rawHeader("Accept").data() == b"application/vnd.github+json"
    assert not request.hasRawHeader("Authorization")
    assert not releases
    events: list[str] = []
    QTimer.singleShot(0, lambda: events.append("GUI responsive"))
    qtbot.waitUntil(lambda: bool(events))
    reply.complete()
    assert len(releases) == 1
    assert releases[0].tag == "v0.7.10"
    assert not checker._timeout.isActive()


@pytest.mark.parametrize(
    ("status", "error", "message"),
    [
        (404, QNetworkReply.NetworkError.ContentNotFoundError, "No published GitHub release"),
        (403, QNetworkReply.NetworkError.ContentAccessDenied, "rate limit"),
        (429, QNetworkReply.NetworkError.UnknownContentError, "rate limit"),
        (None, QNetworkReply.NetworkError.HostNotFoundError, "Could not contact GitHub"),
        (500, QNetworkReply.NetworkError.NoError, "HTTP 500"),
    ],
)
def test_network_errors_report_failure_and_allow_retry(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    status: int | None,
    error: QNetworkReply.NetworkError,
    message: str,
) -> None:
    reply = _Reply(b"", status, error)
    checker, manager = _checker(monkeypatch, reply)
    failures: list[str] = []
    releases: list[GitHubRelease] = []
    checker.failed.connect(failures.append)
    checker.succeeded.connect(releases.append)
    checker.check()
    reply.complete()
    assert len(failures) == 1 and message in failures[0]
    assert not releases
    retry = _Reply(_payload())
    manager.get.return_value = retry
    checker.check()
    retry.complete()
    assert len(releases) == 1


def test_timeout_aborts_request_without_a_second_error(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply = _Reply(_payload())
    checker, _ = _checker(monkeypatch, reply)
    failures: list[str] = []
    checker.failed.connect(failures.append)
    checker.check()
    checker._timeout.timeout.emit()
    assert reply.aborted
    assert len(failures) == 1 and "timed out" in failures[0]
    assert checker._reply is None


def test_oversized_response_is_aborted_before_parsing(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply = _Reply(b"x" * (1024 * 1024 + 1))
    checker, _ = _checker(monkeypatch, reply)
    failures: list[str] = []
    checker.failed.connect(failures.append)
    checker.check()
    reply.readyRead.emit()
    assert reply.aborted
    assert len(failures) == 1 and "too large" in failures[0]


class _DialogChecker(QObject):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self.check_count = 0
        self.cancel_count = 0

    def check(self) -> None:
        self.check_count += 1

    def cancel(self) -> None:
        self.cancel_count += 1


@pytest.mark.parametrize(
    ("tag", "message", "has_update"),
    [
        ("v0.7.10", "is available", True),
        ("v0.7.2", "up to date", False),
        ("v0.7.1", "newer than the latest", False),
    ],
)
def test_about_checks_only_on_click_and_reports_version_comparison(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tag: str,
    message: str,
    has_update: bool,
) -> None:
    monkeypatch.setattr(about_dialog, "GitHubReleaseChecker", _DialogChecker)
    monkeypatch.setattr(about_dialog, "__version__", "0.7.2")
    dialog = about_dialog.AboutDialog("Version 0.7.2")
    qtbot.addWidget(dialog)
    dialog.show()
    checker = cast(_DialogChecker, dialog._checker)
    assert checker.check_count == 0
    dialog.check_button.click()
    assert checker.check_count == 1
    assert not dialog.check_button.isEnabled()
    assert "Checking GitHub" in dialog.status.text()
    checker.succeeded.emit(parse_latest_release(_payload(tag)))
    assert message in dialog.status.text()
    assert dialog.check_button.isEnabled()
    assert dialog.release_button.isVisible() == has_update
    browser = Mock(return_value=True)
    monkeypatch.setattr(about_dialog.QDesktopServices, "openUrl", browser)
    if has_update:
        dialog.release_button.click()
        browser.assert_called_once_with(QUrl(f"{GITHUB_REPOSITORY_URL}/releases/tag/{tag}"))
    dialog.check_button.click()
    assert not dialog.release_button.isVisible()
    checker.failed.emit("Offline; please try again.")
    assert dialog.check_button.isEnabled()
    assert dialog.status.text() == "Offline; please try again."
    dialog.reject()
    assert checker.cancel_count == 1


def test_help_about_opens_update_capable_dialog(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = QMainWindow()
    qtbot.addWidget(window)
    factory = Mock()
    monkeypatch.setattr(app_module, "AboutDialog", factory)
    MainAppWindow._on_about(cast(MainAppWindow, window))
    factory.assert_called_once_with(app_module._ABOUT_TEXT, window)
    factory.return_value.exec.assert_called_once_with()
    assert GITHUB_REPOSITORY_URL in app_module._ABOUT_TEXT


def test_closing_about_aborts_inflight_check_without_an_error(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply = _Reply(_payload())
    manager = Mock()
    manager.get.return_value = reply
    monkeypatch.setattr(github_releases, "QNetworkAccessManager", lambda _parent: manager)
    dialog = about_dialog.AboutDialog("Version 0.7.2")
    qtbot.addWidget(dialog)
    failures: list[str] = []
    dialog._checker.failed.connect(failures.append)
    dialog.show()
    dialog.check_button.click()
    dialog.reject()
    assert reply.aborted
    assert not failures
