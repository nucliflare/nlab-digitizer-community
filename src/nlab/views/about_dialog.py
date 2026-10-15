"""Application information and an explicit GitHub update check."""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QWidget

from nlab import __version__
from nlab.utils.github_releases import GitHubRelease, GitHubReleaseChecker, release_version


class AboutDialog(QDialog):
    def __init__(self, about_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About Nuclear Lab Digitizer")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._release_url: str | None = None
        self._checker = GitHubReleaseChecker(self)
        self._checker.succeeded.connect(self._on_release)
        self._checker.failed.connect(self._on_error)

        layout = QVBoxLayout(self)
        info = QLabel(about_text)
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setOpenExternalLinks(True)
        layout.addWidget(info)
        self.status = QLabel("Check GitHub for a newer published version.")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        self.status.setMaximumWidth(480)
        self.status.setMinimumHeight(self.status.fontMetrics().height() * 3)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.check_button = buttons.addButton(
            "Check for Updates", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.release_button = buttons.addButton(
            "View Release on GitHub", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.release_button.setVisible(False)
        self.check_button.clicked.connect(self._check_for_updates)
        self.release_button.clicked.connect(self._open_release)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @Slot()
    def _check_for_updates(self) -> None:
        self.check_button.setEnabled(False)
        self.release_button.setVisible(False)
        self._release_url = None
        self.status.setText("Checking GitHub for updates…")
        self._checker.check()

    @Slot(object)
    def _on_release(self, release: GitHubRelease) -> None:
        self.check_button.setEnabled(True)
        try:
            installed = release_version(__version__)
        except ValueError as exc:
            self._on_error(str(exc))
            return
        if release.version > installed:
            self.status.setText(
                f"Version {release.tag} is available. You are running {__version__}."
            )
            self._release_url = release.url
            self.release_button.setVisible(True)
        elif release.version == installed:
            self.status.setText(f"You are up to date (version {__version__}).")
        else:
            self.status.setText(
                f"You are running {__version__}, which is newer than the latest "
                f"published GitHub release ({release.tag})."
            )

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self.check_button.setEnabled(True)
        self.status.setText(message)

    @Slot()
    def _open_release(self) -> None:
        if self._release_url is not None:
            if not QDesktopServices.openUrl(QUrl(self._release_url)):
                self.status.setText(f"Could not open your browser. Visit {self._release_url}")

    def done(self, result: int) -> None:
        self._checker.cancel()
        super().done(result)
