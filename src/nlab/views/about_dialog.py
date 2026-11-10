"""Application/device information and an explicit GitHub update check."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

from PySide6.QtCore import Qt, QUrl, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nlab import __version__
from nlab.hardware.digitizer.device_information import DigitizerInformation, InformationSection
from nlab.utils.github_releases import GitHubRelease, GitHubReleaseChecker, release_version


class AboutDialog(QDialog):
    def __init__(
        self, about_text: str, parent: QWidget | None = None, *,
        digitizers: Sequence[tuple[str, DigitizerInformation]] = (),
    ) -> None:
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
        hardware = QGroupBox("Digitizer information")
        hardware_layout = QVBoxLayout(hardware)
        note = QLabel("Read-only connection snapshot. Reconnect to refresh device details.")
        note.setWordWrap(True)
        hardware_layout.addWidget(note)
        self.device_tree = QTreeWidget()
        self.device_tree.setHeaderLabels(["Information", "Value"])
        self.device_tree.setColumnWidth(0, 270)
        self.device_tree.setMinimumSize(650, 280)
        self.device_tree.setAlternatingRowColors(True)
        self._populate_digitizers(digitizers)
        hardware_layout.addWidget(self.device_tree)
        layout.addWidget(hardware)
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
        self.copy_button = buttons.addButton(
            "Copy Digitizer Details", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.copy_button.setEnabled(bool(digitizers))
        self.copy_button.clicked.connect(self._copy_digitizer_details)
        self.release_button = buttons.addButton(
            "View Release on GitHub", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.release_button.setVisible(False)
        self.check_button.clicked.connect(self._check_for_updates)
        self.release_button.clicked.connect(self._open_release)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _add_sections(parent: QTreeWidgetItem, sections: tuple[InformationSection, ...]) -> None:
        for section in sections:
            values = dict(section.rows)
            summary = ", ".join(f"{label} {values[key]}" for key, label in (
                ("ip_version", "IP"), ("hw_version", "HW"), ("sw_version", "SW"),
                ("id_number", "ID"), ("parameter_abi_version", "ABI"),
                ("frame_bytes", "Frame bytes"), ("record_layout", "Layout"),
                ("Selected output", "Output"),
            ) if key in values)
            group = QTreeWidgetItem(parent, [section.title, summary])
            for key, value in section.rows or (("Status", "Not reported by target"),):
                row = QTreeWidgetItem(group, [key, value])
                row.setToolTip(1, f"<qt>{escape(value)}</qt>")

    def _populate_digitizers(self, digitizers: Sequence[tuple[str, DigitizerInformation]]) -> None:
        groups: dict[tuple[str, str], QTreeWidgetItem] = {}
        for label, information in digitizers:
            board = label.split(":CH", 1)[0]
            key = (board, information.uri)
            if key not in groups:
                root = QTreeWidgetItem(self.device_tree, [board, information.uri])
                self._add_sections(root, information.board_sections)
                root.setExpanded(True)
                if root.childCount():
                    root.child(0).setExpanded(True)
                groups[key] = root
            channel = QTreeWidgetItem(groups[key], [label])
            self._add_sections(channel, information.channel_sections)
            channel.setExpanded(True)
        if not digitizers:
            QTreeWidgetItem(self.device_tree, ["No connected digitizers (offline mode)"])

    @Slot()
    def _copy_digitizer_details(self) -> None:
        lines: list[str] = []

        def append(item: QTreeWidgetItem, depth: int) -> None:
            value = f": {item.text(1)}" if item.text(1) else ""
            lines.append(f"{'  ' * depth}{item.text(0)}{value}")
            for index in range(item.childCount()):
                append(item.child(index), depth + 1)

        for index in range(self.device_tree.topLevelItemCount()):
            item = self.device_tree.topLevelItem(index)
            if item is not None:
                append(item, 0)
        QApplication.clipboard().setText("\n".join(lines))

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
