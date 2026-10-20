"""Searchable offline parameter reference; no WebEngine or network dependency."""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from nlab.help.content import HelpTopic, load_topics
from nlab.help.context import HelpContext


class HelpDialog(QDialog):
    """Browse a version-matched manual, including from a modal settings dialog."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Nuclear Lab Digitizer — Help")
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setModal(False)
        self.resize(1000, 690)
        screen = self.screen()
        if screen is not None:
            area = screen.availableGeometry()
            self.resize(min(1000, area.width() - 32), min(690, area.height() - 48))
        self.topics: dict[str, HelpTopic] = load_topics()
        self._history: list[HelpContext] = []
        self._history_index = -1
        root = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.back_button = QPushButton("Back", self)
        self.forward_button = QPushButton("Forward", self)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Search parameters, units, or workflows…")
        self.search.setClearButtonEnabled(True)
        toolbar.addWidget(self.back_button)
        toolbar.addWidget(self.forward_button)
        toolbar.addWidget(self.search, 1)
        root.addLayout(toolbar)
        self.selection = QLabel(self)
        self.selection.setTextFormat(Qt.TextFormat.PlainText)
        self.selection.setWordWrap(True)
        self.selection.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.selection)
        splitter = QSplitter(self)
        self.contents = QListWidget(splitter)
        self.contents.setMinimumWidth(180)
        self.contents.setWordWrap(True)
        self.contents.setResizeMode(QListView.ResizeMode.Adjust)
        self.contents.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.browser = QTextBrowser(splitter)
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(False)
        self.browser.document().setDefaultStyleSheet(
            "h1 { font-size: 21px; } p, li { line-height: 140%; }"
        )
        splitter.setSizes([270, 700])
        root.addWidget(splitter, 1)
        footer = QHBoxLayout()
        footer.addWidget(QLabel(
            "Offline help • F1, then click a label or control • Esc cancels", self,
        ))
        footer.addStretch(1)
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.close)
        footer.addWidget(close_button)
        root.addLayout(footer)
        for topic in self.topics.values():
            item = QListWidgetItem(topic.title, self.contents)
            item.setToolTip(topic.title)
            item.setData(Qt.ItemDataRole.UserRole, topic.key)
        self.search.textChanged.connect(self._filter)
        self.contents.currentItemChanged.connect(self._selected)
        self.browser.anchorClicked.connect(self._follow_link)
        self.back_button.clicked.connect(lambda: self._navigate(-1))
        self.forward_button.clicked.connect(lambda: self._navigate(1))
        self.show_topic(HelpContext("help"))

    @property
    def current_topic(self) -> str:
        return self._history[self._history_index].topic

    def show_topic(self, context: HelpContext) -> None:
        if context.topic not in self.topics:
            context = HelpContext("help", context.control, context.tooltip, specific=False)
        self.search.clear()
        if self._history_index < 0 or self._history[self._history_index] != context:
            del self._history[self._history_index + 1:]
            self._history.append(context)
            self._history_index = len(self._history) - 1
        self._display(context)

    def _display(self, context: HelpContext) -> None:
        topic = self.topics[context.topic]
        self.browser.setMarkdown(f"# {topic.title}\n\n{topic.markdown}")
        lines = []
        if context.control:
            lines.append(f"Selected: {context.control}")
        if context.tooltip:
            lines.append(context.tooltip)
        if not context.specific:
            lines.append("Workspace guide: a dedicated entry for this item is not available yet.")
        self.selection.setText("\n".join(lines))
        self.selection.setVisible(bool(lines))
        self.contents.blockSignals(True)
        try:
            for row in range(self.contents.count()):
                item = self.contents.item(row)
                if item.data(Qt.ItemDataRole.UserRole) == context.topic:
                    self.contents.setCurrentItem(item)
                    self.contents.scrollToItem(item)
                    break
        finally:
            self.contents.blockSignals(False)
        self.back_button.setEnabled(self._history_index > 0)
        self.forward_button.setEnabled(self._history_index + 1 < len(self._history))

    def _filter(self, query: str) -> None:
        words = query.casefold().split()
        for row in range(self.contents.count()):
            item = self.contents.item(row)
            topic = self.topics[item.data(Qt.ItemDataRole.UserRole)]
            haystack = f"{topic.key} {topic.title} {topic.markdown}".casefold()
            item.setHidden(not all(word in haystack for word in words))

    def _selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is not None:
            # Preserve the search while browsing its results.
            query = self.search.text()
            self.show_topic(HelpContext(current.data(Qt.ItemDataRole.UserRole)))
            self.search.setText(query)

    def _navigate(self, direction: int) -> None:
        index = self._history_index + direction
        if 0 <= index < len(self._history):
            self.search.clear()
            self._history_index = index
            self._display(self._history[index])

    def _follow_link(self, url: QUrl) -> None:
        if url.fragment() in self.topics and not url.scheme():
            self.show_topic(HelpContext(url.fragment()))
        elif url.scheme() == "https":
            # External browsing is only initiated by an explicit link click.
            QDesktopServices.openUrl(url)
