from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget
from pytestqt.qtbot import QtBot

from nlab.controllers.main_window_controller import MainWindowController
from nlab.controllers.mca_controller import MCAController
from nlab.hardware.digitizer.mca_capture import McaDmaOutputMode, McaRunSummary
from nlab.hardware.digitizer.offline import create_offline_digitizer
from nlab.views.panel_status import PanelStatus


def _workspace(window: QMainWindow) -> MainWindowController:
    controller = object.__new__(MainWindowController)
    controller._window = window  # type: ignore[assignment]
    return controller


class _Panel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.status = PanelStatus(self)


def test_mca_progress_errors_and_run_details_use_status_bar_without_resizing(
    qtbot: QtBot,
) -> None:
    window = QMainWindow()
    qtbot.addWidget(window)
    digitizer = create_offline_digitizer(0)
    mca = MCAController(digitizer.mca)
    controller = _workspace(window)
    dock = controller._make_dock("mca_ch0", "D1:CH0", mca)
    window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    window.resize(1400, 900)
    window.show()
    QApplication.processEvents()
    qtbot.waitUntil(mca.isVisible)
    before_size = window.size()
    before_hint = window.minimumSizeHint()
    before_controls = mca.ui.groupMeasurement.size()
    bar = window.statusBar()

    mca._on_dma_progress(2048)
    assert bar.currentMessage().startswith("MCA D1:CH0: Recording: 2,048 records")
    error = "Transport failure: " + "very long diagnostic message " * 100
    mca._on_dma_error(error)
    assert bar.currentMessage() == f"MCA D1:CH0: Error: {error}"
    assert error in bar.toolTip()

    summary = McaRunSummary(
        channel=0,
        mode=McaDmaOutputMode.BINARY,
        path=None,
        started_utc="2026-10-15T10:00:00+00:00",
        finished_utc="2026-10-15T10:00:02+00:00",
        duration_s=2,
        records=2048,
        continuity="invalid",
        diagnostics={"dma_fault": 1, "dma_error_count": 4},
        error=error,
    )
    mca._on_dma_summary(summary)
    qtbot.waitUntil(lambda: "dma_error_count (cumulative): 4" in bar.toolTip())

    assert bar.currentMessage() == f"MCA D1:CH0: {summary.status_text()}"
    assert window.size() == before_size
    assert window.minimumSizeHint() == before_hint
    assert mca.ui.groupMeasurement.size() == before_controls
    assert not hasattr(mca.ui, "lblDmaStatus")
    digitizer.close()


def test_tabified_docks_restore_status_and_suppress_background_progress(qtbot: QtBot) -> None:
    window = QMainWindow()
    qtbot.addWidget(window)
    controller = _workspace(window)
    first, second = _Panel(), _Panel()
    first.status.setText("Recording channel zero")
    second.status.setText("Ready on channel one")
    docks = [
        controller._make_dock("mca_ch0", "D1:CH0", first),
        controller._make_dock("mca_ch1", "D1:CH1", second),
    ]
    for dock in docks:
        window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    window.tabifyDockWidget(*docks)
    docks[0].raise_()
    window.show()
    bar = window.statusBar()
    qtbot.waitUntil(lambda: bar.currentMessage() == "MCA D1:CH0: Recording channel zero")

    second.status.setText("Background channel finished")
    assert bar.currentMessage() == "MCA D1:CH0: Recording channel zero"
    docks[1].raise_()
    qtbot.waitUntil(lambda: bar.currentMessage() == "MCA D1:CH1: Background channel finished")
    first.status.setText("Channel zero finished")
    assert bar.currentMessage() == "MCA D1:CH1: Background channel finished"
    docks[0].raise_()
    qtbot.waitUntil(lambda: bar.currentMessage() == "MCA D1:CH0: Channel zero finished")


def test_primary_tabs_restore_messages_and_clear_status_on_other_pages(qtbot: QtBot) -> None:
    window = QMainWindow()
    qtbot.addWidget(window)
    tabs = QTabWidget()
    window.setCentralWidget(tabs)
    host = QMainWindow()
    tabs.addTab(host, "MCA")
    coincidence = _Panel()
    QVBoxLayout(coincidence)
    tabs.addTab(coincidence, "Coincidence")
    tabs.addTab(QWidget(), "Other")
    controller = _workspace(window)
    mca = _Panel()
    dock = controller._make_dock("mca_ch0", "D2:CH0", mca)
    host.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    controller._bind_panel_status(coincidence, "Coincidence D2")
    mca.status.setText("Recording")
    coincidence.status.setText("Waiting for both readers")
    window.show()
    bar = window.statusBar()
    qtbot.waitUntil(lambda: bar.currentMessage() == "MCA D2:CH0: Recording")

    tabs.setCurrentIndex(1)
    qtbot.waitUntil(lambda: bar.currentMessage() == "Coincidence D2: Waiting for both readers")
    mca.status.setText("Verified")
    assert bar.currentMessage() == "Coincidence D2: Waiting for both readers"
    tabs.setCurrentIndex(2)
    qtbot.waitUntil(lambda: bar.currentMessage() == "")
    assert bar.toolTip() == ""
    tabs.setCurrentIndex(0)
    qtbot.waitUntil(lambda: bar.currentMessage() == "MCA D2:CH0: Verified")
