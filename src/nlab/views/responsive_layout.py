"""Responsive layout helpers for compact and low-resolution displays."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QScreen, QWindow
from PySide6.QtWidgets import (
    QBoxLayout,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from nlab.ui.ui_external_device_view import Ui_ExternalDeviceView
from nlab.ui.ui_global_view import Ui_GlobalView
from nlab.ui.ui_mca_view import Ui_MCAView
from nlab.ui.ui_psd_view import Ui_PSDView
from nlab.ui.ui_psu_view import Ui_PSUView
from nlab.ui.ui_scope_view import Ui_ScopeView


def _scroll_area(name: str, content: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setObjectName(name)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
    scroll.setWidget(content)
    return scroll


def _wrap_layout(
    parent_layout: QBoxLayout,
    content_layout: QLayout,
    name: str,
) -> QScrollArea:
    """Replace a direct child layout with a scrollable content widget."""
    index = parent_layout.indexOf(content_layout)
    if index < 0:
        raise ValueError(f"{content_layout.objectName()} is not in {parent_layout.objectName()}")

    stretch = parent_layout.stretch(index)
    parent_layout.removeItem(content_layout)
    content = QWidget()
    content.setObjectName(f"{name}Contents")
    content.setLayout(content_layout)
    scroll = _scroll_area(name, content)
    parent_layout.insertWidget(index, scroll, stretch)
    return scroll


def _wrap_widget(
    parent_layout: QBoxLayout,
    content: QWidget,
    name: str,
) -> QScrollArea:
    """Replace a direct child widget with a scroll area containing it."""
    index = parent_layout.indexOf(content)
    if index < 0:
        raise ValueError(f"{content.objectName()} is not in {parent_layout.objectName()}")

    stretch = parent_layout.stretch(index)
    parent_layout.removeWidget(content)
    scroll = _scroll_area(name, content)
    parent_layout.insertWidget(index, scroll, stretch)
    return scroll


def _scroll_layout_contents(root_layout: QBoxLayout, name: str) -> QScrollArea:
    """Move every item in a root layout into one scrollable content widget."""
    content_layout = QVBoxLayout()
    content_layout.setObjectName(f"{name}Layout")
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(root_layout.spacing())
    while root_layout.count():
        item = root_layout.takeAt(0)
        if item is not None:
            content_layout.addItem(item)

    content = QWidget()
    content.setObjectName(f"{name}Contents")
    content.setLayout(content_layout)
    scroll = _scroll_area(name, content)
    root_layout.addWidget(scroll)
    return scroll


def _compact_forms(forms: Iterable[QFormLayout]) -> None:
    for form in forms:
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)


def _clear_box(layout: QBoxLayout, preserved: set[QLayout]) -> None:
    """Detach box entries and discard only responsive wrapper rows."""
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            continue
        child = item.layout()
        if child is None or child in preserved:
            continue
        _clear_box(child, preserved)  # type: ignore[arg-type]
        child.deleteLater()


def _add_box_entry(layout: QBoxLayout, entry: QLayout | QWidget) -> None:
    if isinstance(entry, QWidget):
        layout.addWidget(entry)
    else:
        layout.addLayout(entry)


def _set_box_columns(
    layout: QBoxLayout,
    entries: Sequence[QLayout | QWidget],
    columns: int | None,
) -> None:
    preserved = {entry for entry in entries if isinstance(entry, QLayout)}
    _clear_box(layout, preserved)
    if columns is None:
        layout.setDirection(QBoxLayout.Direction.LeftToRight)
        for entry in entries:
            _add_box_entry(layout, entry)
        return

    layout.setDirection(QBoxLayout.Direction.TopToBottom)
    for start in range(0, len(entries), columns):
        row = QHBoxLayout()
        row.setSpacing(layout.spacing())
        for entry in entries[start : start + columns]:
            _add_box_entry(row, entry)
        row.addStretch(1)
        layout.addLayout(row)


class MCAResponsiveLayout(QObject):
    """Switch the MCA view when its top-level window moves between screens."""

    _COMPACT_HEIGHT = 1080

    def __init__(self, view: QWidget, ui: Ui_MCAView, watch_screen: bool) -> None:
        super().__init__(view)
        self._view = view
        self._ui = ui
        self._watch_screen = watch_screen
        self._window: QWindow | None = None
        self._scroll: QScrollArea | None = None
        self._compact = False
        self._mca_entries: tuple[QLayout | QWidget, ...] = (
            ui.vPolarity,
            ui.vBaseline,
            ui.vDebug1,
            ui.vDebug2,
            ui.vPileups,
            ui.vBinning,
            ui.cbExtTrigger,
        )
        self._measurement_entries: tuple[QLayout | QWidget, ...] = (
            ui.vTime,
            ui.vStartStop,
            ui.vRefresh,
            ui.vDmaFile,
        )
        self._forms = (
            ui.formSignal,
            ui.formCrRc2,
            ui.formCfd,
            ui.formTrapez,
            ui.formChargeComp,
            ui.formPsdZc,
            ui.formStatistics,
        )
        self._form_policies = tuple(
            (form.rowWrapPolicy(), form.fieldGrowthPolicy()) for form in self._forms
        )
        self._measurement_policy = QSizePolicy(ui.groupMeasurement.sizePolicy())
        self._measurement_spacing = ui.layoutMeasurement.spacing()
        self._control_minimum = ui.controlPanel.minimumSize()
        self._control_maximum = ui.controlPanel.maximumSize()
        self._debug_minimum = ui.plotDebug.minimumSize()
        self._histogram_minimum = ui.plotHistogram.minimumSize()
        self._maximum_widths = {
            widget: widget.maximumWidth()
            for widget in (
                ui.spinTimeLimit,
                ui.spinRefreshRate,
                ui.btnStart,
                ui.btnStop,
                ui.btnDmaFile,
            )
        }
        if watch_screen:
            view.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if (
            self._watch_screen
            and watched is self._view
            and event.type() == QEvent.Type.Show
        ):
            QTimer.singleShot(0, self._bind_window)
        return super().eventFilter(watched, event)

    def _bind_window(self) -> None:
        window = self._view.window().windowHandle()
        if window is None:
            return
        if window is not self._window:
            if self._window is not None:
                self._window.screenChanged.disconnect(self._apply_screen)
            self._window = window
            window.screenChanged.connect(self._apply_screen)
        self._apply_screen(window.screen())

    def _apply_screen(self, screen: QScreen) -> None:
        physical_height = round(screen.geometry().height() * screen.devicePixelRatio())
        self.apply_screen_height(physical_height)

    def apply_screen_height(self, height: int) -> None:
        compact = height < self._COMPACT_HEIGHT
        if compact == self._compact:
            return
        if compact:
            self._apply_compact()
        else:
            self._apply_regular()
        self._compact = compact
        self._view.updateGeometry()

    def _apply_compact(self) -> None:
        ui = self._ui
        _set_box_columns(ui.layoutMca, self._mca_entries, columns=2)
        _set_box_columns(ui.layoutMeasurement, self._measurement_entries, columns=2)
        ui.topRow.removeWidget(ui.groupMca)
        ui.topRow.removeWidget(ui.groupMeasurement)
        ui.mainLayout.removeItem(ui.topRow)
        ui.controlLayout.insertWidget(0, ui.groupMca)
        ui.controlLayout.insertWidget(1, ui.groupMeasurement)
        ui.groupMeasurement.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        ui.layoutMeasurement.setSpacing(4)
        for widget, width in (
            (ui.spinTimeLimit, 112),
            (ui.spinRefreshRate, 72),
            (ui.btnStart, 76),
            (ui.btnStop, 76),
            (ui.btnDmaFile, 96),
        ):
            widget.setMaximumWidth(width)
        _compact_forms(self._forms)
        self._scroll = _wrap_widget(ui.bodyRow, ui.controlPanel, "mcaControlScroll")
        self._scroll.setMinimumWidth(280)
        self._scroll.setMaximumWidth(480)
        ui.controlPanel.setMinimumWidth(260)
        ui.controlPanel.setMaximumWidth(460)
        ui.plotDebug.setMinimumSize(280, 100)
        ui.plotHistogram.setMinimumSize(280, 140)

    def _apply_regular(self) -> None:
        ui = self._ui
        if self._scroll is not None:
            index = ui.bodyRow.indexOf(self._scroll)
            stretch = ui.bodyRow.stretch(index)
            ui.bodyRow.removeWidget(self._scroll)
            content = self._scroll.takeWidget()
            if content is not ui.controlPanel:
                raise RuntimeError("MCA control scroll lost its control panel")
            ui.bodyRow.insertWidget(index, ui.controlPanel, stretch)
            self._scroll.setParent(None)
            self._scroll.deleteLater()
            self._scroll = None

        ui.controlLayout.removeWidget(ui.groupMca)
        ui.controlLayout.removeWidget(ui.groupMeasurement)
        _set_box_columns(ui.layoutMca, self._mca_entries, columns=None)
        _set_box_columns(ui.layoutMeasurement, self._measurement_entries, columns=None)
        ui.mainLayout.insertLayout(0, ui.topRow)
        ui.topRow.addWidget(ui.groupMca)
        ui.topRow.addWidget(ui.groupMeasurement)
        ui.groupMeasurement.setSizePolicy(self._measurement_policy)
        ui.layoutMeasurement.setSpacing(self._measurement_spacing)
        for widget, width in self._maximum_widths.items():
            widget.setMaximumWidth(width)
        for form, (row_wrap, field_growth) in zip(
            self._forms,
            self._form_policies,
            strict=True,
        ):
            form.setRowWrapPolicy(row_wrap)
            form.setFieldGrowthPolicy(field_growth)
        ui.controlPanel.setMinimumSize(self._control_minimum)
        ui.controlPanel.setMaximumSize(self._control_maximum)
        ui.plotDebug.setMinimumSize(self._debug_minimum)
        ui.plotHistogram.setMinimumSize(self._histogram_minimum)


def configure_scope_layout(view: QWidget, ui: Ui_ScopeView) -> None:
    del view  # The generated form already owns every widget used below.
    _compact_forms((ui.formTrigger, ui.formTiming, ui.formDisplay))
    scroll = _wrap_layout(ui.horizontalLayout, ui.controlLayout, "scopeControlScroll")
    scroll.setMinimumWidth(300)
    scroll.setMaximumWidth(460)
    ui.plotWaveform.setMinimumSize(320, 240)


def configure_mca_layout(
    view: QWidget,
    ui: Ui_MCAView,
    *,
    screen_height: int | None = None,
) -> MCAResponsiveLayout:
    responsive = MCAResponsiveLayout(view, ui, watch_screen=screen_height is None)
    if screen_height is None:
        screen = view.screen()
        screen_height = (
            round(screen.geometry().height() * screen.devicePixelRatio())
            if screen is not None
            else 720
        )
    responsive.apply_screen_height(screen_height)
    return responsive


def configure_psd_layout(view: QWidget, ui: Ui_PSDView) -> None:
    view.setMinimumSize(640, 400)
    placements = (
        (ui.labelEnergyBins, 0, 0),
        (ui.spinEnergyBins, 0, 1),
        (ui.labelRatioBins, 0, 2),
        (ui.spinRatioBins, 0, 3),
        (ui.labelEnergyShift, 1, 0),
        (ui.spinEnergyShift, 1, 1),
        (ui.labelRatioMin, 1, 2),
        (ui.spinRatioMin, 1, 3),
        (ui.labelRatioMax, 2, 0),
        (ui.spinRatioMax, 2, 1),
        (ui.labelCut, 2, 2),
        (ui.spinCut, 2, 3),
        (ui.labelMatrixScale, 0, 4),
        (ui.comboMatrixScale, 1, 4),
        (ui.btnClear, 0, 5),
    )
    for widget, row, column in placements:
        ui.controlsLayout.removeWidget(widget)
        row_span = 3 if widget is ui.btnClear else 1
        ui.controlsLayout.addWidget(widget, row, column, row_span, 1)
    ui.controlsLayout.setColumnStretch(6, 1)


def configure_psu_layout(view: QWidget, ui: Ui_PSUView) -> QScrollArea:
    del view
    _compact_forms(
        (
            ui.formSipm,
            ui.formSipmCompens,
            ui.formHv,
            ui.formHvCompens,
            ui.formTemp,
            ui.formMonitoring,
        )
    )

    ui.controlLayout.removeWidget(ui.groupTemp)
    ui.controlLayout.removeWidget(ui.groupMonitoring)
    ui.controlLayout.removeItem(ui.verticalSpacer)
    telemetry_column = QVBoxLayout()
    telemetry_column.setObjectName("psuTelemetryColumn")
    telemetry_column.setSpacing(ui.columnsLayout.spacing())
    telemetry_column.addWidget(ui.groupTemp)
    telemetry_column.addWidget(ui.groupMonitoring)
    telemetry_column.addStretch(1)
    ui.columnsLayout.addLayout(telemetry_column)
    for column in range(3):
        ui.columnsLayout.setStretch(column, 1)

    scroll = _wrap_layout(ui.horizontalLayout, ui.controlLayout, "psuControlScroll")
    ui.horizontalLayout.setDirection(QBoxLayout.Direction.TopToBottom)
    scroll.setMinimumWidth(0)
    scroll.setMaximumWidth(16_777_215)
    scroll.setMinimumHeight(300)
    scroll.setMaximumHeight(320)
    scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
    ui.plotHvVoltage.setMinimumSize(300, 220)
    return scroll


def configure_global_layout(view: QWidget, ui: Ui_GlobalView) -> None:
    del view
    _compact_forms((ui.formSyncState, ui.formTemperatureSettings, ui.formTemperatureReadback))

    ui.labelSoftwareStartHelp.setMaximumWidth(520)
    ui.btnPrepareSoftware.setMaximumWidth(240)
    ui.btnSoftwareStart.setMaximumWidth(260)
    ui.btnSoftwareReset.setMaximumWidth(240)

    _clear_box(ui.layoutSyncButtons, set())
    ui.layoutSyncButtons.setDirection(QBoxLayout.Direction.TopToBottom)
    ui.layoutSyncButtons.addWidget(ui.btnPrepareSoftware)
    sync_actions = QHBoxLayout()
    sync_actions.setObjectName("globalSyncActions")
    sync_actions.setSpacing(ui.layoutSyncButtons.spacing())
    sync_actions.addWidget(ui.btnSoftwareStart)
    sync_actions.addWidget(ui.btnSoftwareReset)
    sync_actions.addStretch(1)
    ui.layoutSyncButtons.addLayout(sync_actions)

    groups = (
        ui.groupSoftwareStart,
        ui.groupTemperatureCorrection,
        ui.groupDiagnostics,
    )
    for group in groups:
        ui.verticalLayout.removeWidget(group)
        group.setMaximumWidth(16_777_215)

    control_column = QVBoxLayout()
    control_column.setObjectName("globalControlColumn")
    control_column.setSpacing(ui.verticalLayout.spacing())
    control_column.addWidget(ui.groupSoftwareStart)
    control_column.addWidget(ui.groupTemperatureCorrection)
    control_column.addStretch(1)

    panel_columns = QHBoxLayout()
    panel_columns.setObjectName("globalPanelColumns")
    panel_columns.setSpacing(ui.verticalLayout.spacing())
    panel_columns.addLayout(control_column, 1)
    panel_columns.addWidget(
        ui.groupDiagnostics,
        1,
        Qt.AlignmentFlag.AlignTop,
    )
    ui.verticalLayout.addLayout(panel_columns)
    _scroll_layout_contents(ui.verticalLayout, "globalContentScroll")


def configure_external_layout(view: QWidget, ui: Ui_ExternalDeviceView) -> None:
    del view
    _compact_forms((ui.formMonitoring,))
    scroll = _wrap_layout(ui.horizontalLayout, ui.controlLayout, "externalControlScroll")
    # Includes the compact table/group margins and the vertical scroll bar, so
    # the control pane does not introduce a second, horizontal scroll bar.
    scroll.setMinimumWidth(430)
    scroll.setMaximumWidth(650)
    ui.plotWidget.setMinimumSize(300, 220)
