"""Responsive layout helpers for compact and low-resolution displays."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer
from PySide6.QtGui import QScreen, QWindow
from PySide6.QtWidgets import (
    QBoxLayout,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLayout,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from nlab.ui.ui_current_monitor_view import Ui_CurrentMonitorView
from nlab.ui.ui_external_device_view import Ui_ExternalDeviceView
from nlab.ui.ui_global_view import Ui_GlobalView
from nlab.ui.ui_mca_view import Ui_MCAView
from nlab.ui.ui_psd_view import Ui_PSDView
from nlab.ui.ui_psu_view import Ui_PSUView
from nlab.ui.ui_scope_view import Ui_ScopeView

QWIDGETSIZE_MAX = 16_777_215  # Qt's "no maximum" widget size


class _ScrollColumn(QScrollArea):
    """A vertical scroll column that prefers its content's natural width.

    QScrollArea's own size hint ignores the content's preferred width, so a
    splitter would open the wrapped column at its minimum and clip its labels.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scroll_bar_extent = 0

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        hint = super().sizeHint()
        content = self.widget()
        if content is not None:
            width = content.sizeHint().width() + self.scroll_bar_extent
            hint.setWidth(max(self.minimumWidth(), min(self.maximumWidth(), width)))
        return hint


def _scroll_area(
    name: str, content: QWidget, scroll: QScrollArea | None = None,
) -> QScrollArea:
    scroll = QScrollArea() if scroll is None else scroll
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


def _fit_scroll_width(scroll: QScrollArea, content: QWidget) -> None:
    """Size a vertical scroll column to its content's width plus the scroll bar.

    Without the allowance, a column whose content has a fixed width would also
    need a horizontal scroll bar whenever its vertical one appears.
    """
    bar = scroll.verticalScrollBar()
    extent = max(
        bar.sizeHint().width(),
        scroll.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent, None, bar),
    )
    scroll.setMinimumWidth(content.minimumWidth() + extent)
    if content.maximumWidth() < QWIDGETSIZE_MAX:
        scroll.setMaximumWidth(content.maximumWidth() + extent)
    else:
        scroll.setMaximumWidth(QWIDGETSIZE_MAX)
    if isinstance(scroll, _ScrollColumn):
        scroll.scroll_bar_extent = extent


def scroll_column(name: str, content: QWidget) -> QScrollArea:
    """A scrollable column for *content*, whose height then never sets the
    window's minimum height; the column keeps the content's width limits and
    preferred width."""
    scroll = _scroll_area(name, content, _ScrollColumn())
    _fit_scroll_width(scroll, content)
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
    """Choose the wide or compact MCA layout for the screen showing the view.

    The choice uses the screen's available area in device-independent pixels,
    the units every widget size is in. Display scaling shrinks that area while
    the physical resolution stays the same: 1920x1080 at the common 125%
    Windows default offers 1536x816 above the taskbar, too small for the wide
    layout that a physical 1080-line test selected.

    The control column scrolls in both layouts, so its length never becomes
    the window's minimum height. When it did, the window could not fit a
    1080p work area, and a maximized window squeezed the channel docks below
    their content, drawing the dock tab bar over the controls.
    """

    # Available device-independent area the wide layout needs: its header row
    # (MCA and Measurement groups) is about 1,390 px wide with the window frame,
    # and below about 900 px the plots beside it become too short to be useful.
    _COMPACT_HEIGHT = 900
    _COMPACT_WIDTH = 1440

    def __init__(self, view: QWidget, ui: Ui_MCAView, watch_screen: bool) -> None:
        super().__init__(view)
        self._view = view
        self._ui = ui
        self._watch_screen = watch_screen
        self._window: QWindow | None = None
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
            ui.vSeries,
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
                ui.spinSeriesRuns,
                ui.spinRefreshRate,
                ui.btnStart,
                ui.btnStop,
                ui.btnDmaFile,
            )
        }
        self._scroll = _wrap_widget(ui.bodyRow, ui.controlPanel, "mcaControlScroll")
        _fit_scroll_width(self._scroll, ui.controlPanel)
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
        available = screen.availableGeometry()
        self.apply_screen_height(available.height(), available.width())

    def apply_screen_height(self, height: int, width: int | None = None) -> None:
        """Lay out for an available screen area in device-independent pixels."""
        compact = height < self._COMPACT_HEIGHT or (
            width is not None and width < self._COMPACT_WIDTH
        )
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
            (ui.spinSeriesRuns, 72),
            (ui.spinRefreshRate, 72),
            (ui.btnStart, 76),
            (ui.btnStop, 76),
            (ui.btnDmaFile, 96),
        ):
            widget.setMaximumWidth(width)
        _compact_forms(self._forms)
        self._scroll.setMinimumWidth(280)
        self._scroll.setMaximumWidth(480)
        ui.controlPanel.setMinimumWidth(260)
        ui.controlPanel.setMaximumWidth(460)
        ui.plotDebug.setMinimumSize(280, 100)
        ui.plotHistogram.setMinimumSize(280, 140)

    def _apply_regular(self) -> None:
        ui = self._ui
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
        _fit_scroll_width(self._scroll, ui.controlPanel)
        ui.plotDebug.setMinimumSize(self._debug_minimum)
        ui.plotHistogram.setMinimumSize(self._histogram_minimum)


def configure_scope_layout(view: QWidget, ui: Ui_ScopeView) -> None:
    del view  # The generated form already owns every widget used below.
    _compact_forms((ui.formTrigger, ui.formTiming, ui.formDisplay))
    scroll = _wrap_layout(ui.horizontalLayout, ui.controlLayout, "scopeControlScroll")
    scroll.setMinimumWidth(300)
    scroll.setMaximumWidth(460)
    ui.plotWaveform.setMinimumSize(320, 240)


def configure_current_layout(view: QWidget, ui: Ui_CurrentMonitorView) -> QScrollArea:
    """Settings and readings in one scrollable column beside the plot, as on Scope."""
    del view
    _compact_forms((
        ui.formAcquisition, ui.formSumTiming, ui.formSumRecording,
        ui.formCalibration, ui.formReading,
    ))
    scroll = _wrap_layout(ui.horizontalLayout, ui.controlLayout, "currentControlScroll")
    scroll.setMinimumWidth(300)
    scroll.setMaximumWidth(460)
    ui.plotCurrent.setMinimumSize(320, 240)
    return scroll


def configure_mca_layout(
    view: QWidget,
    ui: Ui_MCAView,
    *,
    screen_height: int | None = None,
) -> MCAResponsiveLayout:
    responsive = MCAResponsiveLayout(view, ui, watch_screen=screen_height is None)
    if screen_height is not None:
        responsive.apply_screen_height(screen_height)
        return responsive
    screen = view.screen()
    if screen is None:
        responsive.apply_screen_height(720)
    else:
        available = screen.availableGeometry()
        responsive.apply_screen_height(available.height(), available.width())
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

    # One control column beside the plot, as on the Scope tab: the groups stack
    # at a readable width and the voltage history takes the remaining space.
    groups = (
        ui.groupSipm,
        ui.groupSipmCompens,
        ui.groupHv,
        ui.groupHvCompens,
    )
    ui.columnLeft.removeWidget(ui.groupSipm)
    ui.columnLeft.removeWidget(ui.groupSipmCompens)
    ui.columnRight.removeWidget(ui.groupHv)
    ui.columnRight.removeWidget(ui.groupHvCompens)
    ui.controlLayout.removeItem(ui.columnsLayout)
    ui.columnsLayout.deleteLater()
    for index, group in enumerate(groups):
        ui.controlLayout.insertWidget(index, group)

    scroll = _wrap_layout(ui.horizontalLayout, ui.controlLayout, "psuControlScroll")
    scroll.setMinimumWidth(300)
    scroll.setMaximumWidth(460)
    ui.plotHvVoltage.setMinimumSize(320, 240)
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

    # Both columns keep their natural width and pack to the left; a wide
    # window adds empty space on the right instead of spreading each group's
    # labels and values apart.
    panel_columns = QHBoxLayout()
    panel_columns.setObjectName("globalPanelColumns")
    panel_columns.setSpacing(ui.verticalLayout.spacing())
    panel_columns.addLayout(control_column, 0)
    panel_columns.addWidget(
        ui.groupDiagnostics,
        0,
        Qt.AlignmentFlag.AlignTop,
    )
    panel_columns.addStretch(1)
    ui.verticalLayout.addLayout(panel_columns)
    _scroll_layout_contents(ui.verticalLayout, "globalContentScroll")


def configure_external_layout(view: QWidget, ui: Ui_ExternalDeviceView) -> None:
    del view
    _compact_forms((ui.formMonitoring,))
    scroll = _wrap_layout(ui.horizontalLayout, ui.controlLayout, "externalControlScroll")
    # Includes the compact table/group margins and the vertical scroll bar, so
    # the control pane does not introduce a second, horizontal scroll bar.
    scroll.setMinimumWidth(440)
    scroll.setMaximumWidth(650)
    ui.plotWidget.setMinimumSize(300, 220)
