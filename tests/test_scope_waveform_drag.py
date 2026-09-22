from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt
from pytestqt.qtbot import QtBot

from nlab.controllers.scope_controller import DisplayMode, ScopeController
from nlab.hardware.digitizer.scope import PARAMETER_SPECS, Scope, TriggerMode
from nlab.views.plot_viewbox import DraggableScopeCurve


def _scope_mock() -> MagicMock:
    scope = MagicMock(spec=Scope)
    scope.specs = PARAMETER_SPECS
    scope.get_trigger_level.return_value = 0
    scope.get_dac_value.return_value = 512
    scope.get_pretrigger_samples.return_value = 32
    scope.get_frame_samples.return_value = 1024
    scope.get_frame_period_cycles.return_value = 0
    scope.get_trigger_mode.return_value = TriggerMode.ANY_BELOW
    scope.frame_period_cycles_supported.return_value = True
    scope.get_dma_enable.return_value = False
    scope.get_enable.return_value = False
    scope.get_viewer_frame_samples_limit.return_value = None
    return scope


def _raw_controller(qtbot: QtBot) -> tuple[ScopeController, MagicMock]:
    scope = _scope_mock()
    controller = ScopeController(scope, scope_dma=None, channel=0)
    qtbot.addWidget(controller)
    controller._set_display_mode(DisplayMode.RAW)
    controller._on_frame_received(
        [
            np.array([0, 8, 16, 24], dtype=np.float64),
            np.array([100, 100, 200, 100], dtype=np.int16),
        ]
    )
    scope.set_dac_value.reset_mock()
    scope.set_pretrigger_samples.reset_mock()
    return controller, scope


def test_waveform_drag_previews_both_axes_and_commits_once(qtbot: QtBot) -> None:
    controller, scope = _raw_controller(qtbot)
    curve = controller._raw_curve
    original_x, original_y = curve.getData()
    original_x = original_x.copy()
    original_y = original_y.copy()

    curve.drag_started.emit()
    curve.drag_moved.emit(0.016, 128.0)

    assert controller.ui.spinPretrigger.value() == 80
    assert controller.ui.spinDacValue.value() == 510
    assert controller.ui.sliderDacValue.value() == 510
    preview_x, preview_y = curve.getData()
    np.testing.assert_allclose(preview_x, original_x + 0.016)
    assert np.all(preview_y > original_y)
    assert "preview approximate" in controller.ui.lblRecordingStatus.text()
    scope.set_dac_value.assert_not_called()
    scope.set_pretrigger_samples.assert_not_called()

    # A viewer worker may finish while the user is still dragging.
    controller._on_frame_received(
        [np.array([0, 8, 16, 24]), np.array([400, 400, 400, 400])]
    )
    frozen_x, frozen_y = curve.getData()
    np.testing.assert_array_equal(frozen_x, preview_x)
    np.testing.assert_array_equal(frozen_y, preview_y)

    curve.drag_finished.emit(False)

    scope.set_pretrigger_samples.assert_called_once_with(40)
    scope.set_dac_value.assert_called_once_with(510)
    assert controller._waveform_drag is None
    assert not curve.clickable  # The preview is not a new measurement.
    restored_x, restored_y = curve.getData()
    np.testing.assert_array_equal(restored_x, original_x)
    np.testing.assert_array_equal(restored_y, original_y)


def test_waveform_drag_snaps_clamps_and_cancels(qtbot: QtBot) -> None:
    controller, scope = _raw_controller(qtbot)
    curve = controller._raw_curve
    original_x, original_y = curve.getData()
    original_x = original_x.copy()
    original_y = original_y.copy()

    curve.drag_started.emit()
    curve.drag_moved.emit(0.003, 0.0)
    assert controller.ui.spinPretrigger.value() == 64
    curve.drag_moved.emit(0.005, 0.0)
    assert controller.ui.spinPretrigger.value() == 72
    curve.drag_moved.emit(100.0, 100_000.0)
    assert controller.ui.spinPretrigger.value() == 2040
    assert controller.ui.spinDacValue.value() == 0
    curve.drag_moved.emit(-100.0, -100_000.0)
    assert controller.ui.spinPretrigger.value() == 0
    assert controller.ui.spinDacValue.value() == 1023
    curve.drag_finished.emit(True)

    assert controller.ui.spinPretrigger.value() == 64
    assert controller.ui.spinDacValue.value() == 512
    x, y = curve.getData()
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)
    scope.set_dac_value.assert_not_called()
    scope.set_pretrigger_samples.assert_not_called()


def test_waveform_drag_uses_measured_dac_direction_and_live_frame(
    qtbot: QtBot,
) -> None:
    controller, scope = _raw_controller(qtbot)
    controller._dac_adc_slope = -32.0
    controller._refresh_timer.start(100_000)
    controller._request_frame = MagicMock()
    curve = controller._raw_curve

    curve.drag_started.emit()
    curve.drag_moved.emit(0.0, 64.0)
    assert controller.ui.spinDacValue.value() == 510
    assert "approximate" not in controller.ui.lblRecordingStatus.text()
    curve.drag_finished.emit(False)

    scope.set_dac_value.assert_called_once_with(510)
    scope.set_pretrigger_samples.assert_not_called()
    controller._request_frame.assert_called_once_with()
    assert "preview until" in controller.ui.lblRecordingStatus.text()
    assert not curve.clickable

    controller._on_frame_received(
        [np.array([0, 8, 16, 24]), np.array([200, 200, 300, 200])]
    )
    _, displayed = curve.getData()
    np.testing.assert_array_equal(displayed, [200, 200, 300, 200])
    assert controller._waveform_pending_status is None
    assert curve.clickable


def test_measured_positive_slope_overrides_negative_fallback(qtbot: QtBot) -> None:
    controller, _scope = _raw_controller(qtbot)
    controller._dac_adc_slope = 32.0
    curve = controller._raw_curve

    curve.drag_started.emit()
    curve.drag_moved.emit(0.0, 64.0)

    assert controller.ui.spinDacValue.value() == 514
    assert "approximate" not in controller.ui.lblRecordingStatus.text()
    curve.drag_finished.emit(True)


def test_waveform_drag_failure_refreshes_controls_and_display(qtbot: QtBot) -> None:
    controller, scope = _raw_controller(qtbot)
    scope.set_dac_value.side_effect = RuntimeError("DAC disconnected")
    curve = controller._raw_curve
    original_x, original_y = curve.getData()
    original_x = original_x.copy()
    original_y = original_y.copy()

    curve.drag_started.emit()
    curve.drag_moved.emit(0.016, 128.0)
    curve.drag_finished.emit(False)

    assert controller._waveform_drag is None
    assert controller.ui.spinPretrigger.value() == 64
    assert controller.ui.spinDacValue.value() == 512
    assert "DAC disconnected" in controller.ui.lblRecordingStatus.text()
    x, y = curve.getData()
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)


def test_waveform_drag_only_available_on_live_raw_trace(qtbot: QtBot) -> None:
    controller, _scope = _raw_controller(qtbot)
    curve = controller._raw_curve
    assert curve.clickable

    controller._set_display_mode(DisplayMode.PERSISTENCE)
    assert not curve.clickable
    controller._set_display_mode(DisplayMode.RAW)
    assert curve.clickable
    controller._set_controls_enabled(False)
    assert not curve.clickable
    controller._set_controls_enabled(True)
    assert curve.clickable


def test_pretrigger_marker_shifts_frozen_trace_and_commits_on_release(
    qtbot: QtBot,
) -> None:
    controller, scope = _raw_controller(qtbot)
    line = controller._pretrigger_line
    curve = controller._raw_curve
    original_x, original_y = curve.getData()
    original_x = original_x.copy()
    original_y = original_y.copy()

    assert line.value() == 0.064
    assert line.bounds() == (0.0, 2.04)
    assert line.pen.style() == Qt.PenStyle.DashLine
    assert line.pen.color().name() == controller._PRETRIGGER_MARKER_COLOR
    assert line.label.color.name() == controller._PRETRIGGER_MARKER_COLOR
    assert controller._threshold_line.pen.color().name() == controller._THRESHOLD_MARKER_COLOR
    assert controller._threshold_line.label.color.name() == controller._THRESHOLD_MARKER_COLOR
    assert controller._PRETRIGGER_MARKER_COLOR in controller.ui.labelPretrigger.styleSheet()
    assert controller._THRESHOLD_MARKER_COLOR in controller.ui.labelTriggerLevel.styleSheet()

    line.moving = True
    line.setValue(0.081)
    assert controller.ui.spinPretrigger.value() == 80
    assert line.value() == 0.08
    assert "80 ns" in line.label.format
    preview_x, preview_y = curve.getData()
    np.testing.assert_allclose(preview_x, original_x + 0.016)
    np.testing.assert_array_equal(preview_y, original_y)
    assert not curve.clickable
    scope.set_pretrigger_samples.assert_not_called()

    controller._on_frame_received(
        [np.array([0, 8, 16, 24]), np.array([400, 400, 400, 400])]
    )
    frozen_x, frozen_y = curve.getData()
    np.testing.assert_array_equal(frozen_x, preview_x)
    np.testing.assert_array_equal(frozen_y, preview_y)

    line.moving = False
    line.sigPositionChangeFinished.emit(line)
    scope.set_pretrigger_samples.assert_called_once_with(40)
    restored_x, restored_y = curve.getData()
    np.testing.assert_array_equal(restored_x, original_x)
    np.testing.assert_array_equal(restored_y, original_y)
    assert controller._waveform_pending_status is not None
    assert not line.movable

    controller._on_frame_received(
        [np.array([0, 8, 16, 24]), np.array([200, 200, 300, 200])]
    )
    assert controller._waveform_pending_status is None
    assert curve.clickable
    assert line.movable


def test_pretrigger_marker_cancel_and_time_axis_follow_widget(qtbot: QtBot) -> None:
    controller, scope = _raw_controller(qtbot)
    line = controller._pretrigger_line
    curve = controller._raw_curve
    original_x, original_y = curve.getData()
    original_x = original_x.copy()
    original_y = original_y.copy()

    line.moving = True
    line.setValue(0.072)
    assert controller.ui.spinPretrigger.value() == 72
    controller._set_controls_enabled(False)
    assert controller._pretrigger_line_drag is None
    assert not line.movable
    assert controller.ui.spinPretrigger.value() == 64
    np.testing.assert_array_equal(curve.getData()[0], original_x)
    np.testing.assert_array_equal(curve.getData()[1], original_y)
    scope.set_pretrigger_samples.assert_not_called()

    line.moving = False
    line.sigPositionChangeFinished.emit(line)
    scope.set_pretrigger_samples.assert_not_called()
    controller._set_controls_enabled(True)

    controller.ui.spinPretrigger.setValue(80)
    assert line.value() == 0.08
    controller.ui.spinFrameSamples.setValue(512)
    controller._on_frame_samples_changed()
    assert controller._time_scale.unit == "ns"
    assert line.value() == 80
    assert line.bounds() == (0.0, 2040.0)

def test_pretrigger_marker_write_failure_reads_back_hardware(qtbot: QtBot) -> None:
    controller, scope = _raw_controller(qtbot)
    line = controller._pretrigger_line
    scope.set_pretrigger_samples.side_effect = RuntimeError("pretrigger write failed")
    line.moving = True
    line.setValue(0.08)
    line.moving = False
    line.sigPositionChangeFinished.emit(line)

    scope.set_pretrigger_samples.assert_called_once_with(40)
    assert controller.ui.spinPretrigger.value() == 64
    assert line.value() == 0.064
    assert line.movable
    assert "pretrigger write failed" in controller.ui.lblRecordingStatus.text()


class _DragEvent:
    def __init__(self, origin: QPointF, pos: QPointF, *, start=False, finish=False):
        self.origin = origin
        self.position = pos
        self.start = start
        self.finish = finish
        self.accepted = False

    def button(self) -> Qt.MouseButton:
        return Qt.MouseButton.LeftButton

    def buttonDownPos(self) -> QPointF:  # noqa: N802
        return self.origin

    def pos(self) -> QPointF:
        return self.position

    def isStart(self) -> bool:  # noqa: N802
        return self.start

    def isFinish(self) -> bool:  # noqa: N802
        return self.finish

    def accept(self) -> None:
        self.accepted = True


class _HoverEvent:
    def __init__(self, pos: QPointF, *, exit: bool = False):
        self.position = pos
        self.exit = exit

    def isExit(self) -> bool:  # noqa: N802
        return self.exit

    def pos(self) -> QPointF:
        return self.position

    def acceptDrags(self, button: Qt.MouseButton) -> bool:  # noqa: N802
        return button == Qt.MouseButton.LeftButton


def test_curve_accepts_drag_on_trace_and_changes_cursor(qtbot: QtBot) -> None:
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    plot.show()
    curve = DraggableScopeCurve(pen="c")
    curve.setData([0, 1, 2], [0, 1, 0])
    plot.addItem(curve)
    qtbot.waitExposed(plot)
    origin = QPointF(1, 1)
    assert curve.mouseShape().contains(origin)
    curve.hoverEvent(_HoverEvent(origin))
    assert curve.cursor().shape() == Qt.CursorShape.OpenHandCursor
    events: list[tuple[str, float, float] | str] = []
    curve.drag_started.connect(lambda: events.append("start"))
    curve.drag_moved.connect(lambda x, y: events.append(("move", x, y)))
    curve.drag_finished.connect(lambda cancelled: events.append("cancel" if cancelled else "end"))

    start = _DragEvent(origin, origin, start=True)
    curve.mouseDragEvent(start)
    assert start.accepted
    assert curve.cursor().shape() == Qt.CursorShape.ClosedHandCursor
    curve.mouseDragEvent(_DragEvent(origin, QPointF(1.5, 1.25)))
    curve.mouseDragEvent(_DragEvent(origin, QPointF(1.5, 1.25), finish=True))

    assert events == [
        "start",
        ("move", 0.0, 0.0),
        ("move", 0.5, 0.25),
        ("move", 0.5, 0.25),
        "end",
    ]


def test_disabling_controls_cancels_an_active_mouse_drag(qtbot: QtBot) -> None:
    controller, scope = _raw_controller(qtbot)
    curve = controller._raw_curve
    origin = QPointF(0.008, 100)
    assert curve.mouseShape().contains(origin)

    curve.mouseDragEvent(_DragEvent(origin, origin, start=True))
    curve.mouseDragEvent(_DragEvent(origin, QPointF(0.024, 228)))
    assert controller._waveform_drag is not None
    assert controller.ui.spinDacValue.value() == 510

    controller._set_controls_enabled(False)

    assert controller._waveform_drag is None
    assert controller.ui.spinDacValue.value() == 512
    assert controller.ui.spinPretrigger.value() == 64
    scope.set_dac_value.assert_not_called()
    scope.set_pretrigger_samples.assert_not_called()
