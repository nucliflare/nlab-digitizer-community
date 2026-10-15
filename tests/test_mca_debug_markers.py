from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from PySide6.QtCore import Qt
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot

from nlab.analysis.energy_calibration import CalibrationPoint, fit_energy_calibration
from nlab.controllers.mca_controller import MCAController
from nlab.workers.mca_worker import MCAReadback


def _controller(qtbot: QtBot, monkeypatch: MonkeyPatch) -> tuple[MCAController, MagicMock]:
    mca = MagicMock()
    mca.get_debug_signal_selectors.return_value = tuple(range(9))
    mca.edge_det_coeff_is_hardware_backed.return_value = False
    monkeypatch.setattr(MCAController, "_load_hardware_state", lambda self: None)
    controller = MCAController(mca, mca_dma=None, channel=0)
    qtbot.addWidget(controller)
    mca.set_trigger_level.reset_mock()
    mca.set_pretrigger_samples.reset_mock()
    return controller, mca


def test_debug_markers_match_hardware_ranges_widgets_and_colors(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    controller, mca = _controller(qtbot, monkeypatch)
    threshold = controller._debug_threshold_line
    offset = controller._debug_pretrigger_line

    assert threshold.value() == -512
    assert threshold.bounds() == (-32768, 32767)
    assert offset.value() == 24
    assert offset.bounds() == (0, 4094)
    assert threshold.pen.style() == Qt.PenStyle.DashLine
    assert offset.pen.style() == Qt.PenStyle.DashLine
    assert threshold.pen.color().name() == "#a66f6f"
    assert offset.pen.color().name() == "#648b71"
    assert threshold.label.color.name() == threshold.pen.color().name()
    assert offset.label.color.name() == offset.pen.color().name()
    assert "#a66f6f" in controller.ui.labelTriggerLevel.styleSheet()
    assert "#648b71" in controller.ui.labelPretrigger.styleSheet()
    assert controller.ui.labelPretrigger.text() == "Pretrigger offset:"

    controller.ui.spinTriggerLevel.setValue(-300)
    controller.ui.sliderPretrigger.setValue(80)
    assert threshold.value() == -300
    assert offset.value() == 80
    assert "80 ns" in offset.label.format
    mca.set_trigger_level.assert_not_called()
    mca.set_pretrigger_samples.assert_not_called()

    controller.ui.spinFrameSamples.setValue(2048)
    assert controller._debug_time_scale.unit == "\N{MICRO SIGN}s"
    assert offset.value() == 0.08
    assert offset.bounds() == (0.0, 4.094)


def test_mca_dma_selection_is_application_configuration(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    controller, _mca = _controller(qtbot, monkeypatch)

    controller.ui.cbDmaEnable.setChecked(True)
    assert controller.configuration_settings()["dma_enabled"] is True

    controller.apply_configuration_settings({"dma_enabled": False})
    assert not controller.ui.cbDmaEnable.isChecked()


def test_threshold_marker_commits_once_via_live_reconfiguration(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    controller, mca = _controller(qtbot, monkeypatch)
    controller._worker = object()
    mca.reconfigure_while_running.side_effect = lambda write: (write(), True)[1]
    controller._last_histogram = np.array([1, 2], dtype=np.uint32)
    controller._last_elapsed_s = 3.0
    line = controller._debug_threshold_line

    line.moving = True
    line.setValue(-400.6)
    assert line.value() == -401
    assert controller.ui.spinTriggerLevel.value() == -401
    assert controller.ui.sliderTriggerLevel.value() == -401
    mca.set_trigger_level.assert_not_called()
    mca.reconfigure_while_running.assert_not_called()

    line.moving = False
    line.sigPositionChangeFinished.emit(line)
    mca.reconfigure_while_running.assert_called_once()
    mca.set_trigger_level.assert_called_once_with(-401)
    assert controller._last_histogram is None
    assert controller._last_elapsed_s == 0.0


def test_offset_marker_previews_both_debug_traces_and_commits_once(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    controller, mca = _controller(qtbot, monkeypatch)
    first = controller._debug1_curve
    second = controller._debug2_curve
    first.setData([0, 8, 16], [10, 20, 10])
    second.setData([0, 8, 16], [2, 3, 2])
    original = [(x.copy(), y.copy()) for x, y in (first.getData(), second.getData())]
    line = controller._debug_pretrigger_line

    line.moving = True
    line.setValue(29.2)
    assert controller.ui.spinPretrigger.value() == 30
    assert controller.ui.sliderPretrigger.value() == 30
    assert line.value() == 30
    for curve, (x, y) in zip((first, second), original):
        preview_x, preview_y = curve.getData()
        np.testing.assert_allclose(preview_x, x + 6)
        np.testing.assert_array_equal(preview_y, y)
    mca.set_pretrigger_samples.assert_not_called()

    controller._update_debug_plot(
        np.array([100, 100], dtype=np.int16), np.array([100, 100], dtype=np.int16)
    )
    np.testing.assert_allclose(first.getData()[0], original[0][0] + 6)

    line.moving = False
    line.sigPositionChangeFinished.emit(line)
    mca.set_pretrigger_samples.assert_called_once_with(30)
    assert controller._pretrigger_line_drag is None
    for curve, (x, y) in zip((first, second), original):
        actual_x, actual_y = curve.getData()
        np.testing.assert_array_equal(actual_x, x)
        np.testing.assert_array_equal(actual_y, y)


def test_marker_drag_cancels_when_controls_disable(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    controller, mca = _controller(qtbot, monkeypatch)
    offset = controller._debug_pretrigger_line
    threshold = controller._debug_threshold_line
    offset.moving = True
    offset.setValue(40)
    threshold.moving = True
    threshold.setValue(-400)

    controller._set_controls_enabled(False)

    assert controller.ui.spinPretrigger.value() == 24
    assert controller.ui.spinTriggerLevel.value() == -512
    assert not offset.movable
    assert not threshold.movable
    offset.moving = False
    threshold.moving = False
    offset.sigPositionChangeFinished.emit(offset)
    threshold.sigPositionChangeFinished.emit(threshold)
    mca.set_pretrigger_samples.assert_not_called()
    mca.set_trigger_level.assert_not_called()


def test_frame_length_change_cancels_offset_preview(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    controller, mca = _controller(qtbot, monkeypatch)
    curve = controller._debug1_curve
    curve.setData([0, 8, 16], [10, 20, 10])
    line = controller._debug_pretrigger_line
    line.moving = True
    line.setValue(30)
    assert controller._pretrigger_line_drag is not None

    controller.ui.spinFrameSamples.setValue(2048)

    assert controller._pretrigger_line_drag is None
    assert controller.ui.spinPretrigger.value() == 24
    assert line.value() == 0.024
    np.testing.assert_allclose(curve.getData()[0][:3], [0, 0.008, 0.016])
    mca.set_pretrigger_samples.assert_not_called()


def test_threshold_marker_failed_write_restores_hardware_value(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    controller, mca = _controller(qtbot, monkeypatch)
    mca.set_trigger_level.side_effect = RuntimeError("busy")
    mca.get_trigger_level.return_value = -512
    line = controller._debug_threshold_line
    line.moving = True
    line.setValue(-400)
    line.moving = False
    line.sigPositionChangeFinished.emit(line)

    mca.set_trigger_level.assert_called_once_with(-400)
    assert controller.ui.spinTriggerLevel.value() == -512
    assert line.value() == -512
    assert "busy" in controller.status.text()


def test_offset_marker_failed_write_restores_hardware_value(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    controller, mca = _controller(qtbot, monkeypatch)
    mca.set_pretrigger_samples.side_effect = RuntimeError("busy")
    mca.get_pretrigger_samples.return_value = 24
    line = controller._debug_pretrigger_line
    line.moving = True
    line.setValue(30)
    line.moving = False
    line.sigPositionChangeFinished.emit(line)

    mca.set_pretrigger_samples.assert_called_once_with(30)
    assert controller.ui.spinPretrigger.value() == 24
    assert line.value() == 24
    assert "busy" in controller.status.text()


def test_energy_calibration_snapshot_axis_and_settings_persistence(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    controller, _mca = _controller(qtbot, monkeypatch)
    histogram = np.arange(32, dtype=np.uint32)
    controller._last_elapsed_s = 12.5
    controller._update_histogram(histogram)
    snapshot = controller.spectrum_snapshot()

    assert snapshot.channel == 0
    assert snapshot.elapsed_s == 12.5
    assert not snapshot.counts.flags.writeable
    np.testing.assert_array_equal(snapshot.counts, histogram)

    controller.display_channel_label = "D1:CH0"
    controller._measurement_configuration = lambda: {
        "connection": {
            "digitizers": [
                {"id": "D1", "ip": "192.0.2.10", "port": 30431, "channels": 2}
            ]
        }
    }
    controller._latest_readback = MCAReadback(
        histogram=histogram,
        debug1=np.empty(0, dtype=np.int16),
        debug2=np.empty(0, dtype=np.int16),
        pulse_deadtime_ms=500.0,
    )
    n42_snapshot = controller.spectrum_snapshot()
    assert n42_snapshot.live_time_s == 12.0
    assert n42_snapshot.n42_instrument is not None
    assert n42_snapshot.n42_instrument["identifier"] == "D1:CH0@192.0.2.10:30431"
    assert n42_snapshot.n42_detector == {
        "name": "D1:CH0",
        "category": "Other",
        "kind": "Other",
        "description": "Detector material and geometry not specified",
    }

    calibration = fit_energy_calibration(
        (CalibrationPoint(4.0, 100.0), CalibrationPoint(20.0, 500.0)),
        fingerprint=controller.energy_calibration_fingerprint(),
    )
    controller.apply_energy_calibration(calibration)

    assert controller._energy_axis.tickStrings([4.0, 20.0], 1.0, 16.0) == [
        "100",
        "500",
    ]
    controller._energy_axis.setRange(0.0, 16_383.0)
    assert controller._energy_axis.label.toPlainText() == "Energy (keV)"
    assert not controller._energy_axis.autoSIPrefix
    settings = controller.configuration_settings()
    restored = settings["energy_calibration"]
    assert isinstance(restored, dict)
    assert restored["model"] == "linear"

    controller.clear_energy_calibration()
    controller.apply_configuration_settings(settings)
    assert controller.energy_calibration is not None
    assert np.isclose(controller.energy_calibration.energy(12.0), 300.0)

    controller.ui.comboBinning.setCurrentIndex(1)
    assert not controller.energy_calibration_is_stale()
    assert controller._energy_axis.tickStrings([2.0, 10.0], 1.0, 8.0) == [
        "100",
        "500",
    ]
    assert np.isclose(controller._calibrated_energy(10.0), 500.0)
    assert not controller._energy_axis._stale

    controller.ui.comboPulsePolarity.setCurrentIndex(1)
    assert controller.energy_calibration_is_stale()
    assert controller._energy_axis._stale


def test_double_clicking_mca_roi_opens_integrated_count_history(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    controller, _mca = _controller(qtbot, monkeypatch)
    controller._roi.setRegion((2, 4))
    controller._last_elapsed_s = 1.0
    controller._update_histogram(np.arange(8, dtype=np.uint32))

    controller._roi.double_clicked.emit()

    dialog = controller._roi_history_dialog
    assert dialog is not None
    assert dialog.isVisible()
    assert dialog.region == (2, 4)
    x_data, y_data = dialog.curve.getData()
    np.testing.assert_array_equal(x_data, [1.0])
    np.testing.assert_array_equal(y_data, [9.0])

    controller._last_elapsed_s = 2.0
    controller._update_histogram(np.arange(8, dtype=np.uint32) + 2)
    x_data, y_data = dialog.curve.getData()
    np.testing.assert_array_equal(x_data, [1.0, 2.0])
    np.testing.assert_array_equal(y_data, [9.0, 15.0])

    controller._roi.setRegion((4, 5))
    controller._on_roi_change_finished()
    x_data, y_data = dialog.curve.getData()
    np.testing.assert_array_equal(x_data, [2.0])
    np.testing.assert_array_equal(y_data, [13.0])

    dialog.close()
    qtbot.waitUntil(lambda: controller._roi_history_dialog is None)


def test_mca_roi_history_tracks_differential_display_mode(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    controller, _mca = _controller(qtbot, monkeypatch)
    controller._roi.setRegion((2, 4))
    controller._last_elapsed_s = 1.0
    first = np.arange(8, dtype=np.uint32)
    controller._update_histogram(first)
    controller._roi.double_clicked.emit()

    dialog = controller._roi_history_dialog
    assert dialog is not None
    controller.set_differential_histogram(True)

    x_data, y_data = dialog.curve.getData()
    np.testing.assert_array_equal(x_data, [1.0])
    np.testing.assert_array_equal(y_data, [0.0])
    assert dialog.plot_widget.getAxis("left").labelText == "Counts / refresh"

    controller._last_elapsed_s = 2.0
    second = np.array([0, 2, 5, 3, 8, 6, 7, 8], dtype=np.uint32)
    controller._update_histogram(second)

    x_data, y_data = dialog.curve.getData()
    np.testing.assert_array_equal(x_data, [1.0, 2.0])
    np.testing.assert_array_equal(y_data, [0.0, 7.0])
    assert dialog.status_label.text().endswith("7 counts in latest refresh")

    controller.set_differential_histogram(False)

    x_data, y_data = dialog.curve.getData()
    np.testing.assert_array_equal(x_data, [2.0])
    np.testing.assert_array_equal(y_data, [16.0])
    assert dialog.plot_widget.getAxis("left").labelText == "Integrated counts"
    assert dialog.status_label.text().endswith("16 accumulated counts")

    dialog.close()
    qtbot.waitUntil(lambda: controller._roi_history_dialog is None)
