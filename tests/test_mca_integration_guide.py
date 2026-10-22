"""UI-only CC timing guides and stable legend placement; no live hardware."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from nlab.controllers.mca_controller import MCAController
from nlab.hardware.digitizer.mca import McaGateTiming
from nlab.hardware.digitizer.offline import create_offline_digitizer


def _view(qtbot: QtBot) -> MCAController:
    device = create_offline_digitizer(0)
    view = MCAController(device.mca)
    qtbot.addWidget(view)
    view.show()
    return view


def test_cc_guide_is_black_dashed_stepped_and_only_present_when_enabled(qtbot: QtBot) -> None:
    view = _view(qtbot)
    guide, legend = view._debug_cc_guide, view._debug_plot.legend
    assert not guide.isVisible() and len(legend.items) == 2
    view.ui.spinPretrigger.setValue(32)
    view.ui.spinCcTime.setValue(50)
    view.ui.cbCcEnable.setChecked(True)
    view._debug_plot.setYRange(-1000, 1000, padding=0)
    assert guide.isVisible()
    assert guide.opts["stepMode"] == "right"
    assert guide.opts["pen"].style() == Qt.PenStyle.DashLine
    assert guide.opts["pen"].color().name() == "#000000"
    x, y = guide.getData()
    np.testing.assert_array_equal(x, [0, 0, 50, 256])
    np.testing.assert_allclose(y, [920, -920, 720, 920])
    assert y[1] < y[2] < y[0]  # Inverted short gate, then higher remainder.
    assert "lower: short; upper: remainder" in legend.items[-1][1].text
    assert "not ADC amplitude or a measured FPGA gate" in guide.toolTip()
    for _ in range(5):
        view._update_cc_integration_guide()
    assert len(legend.items) == 3
    view.ui.cbCcEnable.setChecked(False)
    assert not guide.isVisible() and guide.getData() == (None, None)
    assert len(legend.items) == 2


def test_cc_guide_tracks_units_and_clipping_independently_of_pretrigger(qtbot: QtBot) -> None:
    view = _view(qtbot)
    view.ui.cbCcEnable.setChecked(True)
    view.ui.spinFrameSamples.setValue(2048)
    view.ui.spinPretrigger.setValue(80)
    view.ui.spinCcTime.setValue(100)
    assert view._debug_time_scale.unit == "µs"
    np.testing.assert_allclose(view._debug_cc_guide.getData()[0], [0, 0, .1, 2.048])
    view.ui.spinCcTime.setValue(65534)
    np.testing.assert_allclose(view._debug_cc_guide.getData()[0], [0, 0, 2.048, 2.048])
    view.ui.spinCcTime.setValue(0)
    np.testing.assert_allclose(view._debug_cc_guide.getData()[0], [0, 0, 0, 2.048])
    view.ui.spinPretrigger.setValue(4094)
    assert view._debug_cc_guide.isVisible()
    assert len(view._debug_plot.legend.items) == 3
    np.testing.assert_allclose(view._debug_cc_guide.getData()[0], [0, 0, 0, 2.048])
    view.ui.spinFrameSamples.setValue(0)
    assert not view._debug_cc_guide.isVisible()
    assert len(view._debug_plot.legend.items) == 2


def test_cc_guide_does_not_expand_waveform_auto_range_or_change_data(qtbot: QtBot) -> None:
    view = _view(qtbot)
    x, y = np.array([0, 8, 16]), np.array([-100, 50, 0])
    view._debug1_curve.setData(x, y)
    view._debug2_curve.setData(x, y / 2)
    view._debug_plot.autoRange()
    before = np.array(view._debug_plot.viewRange())
    view.ui.cbCcEnable.setChecked(True)
    view._debug_plot.autoRange()
    np.testing.assert_allclose(view._debug_plot.viewRange(), before)
    np.testing.assert_array_equal(view._debug1_curve.getData()[0], x)
    np.testing.assert_array_equal(view._debug1_curve.getData()[1], y)
    view._debug_plot.setYRange(-20, 20, padding=0)
    np.testing.assert_allclose(view._debug_cc_guide.getData()[1], [18.4, -18.4, 14.4, 18.4])


def test_cc_timing_preview_does_not_write_hardware_until_existing_commit(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _view(qtbot)
    view.ui.cbCcEnable.setChecked(True)
    write = Mock(wraps=view._mca.filters.charge_comparison.set_time)
    monkeypatch.setattr(view._mca.filters.charge_comparison, "set_time", write)
    view.ui.spinCcTime.setValue(50)
    view._update_cc_integration_guide()
    write.assert_not_called()
    view.ui.spinCcTime.editingFinished.emit()
    write.assert_called_once_with(50)


def test_transferred_timings_refresh_guide_with_widget_signals_blocked(qtbot: QtBot) -> None:
    view = _view(qtbot)
    view.apply_gate_timing(McaGateTiming(32, 500, 50, True))
    assert view._debug_cc_guide.isVisible()
    np.testing.assert_allclose(view._debug_cc_guide.getData()[0], [0, 0, 50, 500])


def test_cc_short_window_includes_pretrigger_and_does_not_shift_with_marker(qtbot: QtBot) -> None:
    view = _view(qtbot)
    view.ui.spinFrameSamples.setValue(4096)
    view.ui.spinPretrigger.setValue(176)
    view.ui.spinCcTime.setValue(720)
    view.ui.cbCcEnable.setChecked(True)
    guide = view._debug_cc_guide
    np.testing.assert_allclose(guide.getData()[0], [0, 0, .720, 4.096])
    assert "pretrigger included" in guide.toolTip()
    before = guide.getData()[0].copy()
    view.ui.spinPretrigger.setValue(280)
    np.testing.assert_array_equal(guide.getData()[0], before)
    assert view._debug_pretrigger_line.value() == .280


def test_mca_legend_remains_bottom_right_when_resized_and_guide_toggled(qtbot: QtBot) -> None:
    view = _view(qtbot)
    legend = view._debug_plot.legend
    for width, height, enabled in ((1100, 750, False), (1400, 900, True), (1100, 750, False)):
        view.resize(width, height)
        view.ui.cbCcEnable.setChecked(enabled)

        def anchored() -> bool:
            rect = legend.mapRectToParent(legend.boundingRect())
            parent = legend.parentItem().boundingRect()
            return bool(abs(parent.right() - rect.right() - 10) < 1
                        and abs(parent.bottom() - rect.bottom() - 10) < 1)

        qtbot.waitUntil(anchored)
        assert legend.opts["offset"] == (-10, -10)


def _click_legend_sample(view: MCAController, item: object) -> None:
    sample = next(sample for sample, _label in view._debug_plot.legend.items
                  if sample.item is item)
    # Exercise pyqtgraph's actual legend handler, including the visibility
    # toggle and sigSampleClicked forwarding, rather than calling our slot.
    sample.mouseClickEvent(SimpleNamespace(button=lambda: Qt.MouseButton.LeftButton,
                                          accept=lambda: None))


def test_legend_hidden_cc_guide_stays_hidden_through_measurement_refreshes(qtbot: QtBot) -> None:
    view = _view(qtbot)
    view.ui.cbCcEnable.setChecked(True)
    guide = view._debug_cc_guide
    _click_legend_sample(view, guide)
    assert not guide.isVisible()
    view._set_measurement_active(True)  # Simulated lifecycle only; no acquisition started.
    for amplitude in (50, 100, 200):
        trace = np.array([0, -amplitude, -amplitude // 2, 0], dtype=np.int16)
        view._update_debug_plot(trace, trace // 2)
        view._debug_plot.setYRange(-amplitude * 2, amplitude, padding=0)
        assert not guide.isVisible()
        assert len(view._debug_plot.legend.items) == 3
    view._set_measurement_active(False)
    view.ui.spinCcTime.setValue(100)
    view.ui.spinPretrigger.setValue(80)
    view.ui.spinFrameSamples.setValue(2048)
    assert not guide.isVisible()
    assert view.ui.cbCcEnable.isChecked()
    assert view._mca.filters.charge_comparison.get_enable()
    _click_legend_sample(view, guide)
    assert guide.isVisible()
    np.testing.assert_allclose(guide.getData()[0], [0, 0, .1, 2.048])
    view._update_debug_plot(trace, trace)
    assert guide.isVisible()


def test_cc_legend_preference_survives_availability_changes_and_transfer(qtbot: QtBot) -> None:
    view = _view(qtbot)
    view.ui.cbCcEnable.setChecked(True)
    guide = view._debug_cc_guide
    _click_legend_sample(view, guide)
    view.ui.cbCcEnable.setChecked(False)
    assert len(view._debug_plot.legend.items) == 2
    view.ui.cbCcEnable.setChecked(True)
    assert len(view._debug_plot.legend.items) == 3 and not guide.isVisible()
    view.ui.spinFrameSamples.setValue(0)
    assert len(view._debug_plot.legend.items) == 2
    view.ui.spinFrameSamples.setValue(256)
    assert len(view._debug_plot.legend.items) == 3 and not guide.isVisible()
    view.apply_gate_timing(McaGateTiming(32, 500, 50, True))
    assert not guide.isVisible()
    _click_legend_sample(view, guide)
    assert guide.isVisible()
    np.testing.assert_allclose(guide.getData()[0], [0, 0, 50, 500])
    view.ui.cbCcEnable.setChecked(False)
    view.ui.cbCcEnable.setChecked(True)
    assert guide.isVisible()


def test_hiding_debug_trace_does_not_hide_cc_guide(qtbot: QtBot) -> None:
    view = _view(qtbot)
    view.ui.cbCcEnable.setChecked(True)
    _click_legend_sample(view, view._debug1_curve)
    view._update_debug_plot(np.zeros(8), np.zeros(8))
    assert not view._debug1_curve.isVisible()
    assert view._debug2_curve.isVisible() and view._debug_cc_guide.isVisible()
