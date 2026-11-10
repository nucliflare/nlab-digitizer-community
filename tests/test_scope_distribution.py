from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QMainWindow
from pytestqt.qtbot import QtBot

from nlab import app as app_module
from nlab.analysis.scope_distribution import (
    AccumulatedAmplitudeHistogram,
    HistogramMode,
    pulse_integral,
)
from nlab.app import MainAppWindow
from nlab.controllers.main_window_controller import MainWindowController
from nlab.controllers.scope_controller import DisplayMode, ScopeController
from nlab.hardware.digitizer.current_monitor import ScopeCurrentPreview
from nlab.hardware.digitizer.dma import ScopeDmaGeometry
from nlab.hardware.digitizer.scope import PARAMETER_SPECS, Scope, TriggerMode
from nlab.ui.ui_main_window import Ui_MainWindow


def test_histogram_accumulates_frames_at_adc_resolution_and_stays_bounded() -> None:
    histogram = AccumulatedAmplitudeHistogram()
    histogram.add(np.array([-32768, -1, 0, 0, 32767], dtype=np.int16), source="viewer")
    histogram.add(np.array([0, 1, 32767], dtype=np.int16), source="viewer")
    assert histogram.counts[0] == 1
    assert histogram.counts[32767] == 1
    assert histogram.counts[32768] == 3
    assert histogram.counts[-1] == 2
    assert histogram.sample_count == 8
    assert histogram.frame_count == 2
    assert histogram.counts.nbytes == 65536 * 8
    edges, counts = histogram.projected_bins(-40000, 40000)
    assert len(edges) == len(counts) + 1
    assert int(counts.sum()) == 8


def test_fractional_samples_round_and_invalid_values_are_not_clipped() -> None:
    histogram = AccumulatedAmplitudeHistogram()
    histogram.add(
        np.array([-0.51, -0.5, 0.49, 0.5, np.nan, np.inf, -32769, 32768]),
        source="viewer",
    )
    assert histogram.counts[32767:32770].tolist() == [1, 2, 1]
    assert histogram.sample_count == 4
    assert histogram.rejected_count == 4


def test_zoom_crops_native_adc_bins_without_mutating_or_discarding_counts() -> None:
    histogram = AccumulatedAmplitudeHistogram()
    histogram.add(np.arange(-100, 101), source="viewer")
    original = histogram.counts.copy()
    edges, counts = histogram.projected_bins(-2.5, 2.5)
    np.testing.assert_array_equal(edges, [-68.5, 31.5])
    np.testing.assert_array_equal(counts, [100])
    np.testing.assert_array_equal(histogram.counts, original)
    _, full_counts = histogram.projected_bins(-32768.5, 32767.5)
    assert full_counts.size == 656
    assert int(full_counts.sum()) == 201
    assert histogram.projected_bins(40000, 50000)[1].size == 0
    assert histogram.projected_bins(np.nan, 10)[1].size == 0
    assert histogram.projected_bins(10, 0)[1].size == 0


def test_source_change_and_clear_do_not_mix_representations() -> None:
    histogram = AccumulatedAmplitudeHistogram()
    histogram.add(np.array([10, 10]), source="viewer mean4")
    histogram.add(np.array([20]), source="DMA raw")
    assert histogram.sample_count == 1
    assert histogram.frame_count == 1
    assert histogram.counts[32778] == 0
    assert histogram.source == "DMA raw"
    histogram.clear()
    assert not histogram.counts.any()
    assert histogram.sample_count == histogram.frame_count == histogram.rejected_count == 0
    assert histogram.source is None


def test_counter_does_not_silently_overflow() -> None:
    histogram = AccumulatedAmplitudeHistogram()
    histogram.add(np.array([0]), source="viewer")
    histogram.sample_count = int(np.iinfo(np.uint64).max)
    with pytest.raises(OverflowError, match="clear"):
        histogram.add(np.array([0]), source="viewer")
    assert histogram.counts[32768] == 1


def _controller(qtbot: QtBot) -> tuple[ScopeController, MagicMock]:
    scope = MagicMock(spec=Scope)
    scope.specs = PARAMETER_SPECS
    scope.get_trigger_level.return_value = 0
    scope.get_dac_value.return_value = 512
    scope.get_pretrigger_samples.return_value = 32
    scope.get_frame_samples.return_value = 1024
    scope.get_frame_period_cycles.return_value = 0
    scope.get_trigger_mode.return_value = TriggerMode.ANY_BELOW
    scope.supported_trigger_modes.return_value = tuple(TriggerMode)
    scope.frame_period_cycles_supported.return_value = True
    scope.dma_active.return_value = False
    scope.get_enable.return_value = False
    scope.get_viewer_frame_samples_limit.return_value = None
    controller = ScopeController(scope, scope_dma=None, channel=0)
    qtbot.addWidget(controller)
    return controller, scope


def _receive(controller: ScopeController, samples: list[int]) -> None:
    controller._on_frame_received([np.arange(len(samples)) * 8, np.array(samples)])


@pytest.mark.parametrize("mode", [DisplayMode.RAW, DisplayMode.PERSISTENCE])
def test_scope_counts_each_received_frame_and_pauses_when_hidden(qtbot: QtBot, mode: DisplayMode):
    controller, _ = _controller(qtbot)
    controller._set_display_mode(mode)
    _receive(controller, [10, 20])
    assert controller._amplitude_histogram.sample_count == 0
    controller.set_scope_distribution_enabled(True)
    _receive(controller, [10, 10, 20])
    _receive(controller, [20, 30])
    assert controller._amplitude_histogram.frame_count == 2
    assert controller._amplitude_histogram.sample_count == 5
    assert controller._amplitude_histogram.counts[32778:32799:10].tolist() == [2, 2, 1]
    controller._plot_item.setYRange(0, 40, padding=0)
    controller._set_display_mode(DisplayMode.RAW)
    controller.reset_zoom()
    assert controller._amplitude_histogram.sample_count == 5
    controller.set_scope_distribution_enabled(False)
    _receive(controller, [99])
    controller.set_scope_distribution_enabled(True)
    assert controller._amplitude_histogram.sample_count == 5
    controller.clear_scope_distribution()
    assert controller._amplitude_histogram.frame_count == 0
    assert controller._distribution_curve.getData()[0] is None


def test_histogram_geometry_shares_y_and_reclaims_hidden_space(qtbot: QtBot) -> None:
    controller, _ = _controller(qtbot)
    controller.resize(1200, 800)
    controller.show()
    qtbot.wait(30)
    original_width = controller._plot_item.getViewBox().sceneBoundingRect().width()
    controller.set_scope_distribution_enabled(True)
    controller._plot_item.setYRange(90, 120, padding=0)
    _receive(controller, [100, 100, 110])
    qtbot.wait(30)
    main = controller._plot_item.getViewBox()
    side = controller._distribution_plot.getViewBox()
    assert main.sceneBoundingRect().width() < original_width
    np.testing.assert_allclose(main.viewRange()[1], side.viewRange()[1])
    assert abs(main.sceneBoundingRect().top() - side.sceneBoundingRect().top()) < 1
    assert abs(main.sceneBoundingRect().bottom() - side.sceneBoundingRect().bottom()) < 1
    side.setYRange(98, 112, padding=0)
    np.testing.assert_allclose(main.viewRange()[1], side.viewRange()[1])
    x, y = controller._distribution_curve.getData()
    assert x[0] == x[-1] == 0
    assert y[0] == y[-1]
    assert controller._distribution_curve.opts["fillLevel"] == "enclosed"
    assert "3 samples / 1 frames" in controller._distribution_label.text
    controller.set_scope_distribution_enabled(False)
    qtbot.wait(30)
    assert main.sceneBoundingRect().width() >= original_width - 1


def test_single_acquire_counts_once_and_dma_representation_starts_new_histogram(qtbot: QtBot):
    controller, scope = _controller(qtbot)
    controller.set_scope_distribution_enabled(True)
    scope.acquire_frame.return_value = np.array([10, 20, 10], dtype=np.int16)
    controller._on_acquire_frame()
    assert controller._amplitude_histogram.sample_count == 3
    geometry = ScopeDmaGeometry.legacy(1024)
    samples = np.full(geometry.waveform_samples, 40, dtype="<i2")
    preview = ScopeCurrentPreview(
        generation=1, timestamp=10, received_ns=20,
        record=bytes(8) + samples.tobytes(), geometry=geometry,
    )
    controller._on_dma_preview_received(preview)
    assert controller._amplitude_histogram.frame_count == 1
    assert controller._amplitude_histogram.sample_count == geometry.waveform_samples
    assert "DMA preview" in controller._amplitude_histogram.source


def test_replay_clears_live_distribution_without_counting_repeated_playback(qtbot: QtBot):
    controller, _ = _controller(qtbot)
    controller.set_scope_distribution_enabled(True)
    _receive(controller, [10, 20])
    controller.begin_session_replay(frame_count=1, duration_ns=100)
    for _ in range(2):
        controller.display_session_replay_frame(
            np.array([30, 40], dtype=np.int16), sample_period_ns=8,
            frame_index=0, frame_count=1, elapsed_ns=0, duration_ns=100, source="viewer",
        )
    assert controller._amplitude_histogram.sample_count == 0
    controller.end_session_replay()
    _receive(controller, [10])
    assert controller._amplitude_histogram.sample_count == 1


def test_view_menu_and_controller_delegation(qtbot: QtBot) -> None:
    window = QMainWindow()
    qtbot.addWidget(window)
    ui = Ui_MainWindow()
    ui.setupUi(window)
    assert ui.actionScopeDistribution in ui.menuView.actions()
    assert ui.actionClearScopeDistribution in ui.menuView.actions()
    assert ui.actionScopeDistribution.isCheckable()
    assert not ui.actionScopeDistribution.isChecked()
    controllers = [Mock(), Mock()]
    coordinator = SimpleNamespace(_scope_controllers=controllers)
    MainWindowController.set_scope_distribution_enabled(coordinator, True)
    MainWindowController.clear_scope_distributions(coordinator)
    for controller in controllers:
        controller.set_scope_distribution_enabled.assert_called_once_with(True)
        controller.clear_scope_distribution.assert_called_once_with()
    app = SimpleNamespace(_controller=Mock())
    MainAppWindow._on_scope_distribution_toggled(app, True)
    app._controller.set_scope_distribution_enabled.assert_called_once_with(True)


def test_view_preference_round_trips_settings_and_configuration(qtbot: QtBot, monkeypatch):
    owner = QMainWindow()
    qtbot.addWidget(owner)
    ui = Ui_MainWindow()
    ui.setupUi(owner)
    stored = {}

    class Settings:
        def value(self, key, default=None, **_kwargs):
            return stored.get(key, default)

        def setValue(self, key, value):  # noqa: N802
            stored[key] = value

    monkeypatch.setattr(app_module, "QSettings", Settings)
    window = SimpleNamespace(
        ui=ui, _controller=Mock(), _set_log_tab_visible=Mock(),
        _save_developer_settings=Mock(), _apply_view_state=Mock(),
    )
    ui.actionScopeDistribution.setChecked(True)
    MainAppWindow._save_developer_settings(window)
    assert stored["view/scope_distribution"] is True
    ui.actionScopeDistribution.setChecked(False)
    MainAppWindow._restore_developer_settings(window)
    assert ui.actionScopeDistribution.isChecked()
    snapshot = MainAppWindow.configuration_settings(window)
    assert snapshot["scope_distribution"] is True
    MainAppWindow.apply_configuration_settings(window, {"scope_distribution": False})
    assert not ui.actionScopeDistribution.isChecked()
    MainAppWindow._apply_view_state(window)
    window._controller.set_scope_distribution_enabled.assert_called_once_with(False)


def test_wide_zoom_groups_100_adc_codes_and_keeps_rails() -> None:
    histogram = AccumulatedAmplitudeHistogram()
    histogram.add(np.array([-32768, -2, -1, -1, 0, 1, 1, 1, 32767]), source="viewer")
    edges, counts = histogram.projected_bins(-40000, 40000)
    np.testing.assert_array_equal(np.diff(edges), np.full(656, 100))
    assert counts[(32768 - 2) // 100] == 7
    assert counts[0] == counts[-1] == 1
    np.testing.assert_array_equal(edges[[0, -1]], [-32768.5, 32831.5])
    assert int(counts.sum()) == 9


def test_log_counts_preserve_adc_bins_amplitude_zoom_and_stored_counts(qtbot: QtBot) -> None:
    controller, _ = _controller(qtbot)
    controller.set_scope_distribution_enabled(True)
    controller._plot_item.setYRange(-5, 350, padding=0)
    _receive(controller, [-1, 111, 111, 211, 211, 211, 211])
    before_x, before_y = controller._distribution_curve.getData()
    before_x, before_y = before_x.copy(), before_y.copy()
    stored = controller._amplitude_histogram.counts.copy()
    amplitude_range = controller._plot_item.getViewBox().viewRange()[1][:]

    controller._distribution_plot.setLogMode(x=True)
    qtbot.wait(10)

    log_x, log_y = controller._distribution_curve.getData()
    np.testing.assert_allclose(log_x, np.log10(np.maximum(before_x, 0.5)))
    np.testing.assert_array_equal(log_y, before_y)
    assert np.isfinite(log_x).all()
    assert controller._distribution_plot.getAxis("bottom").logMode
    assert controller._distribution_plot.getAxis("top").logMode
    assert not controller._distribution_plot.getAxis("left").logMode
    assert not controller._plot_item.getAxis("left").logMode
    np.testing.assert_allclose(controller._plot_item.getViewBox().viewRange()[1], amplitude_range)
    np.testing.assert_allclose(
        controller._distribution_plot.getViewBox().viewRange()[1], amplitude_range
    )
    np.testing.assert_array_equal(controller._amplitude_histogram.counts, stored)
    assert "Counts (log)" in controller._distribution_label.text
    assert controller._amplitude_histogram.sample_count == 7

    controller._distribution_plot.setLogMode(x=False)
    restored_x, restored_y = controller._distribution_curve.getData()
    np.testing.assert_array_equal(restored_x, before_x)
    np.testing.assert_array_equal(restored_y, before_y)
    np.testing.assert_allclose(controller._plot_item.getViewBox().viewRange()[1], amplitude_range)


def test_log_distribution_updates_and_clears_without_invalid_coordinates(qtbot: QtBot) -> None:
    controller, _ = _controller(qtbot)
    controller.set_scope_distribution_enabled(True)
    controller._plot_item.setYRange(-5, 5, padding=0)
    controller._distribution_plot.setLogMode(x=True)
    for _ in range(2):
        _receive(controller, [0, 0, 100])
    x, y = controller._distribution_curve.getData()
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert np.max(x) == pytest.approx(np.log10(4))
    assert controller._amplitude_histogram.counts[[32768, 32868]].tolist() == [4, 2]
    controller.clear_scope_distribution()
    assert controller._distribution_curve.getData()[0] is None
    assert np.isfinite(controller._distribution_plot.getViewBox().viewRange()).all()
    assert controller._distribution_plot.getAxis("bottom").logMode
    _receive(controller, [0])
    assert controller._amplitude_histogram.sample_count == 1
    assert np.max(controller._distribution_curve.getData()[0]) == 0.0


def test_log_toggle_only_targets_visible_scope_distribution_under_pointer(qtbot: QtBot) -> None:
    controller, _ = _controller(qtbot)
    # Allow for wider text in the headless platform's fallback font.
    controller.resize(1600, 800)
    controller.show()
    controller.set_scope_distribution_enabled(True)
    qtbot.wait(20)
    widget = controller.ui.plotWaveform

    def point_at(item):
        point = widget.mapFromScene(item.sceneBoundingRect().center())
        qtbot.mouseMove(widget.viewport(), point)
        qtbot.wait(5)

    point_at(controller._plot_item)
    assert not controller.toggle_scope_distribution_log_under_mouse()
    assert not controller._distribution_plot.getAxis("bottom").logMode
    point_at(controller._distribution_plot)
    assert controller.toggle_scope_distribution_log_under_mouse()
    assert controller._distribution_plot.getAxis("bottom").logMode
    assert controller.toggle_scope_distribution_log_under_mouse()
    assert not controller._distribution_plot.getAxis("bottom").logMode
    controller.set_scope_distribution_enabled(False)
    assert not controller.toggle_scope_distribution_log_under_mouse()


@pytest.mark.parametrize(
    ("key", "modifiers", "autorepeat", "handled"),
    [
        (Qt.Key.Key_L, Qt.KeyboardModifier.NoModifier, False, True),
        (Qt.Key.Key_L, Qt.KeyboardModifier.ShiftModifier, False, True),
        (Qt.Key.Key_L, Qt.KeyboardModifier.ControlModifier, False, False),
        (Qt.Key.Key_L, Qt.KeyboardModifier.NoModifier, True, False),
        (Qt.Key.Key_K, Qt.KeyboardModifier.NoModifier, False, False),
    ],
)
def test_application_routes_plain_l_to_hovered_distribution(key, modifiers, autorepeat, handled):
    toggle = Mock(return_value=True)
    window = SimpleNamespace(
        isVisible=lambda: True,
        _controller=SimpleNamespace(toggle_scope_distribution_log_under_mouse=toggle),
    )
    event = QKeyEvent(QEvent.Type.KeyPress, key, modifiers, "l", autorepeat)
    assert MainAppWindow._try_scope_distribution_log_shortcut(window, event) is handled
    assert toggle.call_count == int(handled)


def test_only_hovered_channel_toggles_log_distribution() -> None:
    channels = [Mock(), Mock(), Mock()]
    channels[0].toggle_scope_distribution_log_under_mouse.return_value = False
    channels[1].toggle_scope_distribution_log_under_mouse.return_value = True
    coordinator = SimpleNamespace(_scope_controllers=channels)
    assert MainWindowController.toggle_scope_distribution_log_under_mouse(coordinator)
    channels[0].toggle_scope_distribution_log_under_mouse.assert_called_once_with()
    channels[1].toggle_scope_distribution_log_under_mouse.assert_called_once_with()
    channels[2].toggle_scope_distribution_log_under_mouse.assert_not_called()


@pytest.mark.parametrize("negative,expected", [(False, 120), (True, 90)])
def test_peak_mode_counts_one_raw_extremum_per_waveform(negative, expected):
    histogram = AccumulatedAmplitudeHistogram()
    histogram.set_mode(HistogramMode.AMPLITUDE)
    waveform = np.array([100, 110, 90, 120, 100])
    for _ in range(2):
        histogram.add(waveform, source="viewer", negative=negative)
    assert histogram.sample_count == histogram.frame_count == 2
    assert histogram.counts[32768 + expected] == 2
    assert int(histogram.counts.sum()) == 2


def test_derived_mode_polarity_and_window_changes_start_new_counts():
    histogram = AccumulatedAmplitudeHistogram()
    histogram.set_mode(HistogramMode.AMPLITUDE)
    histogram.add(np.array([0, 100, -100]), source="viewer", negative=False)
    histogram.add(np.array([0, 100, -100]), source="viewer", negative=True)
    assert histogram.frame_count == 1
    assert histogram.counts[32768 - 100] == 1
    assert histogram.counts[32768 + 100] == 0
    histogram.add(np.array([0, -50]), source="viewer", negative=True)
    assert histogram.frame_count == 1
    assert histogram.counts[32768 - 50] == 1


@pytest.mark.parametrize("negative,expected", [(False, 64), (True, 32)])
def test_integral_uses_pretrigger_mean_posttrigger_samples_and_selected_side(negative, expected):
    samples = np.array([100, 102, 97, 109, 101])
    assert pulse_integral(
        samples, sample_period_ns=8, pretrigger_ns=16, negative=negative
    ) == expected
    assert pulse_integral(
        samples + 500, sample_period_ns=8, pretrigger_ns=16, negative=negative
    ) == expected


def test_integral_sample_hold_rule_preserves_area_across_equivalent_native_grids():
    viewer = np.array([100, 100, 120, 110, 100])
    native = np.repeat(viewer, 4)
    measured = [pulse_integral(
        values, sample_period_ns=period, pretrigger_ns=16, negative=False
    ) for values, period in [(viewer, 8), (native, 2)]]
    assert measured == [240, 240]


def test_integral_excludes_trigger_straddling_bin_and_rectifies_noise():
    assert pulse_integral(
        np.array([10, 20, 30, 10]), sample_period_ns=8, pretrigger_ns=10, negative=False
    ) == 160
    for negative in (True, False):
        assert pulse_integral(
            np.array([10, 10, 9, 11]), sample_period_ns=8, pretrigger_ns=16, negative=negative
        ) == 8


@pytest.mark.parametrize("pretrigger,period", [(0, 8), (80, 8), (16, 0), (16, np.nan)])
def test_integral_without_usable_baseline_or_timing_is_rejected(pretrigger, period):
    histogram = AccumulatedAmplitudeHistogram()
    histogram.set_mode(HistogramMode.INTEGRAL)
    for _ in range(2):
        histogram.add(np.array([100, 100, 80, 100]), source="viewer", negative=True,
                      sample_period_ns=period, pretrigger_ns=pretrigger)
    assert histogram.sample_count == 0
    # NaN metadata compares unequal, so each invalid context may start fresh.
    assert histogram.rejected_frames >= 1
    assert histogram.last_rejection
    assert not histogram.counts.any()


def test_integral_histogram_grows_range_by_merging_without_losing_events():
    histogram = AccumulatedAmplitudeHistogram()
    histogram.set_mode(HistogramMode.INTEGRAL)
    for height in (1, 32767):
        waveform = np.array([0, 0] + [height] * 240)
        histogram.add(waveform, source="viewer", sample_period_ns=8, pretrigger_ns=16)
    assert histogram.integral_bin_width == 1600
    assert histogram.sample_count == histogram.frame_count == 2
    assert int(histogram.counts.sum()) == 2
    assert histogram.counts.nbytes == 65536 * 8
    assert histogram.counts[int(240 * 8 / 1600)] == 1
    edges, counts = histogram.projected_bins(-32000, 32000)
    assert np.all(np.diff(edges) == 1600)
    assert int(counts.sum()) == 2
    assert edges[-1] > 32767 * 240 * 8


def test_mode_changes_clear_values_but_clear_keeps_selected_mode():
    histogram = AccumulatedAmplitudeHistogram()
    histogram.add(np.array([10, 20]), source="viewer")
    histogram.set_mode(HistogramMode.AMPLITUDE)
    assert histogram.sample_count == 0
    histogram.add(np.array([10, 20]), source="viewer")
    histogram.set_mode(HistogramMode.AMPLITUDE)
    assert histogram.sample_count == 1
    histogram.clear()
    assert histogram.mode == HistogramMode.AMPLITUDE
    assert histogram.sample_count == 0


@pytest.mark.parametrize("mode", [HistogramMode.AMPLITUDE, HistogramMode.INTEGRAL])
def test_event_modes_reject_entire_invalid_waveform(mode):
    histogram = AccumulatedAmplitudeHistogram()
    histogram.set_mode(mode)
    histogram.add(np.array([100, np.nan, 90]), source="viewer",
                  sample_period_ns=8, pretrigger_ns=8)
    assert histogram.sample_count == 0
    assert histogram.rejected_frames == histogram.frame_count == 1


@pytest.mark.parametrize("trigger,expected", [
    (TriggerMode.ANY_ABOVE, 120), (TriggerMode.RISING_EDGE, 120),
    (TriggerMode.ANY_BELOW, 90), (TriggerMode.FALLING_EDGE, 90),
    (TriggerMode.TIMED, 120),
])
def test_scope_peak_mode_uses_selected_trigger_direction(qtbot: QtBot, trigger, expected):
    controller, scope = _controller(qtbot)
    controller.set_scope_distribution_enabled(True)
    controller.set_scope_distribution_mode(HistogramMode.AMPLITUDE)
    blocked = controller.ui.comboTriggerMode.blockSignals(True)
    controller.ui.comboTriggerMode.setCurrentIndex(controller.ui.comboTriggerMode.findData(int(trigger)))
    controller.ui.comboTriggerMode.blockSignals(blocked)
    scope.reset_mock()
    _receive(controller, [100, 110, 90, 120])
    assert controller._amplitude_histogram.sample_count == 1
    assert controller._amplitude_histogram.counts[32768 + expected] == 1
    scope.set_trigger_mode.assert_not_called()
    scope.start.assert_not_called()


def test_scope_integral_mode_has_own_units_and_preserves_main_amplitude_axis(qtbot: QtBot):
    controller, scope = _controller(qtbot)
    controller.set_scope_distribution_enabled(True)
    controller._plot_item.setYRange(50, 150, padding=0)
    controller.set_scope_distribution_mode(HistogramMode.INTEGRAL)
    _receive(controller, [100] * 8 + [90, 110, 80])
    qtbot.wait(10)
    histogram = controller._amplitude_histogram
    assert histogram.sample_count == 1
    assert histogram.counts[int(240 / histogram.integral_bin_width)] == 1
    assert controller._distribution_plot.getAxis("left").isVisible()
    assert "ADC codes·ns" in controller._distribution_plot.getAxis("left").labelText
    np.testing.assert_allclose(controller._plot_item.getViewBox().viewRange()[1], [50, 150])
    controller._distribution_plot.setYRange(0, 1000, padding=0)
    np.testing.assert_allclose(controller._plot_item.getViewBox().viewRange()[1], [50, 150])
    controller._set_distribution_log_counts(True)
    qtbot.wait(10)
    np.testing.assert_allclose(controller._distribution_plot.getViewBox().viewRange()[1], [0, 1000])
    np.testing.assert_allclose(controller._plot_item.getViewBox().viewRange()[1], [50, 150])
    controller.set_scope_distribution_mode(HistogramMode.SAMPLES)
    assert histogram.sample_count == 0
    assert not controller._distribution_plot.getAxis("left").isVisible()
    np.testing.assert_allclose(controller._plot_item.getViewBox().viewRange()[1],
                               controller._distribution_plot.getViewBox().viewRange()[1])


def test_integral_log_before_first_waveform_keeps_auto_range(qtbot: QtBot):
    controller, _ = _controller(qtbot)
    controller.resize(1600, 800)
    controller.show()
    controller.set_scope_distribution_enabled(True)
    controller.set_scope_distribution_mode(HistogramMode.INTEGRAL)
    controller._set_distribution_log_counts(True)
    _receive(controller, [100] * 8 + [-1000] * 8)
    qtbot.wait(10)
    assert controller._distribution_plot.getViewBox().autoRangeEnabled()[1]
    assert controller._distribution_plot.getViewBox().viewRange()[1][1] > 70400


def test_histogram_context_menu_opens_with_exclusive_modes_and_channel_clear(qtbot: QtBot):
    controller, scope = _controller(qtbot)
    controller.resize(1800, 800)
    controller.show()
    controller.set_scope_distribution_enabled(True)
    _receive(controller, [100, 110, 90])
    qtbot.wait(20)
    scope.reset_mock()
    widget = controller.ui.plotWaveform
    position = widget.mapFromScene(
        controller._distribution_plot.getViewBox().sceneBoundingRect().center()
    )
    qtbot.mouseClick(widget.viewport(), Qt.MouseButton.RightButton, pos=position)
    qtbot.waitUntil(controller._distribution_menu.isVisible)
    actions = controller._distribution_mode_actions
    assert set(actions) == set(HistogramMode)
    assert actions[HistogramMode.SAMPLES].isChecked()
    assert controller._distribution_mode_group.isExclusive()
    assert "Transforms" not in [a.text() for a in controller._distribution_menu.actions()]
    actions[HistogramMode.AMPLITUDE].trigger()
    assert controller._amplitude_histogram.mode == HistogramMode.AMPLITUDE
    assert controller._amplitude_histogram.sample_count == 0
    assert not actions[HistogramMode.SAMPLES].isChecked()
    _receive(controller, [100, 110, 90])
    clear = next(a for a in controller._distribution_menu.actions()
                 if a.text() == "Clear histogram")
    clear.trigger()
    assert controller._amplitude_histogram.sample_count == 0
    controller._distribution_log_action.trigger()
    assert controller._distribution_plot.getAxis("bottom").logMode
    controller._distribution_menu.hide()
    scope.start.assert_not_called()
    scope.set_trigger_mode.assert_not_called()
    scope.set_pretrigger_samples.assert_not_called()


def test_histogram_mode_round_trips_channel_configuration(qtbot: QtBot):
    controller, _ = _controller(qtbot)
    controller.set_scope_distribution_mode(HistogramMode.INTEGRAL)
    saved = controller.configuration_settings()
    assert saved["histogram_mode"] == "integral"
    controller.set_scope_distribution_mode(HistogramMode.SAMPLES)
    controller.apply_configuration_settings(saved)
    assert controller._amplitude_histogram.mode == HistogramMode.INTEGRAL
    controller.apply_configuration_settings({"histogram_mode": "future-mode"})
    assert controller._amplitude_histogram.mode == HistogramMode.INTEGRAL


@pytest.mark.parametrize("scope_inverted", [False, True])
@pytest.mark.parametrize("trigger,negative", [
    (TriggerMode.FALLING_EDGE, True), (TriggerMode.ANY_BELOW, True),
    (TriggerMode.RISING_EDGE, False), (TriggerMode.ANY_ABOVE, False),
    (TriggerMode.TIMED, False),
])
def test_integral_axis_follows_scope_pulse_direction(
    qtbot: QtBot, scope_inverted, trigger, negative
):
    controller, _ = _controller(qtbot)
    controller.resize(1600, 800)
    controller.show()
    controller.set_scope_distribution_enabled(True)
    controller.set_scope_distribution_mode(HistogramMode.INTEGRAL)
    controller.ui.comboTriggerMode.setCurrentIndex(
        controller.ui.comboTriggerMode.findData(int(trigger))
    )
    main = controller._plot_item.getViewBox()
    side = controller._distribution_plot.getViewBox()
    main.invertY(scope_inverted)
    main.setYRange(0, 200, padding=0)
    side.setYRange(0, 2000, padding=0)
    qtbot.wait(10)
    assert side.yInverted() == (scope_inverted != negative)
    assert main.yInverted() == scope_inverted
    # Compare actual screen directions of a pulse and a larger positive area.
    pulse_value = 50 if negative else 150
    pulse_direction = (main.mapViewToScene(QPointF(0, pulse_value)).y()
                       - main.mapViewToScene(QPointF(0, 100)).y())
    area_direction = (side.mapViewToScene(QPointF(0, 1500)).y()
                      - side.mapViewToScene(QPointF(0, 500)).y())
    assert pulse_direction * area_direction > 0


def test_integral_direction_updates_immediately_and_other_modes_restore_shared_axis(qtbot: QtBot):
    controller, scope = _controller(qtbot)
    controller.set_scope_distribution_enabled(True)
    controller.set_scope_distribution_mode(HistogramMode.INTEGRAL)
    _receive(controller, [100] * 8 + [90, 80, 100])
    before = controller._amplitude_histogram.counts.copy()
    side = controller._distribution_plot.getViewBox()
    assert side.yInverted()
    for log_counts in (True, False):
        controller._set_distribution_log_counts(log_counts)
        assert side.yInverted()
        np.testing.assert_array_equal(controller._amplitude_histogram.counts, before)
        assert np.min(controller._distribution_curve.getData()[1]) >= 0
    # Changing the trigger updates orientation before any new frame is read.
    controller.ui.comboTriggerMode.setCurrentIndex(
        controller.ui.comboTriggerMode.findData(int(TriggerMode.RISING_EDGE))
    )
    assert not side.yInverted()
    controller.ui.comboTriggerMode.setCurrentIndex(
        controller.ui.comboTriggerMode.findData(int(TriggerMode.FALLING_EDGE))
    )
    assert side.yInverted()
    controller.clear_scope_distribution()
    controller._reset_distribution_zoom()
    assert side.yInverted()
    for mode in (HistogramMode.AMPLITUDE, HistogramMode.SAMPLES):
        controller.set_scope_distribution_mode(mode)
        assert not side.yInverted()
        assert not controller._plot_item.getViewBox().yInverted()
    scope.start.assert_not_called()
