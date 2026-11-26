"""Live Scope noise-spectrum panel: guards, resets, settings, export and help."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QMainWindow
from pytestqt.qtbot import QtBot

from nlab import app as app_module
from nlab.analysis.waveform_spectrum import (
    SpectrumAveraging,
    SpectrumUnit,
    SpectrumWindow,
    mean_decimation_response,
)
from nlab.app import MainAppWindow
from nlab.controllers import scope_controller as scope_module
from nlab.controllers.main_window_controller import MainWindowController
from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer.current_monitor import ScopeCurrentPreview
from nlab.hardware.digitizer.dma import ScopeDmaGeometry
from nlab.hardware.digitizer.scope import PARAMETER_SPECS, Scope, TriggerMode
from nlab.help.context import resolve_context, resolve_context_at
from nlab.ui.ui_main_window import Ui_MainWindow

VIEWER = "Scope viewer (8 ns mean4)"


class _Settings:
    """In-memory QSettings stand-in shared by every controller in a test."""

    def __init__(self, store: dict[str, Any]) -> None:
        self._store = store

    def value(self, key: str, default: Any = None, **_kwargs: Any) -> Any:
        return self._store.get(key, default)

    def setValue(self, key: str, value: Any) -> None:  # noqa: N802
        self._store[key] = value


@pytest.fixture
def settings_store(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    store: dict[str, Any] = {}
    monkeypatch.setattr(scope_module, "QSettings", lambda: _Settings(store))
    return store


def _controller(qtbot: QtBot, trigger: TriggerMode = TriggerMode.TIMED) -> tuple[
    ScopeController, MagicMock
]:
    scope = MagicMock(spec=Scope)
    scope.specs = PARAMETER_SPECS
    scope.get_trigger_level.return_value = 0
    scope.get_dac_value.return_value = 512
    scope.get_pretrigger_samples.return_value = 32
    scope.get_frame_samples.return_value = 1024
    scope.get_frame_period_cycles.return_value = 0
    scope.get_trigger_mode.return_value = trigger
    scope.supported_trigger_modes.return_value = tuple(TriggerMode)
    scope.frame_period_cycles_supported.return_value = True
    scope.dma_active.return_value = False
    scope.get_enable.return_value = False
    scope.get_viewer_frame_samples_limit.return_value = None
    controller = ScopeController(scope, scope_dma=None, channel=0)
    qtbot.addWidget(controller)
    return controller, scope


def _frame(seed: int, n: int = 256, tone: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.round(500 + rng.normal(0, 3, n) + tone * np.cos(2 * np.pi * 37 * t / n))


def _receive(controller: ScopeController, samples: np.ndarray) -> None:
    controller._on_frame_received([np.arange(samples.size) * 8, samples])


def _select_trigger(controller: ScopeController, trigger: TriggerMode) -> None:
    combo = controller.ui.comboTriggerMode
    combo.setCurrentIndex(combo.findData(int(trigger)))


def _preview(samples: np.ndarray, decimation: int = 1) -> ScopeCurrentPreview:
    geometry = (
        ScopeDmaGeometry.legacy(samples.size + 4) if decimation == 1 else ScopeDmaGeometry(
            frame_samples=samples.size * decimation, buffer_samples=samples.size + 4,
            frame_bytes=(samples.size + 4) * 2, waveform_samples=samples.size,
            sample_decimation=decimation, padding_bytes=0,
        )
    )
    return ScopeCurrentPreview(
        generation=1, timestamp=10, received_ns=20,
        record=bytes(8) + samples.astype("<i2").tobytes(), geometry=geometry,
    )


def test_frames_accumulate_only_while_enabled(qtbot: QtBot, settings_store) -> None:
    controller, scope = _controller(qtbot)
    scope.reset_mock()
    _receive(controller, _frame(0))
    assert controller._spectrum.frames == 0
    controller.set_scope_spectrum_enabled(True)
    for seed in range(3):
        _receive(controller, _frame(seed))
    assert controller._spectrum.frames == 3
    assert controller._spectrum_curve.getData()[0] is not None
    controller.set_scope_spectrum_enabled(False)
    _receive(controller, _frame(9))
    controller.set_scope_spectrum_enabled(True)
    assert controller._spectrum.frames == 3  # Hidden: paused, not cleared.
    scope.start.assert_not_called()
    scope.set_trigger_mode.assert_not_called()


def test_session_replay_and_drags_add_nothing(qtbot: QtBot, settings_store) -> None:
    controller, _ = _controller(qtbot)
    controller.set_scope_spectrum_enabled(True)
    _receive(controller, _frame(0))
    controller.begin_session_replay(frame_count=1, duration_ns=100)
    assert controller._spectrum.frames == 0
    for _ in range(2):
        controller.display_session_replay_frame(
            _frame(1).astype(np.int16), sample_period_ns=8, frame_index=0,
            frame_count=1, elapsed_ns=0, duration_ns=100, source="viewer",
        )
        controller._accumulate_scope_spectrum(_frame(2), source=VIEWER, sample_period_ns=8)
    assert controller._spectrum.frames == 0
    controller.end_session_replay()
    controller._waveform_drag = Mock()
    controller._accumulate_scope_spectrum(_frame(3), source=VIEWER, sample_period_ns=8)
    assert controller._spectrum.frames == 0
    controller._waveform_drag = None
    _receive(controller, _frame(4))
    assert controller._spectrum.frames == 1


def test_frame_size_trigger_and_dac_changes_restart(qtbot: QtBot, settings_store) -> None:
    controller, _ = _controller(qtbot)
    controller.set_scope_spectrum_enabled(True)

    def fill() -> None:
        for seed in range(2):
            _receive(controller, _frame(seed))
        assert controller._spectrum.frames == 2

    fill()
    controller._on_frame_samples_changed()
    assert controller._spectrum.frames == 0
    assert controller._spectrum.reset_reason == "frame duration changed"
    fill()
    _select_trigger(controller, TriggerMode.RISING_EDGE)
    assert controller._spectrum.frames == 0
    assert controller._spectrum.reset_reason == "trigger mode changed"
    fill()
    controller.ui.spinDacValue.setValue(controller.ui.spinDacValue.value() + 10)
    assert controller._spectrum.frames == 0
    assert controller._spectrum.reset_reason == "DAC baseline changed"
    fill()
    _receive(controller, _frame(7, n=128))
    assert controller._spectrum.frames == 1
    assert "frame length 256 → 128" in controller._spectrum.reset_reason
    assert "Restarted: frame length" in controller._spectrum_label.text()


def test_viewer_and_dma_preview_sources_are_labelled_and_kept_apart(
    qtbot: QtBot, settings_store,
) -> None:
    controller, scope = _controller(qtbot)
    controller.set_scope_spectrum_enabled(True)
    _receive(controller, _frame(0))
    assert controller._spectrum.source == VIEWER
    assert controller._spectrum.sample_period_ns == 8
    scope.acquire_frame.return_value = _frame(1, n=1024 // 4).astype(np.int16)
    controller._on_acquire_frame()
    assert controller._spectrum.frames == 2 and controller._spectrum.source == VIEWER
    # A Periodic DMA preview has the viewer's 8 ns mean4 grid, but is another source.
    controller._on_dma_preview_received(_preview(_frame(2), decimation=4))
    assert controller._spectrum.frames == 1
    assert controller._spectrum.source == "DMA preview (8 ns/sample)"
    assert controller._spectrum.reset_reason == (
        "source Scope viewer (8 ns mean4) → DMA preview (8 ns/sample)"
    )
    controller._on_dma_preview_received(_preview(_frame(3, n=1020)))
    assert controller._spectrum.frames == 1
    assert controller._spectrum.source == "DMA preview (2 ns/sample)"
    assert controller._spectrum.sample_period_ns == 2
    assert "DMA preview (2 ns/sample)" in controller._spectrum_label.text()


def test_annotation_trigger_banner_and_spur_markers(qtbot: QtBot, settings_store) -> None:
    controller, _ = _controller(qtbot, trigger=TriggerMode.FALLING_EDGE)
    controller.set_scope_spectrum_enabled(True)
    _receive(controller, _frame(0, tone=60.0))
    # One frame's noise bins exceed the floor by 10 dB about once a frame.
    assert controller._spectrum_markers.getData()[0] is None
    for seed in range(1, 4):
        _receive(controller, _frame(seed, tone=60.0))
    text = controller._spectrum_label.text()
    for part in ("fs 125 MHz", "N 256", "Δf 488.3 kHz", "ENBW", "K 4/100",
                 "Parseval", "mean4 points: |H| = 0.653", "synchronised to pulses"):
        assert part in text
    x, _y = controller._spectrum_markers.getData()
    assert x is not None and x.size >= 1
    assert x[0] == pytest.approx(37 * 125e6 / 256, abs=125e6 / 256 / 2)
    tooltip = controller._spectrum_plot.toolTip()
    assert "Strongest lines" in tooltip
    assert "not validated on live hardware" in tooltip
    # Plot-item tooltips bypass the widget wrapping: keep them short and wrapped.
    assert max(len(line) for line in tooltip.splitlines()) <= 68
    assert len(tooltip.splitlines()) <= 10
    _select_trigger(controller, TriggerMode.TIMED)
    assert "synchronised to pulses" not in controller._spectrum_label.text()


def test_units_log_axis_and_mean4_compensation(qtbot: QtBot, settings_store) -> None:
    controller, _ = _controller(qtbot)
    controller.set_scope_spectrum_enabled(True)
    _receive(controller, _frame(0))
    frequencies, decibels = controller._spectrum_curve.getData()
    assert decibels is not None and np.isfinite(decibels).all()
    assert frequencies[0] > 0  # The detrended DC bin is not drawn.
    controller.set_scope_spectrum_unit(SpectrumUnit.PSD)
    assert controller._spectrum_plot.getAxis("left").logMode
    assert "codes²/Hz" in controller._spectrum_axis_label.text
    _, linear = controller._spectrum_display or (None, None)
    assert linear is not None
    np.testing.assert_allclose(10 * np.log10(np.maximum(linear, 1e-300)), decibels, atol=1e-9)
    controller.set_scope_spectrum_compensation(True)
    _, compensated = controller._spectrum_display or (None, None)
    assert compensated is not None
    response = mean_decimation_response(frequencies, 4, 2.0)
    np.testing.assert_allclose(compensated, linear / response**2)
    controller.set_scope_spectrum_log_frequency(True)
    assert controller._spectrum_plot.getAxis("bottom").logMode
    controller.set_scope_spectrum_unit(SpectrumUnit.DB)
    assert not controller._spectrum_plot.getAxis("left").logMode


def test_settings_round_trip_through_qsettings_and_configuration(
    qtbot: QtBot, settings_store,
) -> None:
    controller, _ = _controller(qtbot)
    controller.set_scope_spectrum_window(SpectrumWindow.FLATTOP)
    controller.set_scope_spectrum_averaging(SpectrumAveraging.EXPONENTIAL)
    controller.set_scope_spectrum_average_count(30)
    controller.set_scope_spectrum_unit(SpectrumUnit.ASD)
    controller.set_scope_spectrum_log_frequency(True)
    controller.set_scope_spectrum_compensation(True)
    saved = controller.configuration_settings()
    assert saved["spectrum_window"] == "flattop"
    assert saved["spectrum_averaging"] == "exponential"
    assert saved["spectrum_average_count"] == 30
    controller.save_display_settings()
    restored, _ = _controller(qtbot)
    assert restored._spectrum_settings() == controller._spectrum_settings()
    assert restored._spectrum_actions[("window", SpectrumWindow.FLATTOP)].isChecked()
    assert restored._spectrum_log_action.isChecked()
    settings_store.clear()
    fresh, _ = _controller(qtbot)
    assert fresh._spectrum.window == SpectrumWindow.HANN
    fresh.apply_configuration_settings(saved)
    assert fresh._spectrum_settings() == controller._spectrum_settings()
    fresh.apply_configuration_settings({"spectrum_window": "future", "spectrum_unit": "psd"})
    assert fresh._spectrum.window == SpectrumWindow.FLATTOP
    assert fresh._spectrum_unit == SpectrumUnit.PSD


def test_context_menu_resets_and_switches_averaging(qtbot: QtBot, settings_store) -> None:
    controller, scope = _controller(qtbot)
    controller.set_scope_spectrum_enabled(True)
    scope.reset_mock()
    _receive(controller, _frame(0))
    actions = controller._spectrum_actions
    assert actions[("averaging", SpectrumAveraging.LINEAR)].isChecked()
    actions[("averaging", SpectrumAveraging.MAX_HOLD)].trigger()
    assert controller._spectrum.averaging == SpectrumAveraging.MAX_HOLD
    assert controller._spectrum.frames == 1
    actions[("count", 0)].trigger()
    assert controller._spectrum.average_count == 0 and controller._spectrum.frames == 0
    reset = next(a for a in controller._spectrum_menu.actions() if a.text() == "Reset spectrum")
    _receive(controller, _frame(1))
    reset.trigger()
    assert controller._spectrum.frames == 0
    scope.set_frame_samples.assert_not_called()


def test_export_writes_csv_with_channel_and_validation_note(
    qtbot: QtBot, settings_store, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    controller, _ = _controller(qtbot)
    controller.set_scope_spectrum_enabled(True)
    target = tmp_path / "noise"
    monkeypatch.setattr(
        scope_module.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(target), ""),
    )
    assert controller.export_scope_spectrum() is None  # Nothing accumulated yet.
    for seed in range(2):
        _receive(controller, _frame(seed))
    path = controller.export_scope_spectrum()
    assert path == target.with_suffix(".csv")
    text = path.read_text(encoding="utf-8")
    for line in ("# channel: 0", "# source: Scope viewer (8 ns mean4)", "# frames_k: 2",
                 "# trigger_mode: Periodic", "# validation: Software analysis",
                 "frequency_hz,mean_psd,max_psd,min_psd"):
        assert line in text


def test_layout_below_waveform_and_f1_topic(qtbot: QtBot, settings_store) -> None:
    controller, _ = _controller(qtbot)
    controller.resize(1200, 900)
    controller.show()
    qtbot.wait(30)
    waveform_height = controller._plot_item.getViewBox().sceneBoundingRect().height()
    controller.set_scope_spectrum_enabled(True)
    _receive(controller, _frame(0))
    qtbot.wait(30)
    main = controller._plot_item.getViewBox().sceneBoundingRect()
    side = controller._spectrum_plot.sceneBoundingRect()
    assert side.top() >= main.bottom()
    assert main.height() < waveform_height
    view = controller.ui.plotWaveform

    def topic_at(rect: Any) -> str:
        point = view.viewport().mapToGlobal(view.mapFromScene(rect.center()))
        return resolve_context_at(view.viewport(), QPoint(point)).topic

    assert topic_at(side) == "scope-spectrum"
    assert topic_at(main) == "scope"
    label = controller._spectrum_label
    assert label.isVisible() and "Parseval" in label.text()
    assert resolve_context(label).topic == "scope-spectrum"
    # The wrapping annotation never widens the plot column.
    assert controller._scope_plot_layout.sceneBoundingRect().width() <= view.width() + 1

    def data_areas() -> tuple[Any, Any]:
        return (controller._plot_item.getViewBox().sceneBoundingRect(),
                controller._spectrum_plot.getViewBox().sceneBoundingRect())

    # Different tick labels (-30000 versus -15) must not offset the data areas.
    qtbot.waitUntil(lambda: abs(data_areas()[0].left() - data_areas()[1].left()) < 1,
                    timeout=2000)
    assert abs(data_areas()[0].right() - data_areas()[1].right()) < 1
    # Fractional ticks ("27000.2") are wider than "-30000": still aligned.
    controller._plot_item.setYRange(27_000, 27_001, padding=0)
    qtbot.waitUntil(lambda: abs(data_areas()[0].left() - data_areas()[1].left()) < 1,
                    timeout=2000)
    main_axis = controller._plot_item.getAxis("left")
    assert main_axis.fixedWidth is not None
    controller.set_scope_spectrum_enabled(False)
    qtbot.wait(30)
    assert main_axis.fixedWidth is None  # Automatic again once hidden.
    assert controller._plot_item.getViewBox().sceneBoundingRect().height() >= waveform_height - 1


def test_view_menu_preference_and_delegation(qtbot: QtBot, monkeypatch) -> None:
    owner = QMainWindow()
    qtbot.addWidget(owner)
    ui = Ui_MainWindow()
    ui.setupUi(owner)
    assert ui.actionScopeSpectrum in ui.menuView.actions()
    assert ui.actionScopeSpectrum.isCheckable() and not ui.actionScopeSpectrum.isChecked()
    controllers = [Mock(), Mock()]
    MainWindowController.set_scope_spectrum_enabled(
        SimpleNamespace(_scope_controllers=controllers), True
    )
    for controller in controllers:
        controller.set_scope_spectrum_enabled.assert_called_once_with(True)
    stored: dict[str, Any] = {}
    monkeypatch.setattr(app_module, "QSettings", lambda: _Settings(stored))
    window = SimpleNamespace(
        ui=ui, _controller=Mock(), _set_log_tab_visible=Mock(),
        _save_developer_settings=Mock(), _apply_view_state=Mock(),
    )
    MainAppWindow._on_scope_spectrum_toggled(window, True)
    window._controller.set_scope_spectrum_enabled.assert_called_once_with(True)
    ui.actionScopeSpectrum.setChecked(True)
    MainAppWindow._save_developer_settings(window)
    assert stored["view/scope_spectrum"] is True
    ui.actionScopeSpectrum.setChecked(False)
    MainAppWindow._restore_developer_settings(window)
    assert ui.actionScopeSpectrum.isChecked()
    assert MainAppWindow.configuration_settings(window)["scope_spectrum"] is True
    MainAppWindow.apply_configuration_settings(window, {"scope_spectrum": False})
    assert not ui.actionScopeSpectrum.isChecked()
    window._controller.reset_mock()
    MainAppWindow._apply_view_state(window)
    window._controller.set_scope_spectrum_enabled.assert_called_once_with(False)
