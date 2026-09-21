from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from nlab.controllers import scope_controller as scope_controller_module
from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer.scope import TriggerMode
from nlab.workers.scope_auto_setup_worker import ScopeAutoSetupProcedure


class _FakePulseScope:
    def __init__(self, polarity: int = -1, amplitude: int = 12_000) -> None:
        self.enabled = False
        self.dma_enabled = False
        self.dac_value = 386
        self.trigger_level = 10_000
        self.trigger_mode = TriggerMode.FALLING_EDGE
        self.frame_samples = 1024
        self.pretrigger_samples = 32
        self.polarity = polarity
        self.amplitude = amplitude

    def get_enable(self) -> bool:
        return self.enabled

    def get_dma_enable(self) -> bool:
        return self.dma_enabled

    def get_dac_value(self) -> int:
        return self.dac_value

    def get_trigger_level(self) -> int:
        return self.trigger_level

    def get_trigger_mode(self) -> TriggerMode:
        return self.trigger_mode

    def get_frame_samples(self) -> int:
        return self.frame_samples

    def get_pretrigger_samples(self) -> int:
        return self.pretrigger_samples

    def start(self) -> None:
        self.enabled = True

    def stop(self) -> None:
        self.enabled = False

    def set_dac_value(self, value: int) -> None:
        self.dac_value = value

    def set_trigger_level(self, value: int) -> None:
        self.trigger_level = value

    def set_trigger_mode(self, mode: TriggerMode) -> None:
        self.trigger_mode = mode

    def set_frame_samples(self, value: int) -> None:
        self.frame_samples = value

    def set_pretrigger_samples(self, value: int) -> None:
        self.pretrigger_samples = value

    def acquire_frame(self) -> np.ndarray:
        sample_count = self.frame_samples // 4
        baseline = (self.dac_value - 512) * 64
        noise = np.resize(np.array([-12, -5, 0, 7, 11], dtype=np.int32), sample_count)
        frame = np.full(sample_count, baseline, dtype=np.int32) + noise
        start = sample_count // 3
        width = max(12, sample_count // 10)
        frame[start : start + width] += self.polarity * self.amplitude
        result: np.ndarray = np.clip(frame, -32768, 32767).astype(np.int16)
        return result


@pytest.mark.parametrize(
    ("polarity", "expected_mode"),
    [
        (-1, TriggerMode.FALLING_EDGE),
        (1, TriggerMode.RISING_EDGE),
    ],
)
def test_auto_setup_centers_pulse_and_selects_edge(
    polarity: int, expected_mode: TriggerMode
) -> None:
    scope = _FakePulseScope(polarity)

    result = ScopeAutoSetupProcedure(scope, sleep=lambda _seconds: None).run()  # type: ignore[arg-type]

    assert result.trigger_mode == expected_mode
    assert result.dac_slope == pytest.approx(64.0, abs=1.0)
    assert scope.trigger_mode == expected_mode
    assert scope.dac_value == result.dac_value
    expected_baseline = 29_490 * -polarity
    assert abs(result.baseline - expected_baseline) < 100
    if polarity < 0:
        assert result.baseline - result.pulse_amplitude < result.trigger_level < result.baseline
    else:
        assert result.baseline < result.trigger_level < result.baseline + result.pulse_amplitude
    assert result.verified
    assert not scope.enabled
    assert scope.frame_samples == 1024
    assert scope.pretrigger_samples == 32


def test_auto_setup_restores_settings_when_no_signal_is_found() -> None:
    scope = _FakePulseScope(amplitude=0)
    original = (
        scope.dac_value,
        scope.trigger_level,
        scope.trigger_mode,
        scope.frame_samples,
        scope.pretrigger_samples,
    )

    with pytest.raises(RuntimeError, match="no clear unipolar pulse"):
        ScopeAutoSetupProcedure(scope, sleep=lambda _seconds: None).run()  # type: ignore[arg-type]

    assert (
        scope.dac_value,
        scope.trigger_level,
        scope.trigger_mode,
        scope.frame_samples,
        scope.pretrigger_samples,
    ) == original
    assert not scope.enabled


class _ThreadProbe:
    def __init__(self, events: list[str], *, exits: bool) -> None:
        self._events = events
        self._exits = exits

    def wait(self, timeout_ms: int) -> bool:
        self._events.append(f"wait:{timeout_ms}")
        return self._exits

    def deleteLater(self) -> None:  # noqa: N802 - mirrors the Qt API
        self._events.append("delete")


class _StatusProbe:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:  # noqa: N802 - mirrors the Qt API
        self.text = text


def _auto_setup_finish_controller(
    thread: _ThreadProbe,
    events: list[str],
) -> SimpleNamespace:
    status = _StatusProbe()
    return SimpleNamespace(
        _auto_setup_thread=thread,
        _auto_setup_worker=object(),
        _auto_setup_result=None,
        _auto_setup_error="test failure",
        _channel=1,
        ui=SimpleNamespace(lblRecordingStatus=status),
        _set_auto_setup_busy=lambda busy: events.append(f"busy:{busy}"),
        _load_hardware_state=lambda: events.append("load"),
        _update_frame_gap_enabled=lambda: events.append("gap"),
        _update_axis_ranges=lambda: events.append("axes"),
        _on_auto_setup_finished=lambda: None,
    )


def test_auto_setup_finish_waits_for_thread_before_releasing_wrappers() -> None:
    events: list[str] = []
    thread = _ThreadProbe(events, exits=True)
    controller = _auto_setup_finish_controller(thread, events)

    ScopeController._on_auto_setup_finished(controller)  # type: ignore[arg-type]

    assert events == ["wait:2000", "busy:False", "load", "gap", "axes", "delete"]
    assert controller._auto_setup_thread is None
    assert controller._auto_setup_worker is None
    assert controller.ui.lblRecordingStatus.text == "Auto Setup failed: test failure"


def test_auto_setup_finish_retries_without_releasing_live_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    thread = _ThreadProbe(events, exits=False)
    controller = _auto_setup_finish_controller(thread, events)
    retries: list[tuple[int, object]] = []
    monkeypatch.setattr(
        scope_controller_module,
        "QTimer",
        SimpleNamespace(singleShot=lambda delay, callback: retries.append((delay, callback))),
    )

    ScopeController._on_auto_setup_finished(controller)  # type: ignore[arg-type]

    assert events == ["wait:2000"]
    assert controller._auto_setup_thread is thread
    assert controller._auto_setup_worker is not None
    assert retries == [(100, controller._on_auto_setup_finished)]
