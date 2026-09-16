from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from examples import dma_scope_periodic as example
from nlab.hardware.digitizer.dma import FILE_HEADER_STRUCT
from nlab.hardware.digitizer.scope import TriggerMode
from nlab.utils.settings_io import write_configuration


def _scope() -> Mock:
    scope = Mock()
    scope.get_enable.return_value = False
    scope.get_dma_enable.return_value = False
    scope.get_frame_samples.return_value = 1024
    scope.get_pretrigger_samples.return_value = 0
    scope.get_frame_period_cycles.return_value = 256
    scope.get_trigger_mode.return_value = TriggerMode.TIMED
    scope.get_trigger_level.return_value = 28000
    scope.get_dac_value.return_value = 256
    scope.dma_fault_is_latched.return_value = False
    return scope


def test_cli_defaults_are_the_tested_raw_dma_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    received: list[example.CaptureOptions] = []
    monkeypatch.setattr(example, "run", lambda options: received.append(options) or 0)

    assert example.main([str(tmp_path / "capture.bin")]) == 0

    options = received[0]
    assert options.uri == "ip:192.168.10.128:30431"
    assert options.channel == 0
    assert options.duration_s == 10
    assert options.frame_samples == 8188
    assert options.gap_cycles == 125
    assert options.pretrigger_samples == 0
    assert options.trigger_mode is TriggerMode.TIMED
    assert options.calibration is None


def test_gui_yaml_is_used_and_explicit_cli_capture_values_override_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "scope.yaml"
    write_configuration(
        config,
        {
            "format_version": 3,
            "hardware": {
                "channels": {
                    "0": {
                        "scope": {
                            "dac_value": 212,
                            "trigger_level": 12000,
                            "frame_ns": 8192,
                            "frame_gap_ns": 2048,
                            "pretrigger_ns": 16,
                            "edge_mode": 2,
                            "dma_enabled": False,
                        }
                    }
                }
            },
        },
    )
    received: list[example.CaptureOptions] = []
    monkeypatch.setattr(example, "run", lambda options: received.append(options) or 0)

    assert example.main([str(tmp_path / "a.bin"), "--config", str(config)]) == 0
    from_yaml = received.pop()
    assert from_yaml.calibration == {"dac_value": 212, "trigger_level": 12000}
    assert from_yaml.frame_samples == 4096
    assert from_yaml.gap_cycles == 256
    assert from_yaml.pretrigger_samples == 8
    assert from_yaml.trigger_mode is TriggerMode.FALLING_EDGE

    assert example.main(
        [
            str(tmp_path / "b.bin"),
            "--config",
            str(config),
            "--frame-ns",
            "16376",
            "--gap-ns",
            "1000",
            "--trigger-mode",
            "periodic",
        ]
    ) == 0
    overridden = received.pop()
    assert overridden.frame_samples == 8188
    assert overridden.gap_cycles == 125
    assert overridden.trigger_mode is TriggerMode.TIMED


def test_yaml_without_calibration_is_rejected_before_connection(tmp_path: Path) -> None:
    config = tmp_path / "incomplete.yaml"
    write_configuration(config, {"scope": {"frame_ns": 16376}})

    with pytest.raises(SystemExit, match="2"):
        example.main([str(tmp_path / "capture.bin"), "--config", str(config)])


@pytest.mark.parametrize(
    ("option", "value"),
    [("--frame-ns", "16378"), ("--gap-ns", "1001"), ("--pretrigger-ns", "7")],
)
def test_cli_rejects_nonhardware_time_steps(
    tmp_path: Path, option: str, value: str
) -> None:
    with pytest.raises(SystemExit, match="2"):
        example.main([str(tmp_path / "capture.bin"), option, value])


def test_record_reports_written_frames_and_stops_core_before_cancel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[str] = []
    output = tmp_path / "capture.bin"
    options = example.CaptureOptions(
        output=output,
        uri="ip:192.0.2.1:30431",
        channel=0,
        duration_s=0.01,
        frame_samples=8188,
        gap_cycles=125,
        pretrigger_samples=0,
        trigger_mode=TriggerMode.TIMED,
        calibration=None,
    )

    def stream_to_file(
        path: Path,
        stop_event: threading.Event,
        *,
        on_ready: object,
        on_progress: object,
    ) -> int:
        assert callable(on_ready)
        assert callable(on_progress)
        on_ready()
        assert stop_event.wait(1)
        path.write_bytes(b"\0" * (FILE_HEADER_STRUCT.size + 2 * 8188 * 2))
        on_progress(2 * 8188 * 2)
        return 2

    scope = SimpleNamespace(stop=lambda: events.append("scope_stop"))
    streamer = SimpleNamespace(
        stream_to_file=stream_to_file,
        request_stop=lambda: events.append("dma_cancel"),
    )
    digitizer = SimpleNamespace(scope=scope, scope_dma=streamer)

    assert example._record(digitizer, options) == 2  # type: ignore[arg-type]
    assert events == ["scope_stop", "dma_cancel"]
    progress = capsys.readouterr().out
    assert "100.0%" in progress
    assert "2 frames" in progress
    assert "32.0 KiB" in progress


def test_run_autosets_only_without_yaml_and_restores_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scope = _scope()
    digitizer = SimpleNamespace(scope=scope, scope_dma=Mock(), close=Mock())
    factory = Mock(return_value=digitizer)
    monkeypatch.setattr(example.Digitizer, "from_iio", factory)
    auto = Mock()
    auto.run.return_value = SimpleNamespace(verified=True)
    procedure = Mock(return_value=auto)
    monkeypatch.setattr(example, "ScopeAutoSetupProcedure", procedure)
    monkeypatch.setattr(
        example,
        "_record",
        lambda _digitizer, options: options.output.write_bytes(b"x") or 2,
    )
    options = example.CaptureOptions(
        output=tmp_path / "auto.bin",
        uri="ip:192.0.2.1:30431",
        channel=0,
        duration_s=1,
        frame_samples=8188,
        gap_cycles=125,
        pretrigger_samples=0,
        trigger_mode=TriggerMode.TIMED,
        calibration=None,
    )

    assert example.run(options) == 0
    procedure.assert_called_once_with(scope)
    scope.set_frame_samples.assert_any_call(8188)
    scope.set_frame_samples.assert_any_call(1024)
    scope.set_frame_period_cycles.assert_any_call(125)
    scope.set_frame_period_cycles.assert_any_call(256)
    digitizer.close.assert_called_once()
    factory.assert_called_once_with(0, "ip:192.0.2.1:30431", with_ids=False)


def test_run_uses_yaml_calibration_without_autosetup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scope = _scope()
    digitizer = SimpleNamespace(scope=scope, scope_dma=Mock(), close=Mock())
    monkeypatch.setattr(example.Digitizer, "from_iio", lambda *_args, **_kwargs: digitizer)
    procedure = Mock(side_effect=AssertionError("Auto Setup must not run"))
    monkeypatch.setattr(example, "ScopeAutoSetupProcedure", procedure)
    monkeypatch.setattr(
        example,
        "_record",
        lambda _digitizer, options: options.output.write_bytes(b"x") or 2,
    )
    options = example.CaptureOptions(
        output=tmp_path / "calibrated.bin",
        uri="ip:192.0.2.1:30431",
        channel=0,
        duration_s=1,
        frame_samples=8188,
        gap_cycles=125,
        pretrigger_samples=0,
        trigger_mode=TriggerMode.TIMED,
        calibration={"dac_value": 200, "trigger_level": 12345},
    )

    assert example.run(options) == 0
    procedure.assert_not_called()
    scope.set_dac_value.assert_any_call(200)
    scope.set_trigger_level.assert_any_call(12345)
    digitizer.close.assert_called_once()


def test_run_refuses_active_scope_without_stopping_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scope = _scope()
    scope.get_enable.return_value = True
    digitizer = SimpleNamespace(scope=scope, scope_dma=Mock(), close=Mock())
    monkeypatch.setattr(example.Digitizer, "from_iio", lambda *_args, **_kwargs: digitizer)
    options = example.CaptureOptions(
        output=tmp_path / "active.bin",
        uri="ip:192.0.2.1:30431",
        channel=0,
        duration_s=1,
        frame_samples=8188,
        gap_cycles=125,
        pretrigger_samples=0,
        trigger_mode=TriggerMode.TIMED,
        calibration=None,
    )

    with pytest.raises(RuntimeError, match="already active"):
        example.run(options)
    scope.stop.assert_not_called()
    digitizer.close.assert_called_once()


def test_run_does_not_overwrite_existing_file_or_connect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "existing.bin"
    output.write_bytes(b"important")
    factory = Mock()
    monkeypatch.setattr(example.Digitizer, "from_iio", factory)
    options = example.CaptureOptions(
        output=output,
        uri="ip:192.0.2.1:30431",
        channel=0,
        duration_s=1,
        frame_samples=8188,
        gap_cycles=125,
        pretrigger_samples=0,
        trigger_mode=TriggerMode.TIMED,
        calibration=None,
    )

    with pytest.raises(FileExistsError):
        example.run(options)
    assert output.read_bytes() == b"important"
    factory.assert_not_called()
