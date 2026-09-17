from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMainWindow
from pytestqt.qtbot import QtBot

from nlab.analysis.timing_validation import (
    TimingValidationCancelledError,
    validate_two_channel_timing,
)
from nlab.app import MainAppWindow
from nlab.hardware.digitizer.dma import (
    _LM_EVENT_DTYPE,
    IIO_LM_FILE_VERSION,
    _file_header_bytes,
)
from nlab.hardware.digitizer.mca_capture import McaCaptureWriter, McaDmaOutputMode
from nlab.ui.ui_main_window import Ui_MainWindow
from nlab.views.timing_validation_dialog import TimingValidationDialog

app_module = importlib.import_module("nlab.app")
dialog_module = importlib.import_module("nlab.views.timing_validation_dialog")


def test_timing_validation_is_in_developer_menu(qtbot: QtBot) -> None:
    window = QMainWindow()
    qtbot.addWidget(window)
    ui = Ui_MainWindow()
    ui.setupUi(window)

    assert ui.actionValidateTiming in ui.menuDeveloper.actions()
    assert ui.actionValidateTiming not in ui.menuFile.actions()


def _native_file(path: Path, channel: int, ticks: np.ndarray) -> Path:
    events = np.zeros(1024, dtype=_LM_EVENT_DTYPE)
    events["timestamp"][: len(ticks)] = ticks
    events["trapezoid_energy"][: len(ticks)] = 100
    path.write_bytes(_file_header_bytes(channel, version=IIO_LM_FILE_VERSION) + events.tobytes())
    return path


def test_split_pulse_files_reveal_delay_and_manual_offset(tmp_path: Path) -> None:
    ticks = np.arange(1000, 21000, 200, dtype=np.uint64)
    path_a = _native_file(tmp_path / "ch0.bin", 0, ticks)
    path_b = _native_file(tmp_path / "ch1.bin", 1, ticks + 5)

    raw = validate_two_channel_timing(path_a, path_b, search_window_ns=80)
    corrected = validate_two_channel_timing(path_a, path_b, offset_ns=-40, search_window_ns=80)

    assert raw.channel_a.usable_events == len(ticks)
    assert raw.channel_a.zero_timestamps == 1024 - len(ticks)
    assert raw.channel_b.channel == 1
    assert raw.same_device_metadata is None
    assert raw.matched_a >= len(ticks) - 2
    assert raw.peak_delay_ns == pytest.approx(40, abs=4)
    assert corrected.peak_delay_ns == pytest.approx(0, abs=4)


def test_time_reversal_rejects_pairing(tmp_path: Path) -> None:
    path_a = _native_file(tmp_path / "ch0.bin", 0, np.array([100, 200, 150]))
    path_b = _native_file(tmp_path / "ch1.bin", 1, np.array([100, 200, 300]))

    with pytest.raises(ValueError, match="go backwards"):
        validate_two_channel_timing(path_a, path_b)


def test_missing_overlap_and_same_channel_are_rejected(tmp_path: Path) -> None:
    path_a = _native_file(tmp_path / "ch0.bin", 0, np.array([100, 200]))
    path_b = _native_file(tmp_path / "ch1.bin", 1, np.array([1000, 1100]))
    duplicate = _native_file(tmp_path / "ch0copy.bin", 0, np.array([100, 200]))

    with pytest.raises(ValueError, match="do not overlap"):
        validate_two_channel_timing(path_a, path_b)
    with pytest.raises(ValueError, match="different channel"):
        validate_two_channel_timing(path_a, duplicate)
    with pytest.raises(ValueError, match="lower-index channel"):
        validate_two_channel_timing(path_b, path_a)


def test_cancel_and_invalid_tick_step(tmp_path: Path) -> None:
    path_a = _native_file(tmp_path / "ch0.bin", 0, np.array([100, 200]))
    path_b = _native_file(tmp_path / "ch1.bin", 1, np.array([101, 201]))

    with pytest.raises(TimingValidationCancelledError):
        validate_two_channel_timing(path_a, path_b, cancelled=lambda: True)
    with pytest.raises(ValueError, match="8 ns increments"):
        validate_two_channel_timing(path_a, path_b, offset_ns=1)


def test_hdf5_native_metadata_is_accepted(tmp_path: Path) -> None:
    ticks = np.arange(1000, 3000, 200, dtype=np.uint64)
    path_a = _native_file(tmp_path / "ch0.bin", 0, ticks)
    path_b = tmp_path / "ch1.h5"
    writer = McaCaptureWriter(
        path=path_b,
        mode=McaDmaOutputMode.HDF5,
        configuration_yaml=(
            "connection:\n  backend: iio\n  ip: 192.0.2.1\nmeasurement:\n  channel: 1\n"
        ),
        binary_header=b"",
    )
    events = np.zeros(len(ticks), dtype=_LM_EVENT_DTYPE)
    events["timestamp"] = ticks + 5
    writer.append(b"", events)
    writer.close(complete=True)

    result = validate_two_channel_timing(path_a, path_b, search_window_ns=80)

    assert result.channel_b.channel == 1
    assert result.peak_delay_ns == pytest.approx(40, abs=4)


def test_root_native_metadata_is_accepted(tmp_path: Path) -> None:
    pytest.importorskip("uproot")
    ticks = np.arange(1000, 3000, 200, dtype=np.uint64)
    path_a = _native_file(tmp_path / "ch0.bin", 0, ticks)
    path_b = tmp_path / "ch1.root"
    writer = McaCaptureWriter(
        path=path_b,
        mode=McaDmaOutputMode.ROOT,
        configuration_yaml=(
            "connection:\n  backend: iio\n  ip: 192.0.2.1\nmeasurement:\n  channel: 1\n"
        ),
        binary_header=b"",
    )
    events = np.zeros(len(ticks), dtype=_LM_EVENT_DTYPE)
    events["timestamp"] = ticks + 5
    writer.append(b"", events)
    writer.close(complete=True)

    result = validate_two_channel_timing(path_a, path_b, search_window_ns=80)

    assert result.channel_b.channel == 1
    assert result.peak_delay_ns == pytest.approx(40, abs=4)


def test_structured_capture_without_iio_identity_is_rejected(tmp_path: Path) -> None:
    ticks = np.array([100, 200, 300], dtype=np.uint64)
    path_a = _native_file(tmp_path / "ch0.bin", 0, ticks)
    path_b = tmp_path / "ch1.h5"
    writer = McaCaptureWriter(
        path=path_b,
        mode=McaDmaOutputMode.HDF5,
        configuration_yaml="measurement:\n  channel: 1\n",
        binary_header=b"",
    )
    events = np.zeros(len(ticks), dtype=_LM_EVENT_DTYPE)
    events["timestamp"] = ticks + 1
    writer.append(b"", events)
    writer.close(complete=True)

    with pytest.raises(ValueError, match="IIO capture configuration"):
        validate_two_channel_timing(path_a, path_b)


def test_mismatched_device_configurations_are_rejected(tmp_path: Path) -> None:
    ticks = np.array([100, 200, 300], dtype=np.uint64)
    path_a = _native_file(tmp_path / "ch0.bin", 0, ticks)
    path_b = _native_file(tmp_path / "ch1.bin", 1, ticks + 1)
    path_a.with_suffix(".yaml").write_text(
        "connection:\n  backend: iio\n  ip: 192.0.2.1\n", encoding="utf-8"
    )
    path_b.with_suffix(".yaml").write_text(
        "connection:\n  backend: iio\n  ip: 192.0.2.2\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="different digitizer endpoints"):
        validate_two_channel_timing(path_a, path_b)


def test_matching_device_configurations_are_reported(tmp_path: Path) -> None:
    ticks = np.array([100, 200, 300], dtype=np.uint64)
    path_a = _native_file(tmp_path / "ch0.bin", 0, ticks)
    path_b = _native_file(tmp_path / "ch1.bin", 1, ticks + 1)
    config = "connection:\n  backend: iio\n  ip: 192.0.2.1\n"
    path_a.with_suffix(".yaml").write_text(config, encoding="utf-8")
    path_b.with_suffix(".yaml").write_text(config, encoding="utf-8")

    result = validate_two_channel_timing(path_a, path_b)

    assert result.same_device_metadata is True


def test_timing_dialog_has_background_analysis_controls(qtbot: QtBot, qapp: QApplication) -> None:
    dialog = TimingValidationDialog()
    qtbot.addWidget(dialog)

    assert dialog.offset_ns.singleStep() == 8
    assert dialog.search_window_ns.singleStep() == 8
    assert dialog.analyze_button.isEnabled()
    dialog._start()
    assert "Select two existing" in dialog.status.text()


def test_timing_dialog_processes_files_off_gui_thread(qtbot: QtBot, tmp_path: Path) -> None:
    ticks = np.arange(1000, 5000, 200, dtype=np.uint64)
    path_a = _native_file(tmp_path / "ch0.bin", 0, ticks)
    path_b = _native_file(tmp_path / "ch1.bin", 1, ticks + 5)
    dialog = TimingValidationDialog()
    qtbot.addWidget(dialog)
    dialog.path_a.setText(str(path_a))
    dialog.path_b.setText(str(path_b))

    dialog._start()
    assert not dialog.analyze_button.isEnabled()
    qtbot.waitUntil(lambda: dialog._thread is None, timeout=5000)

    assert dialog.analyze_button.isEnabled()
    assert "strongest bin" in dialog.status.text()
    assert "Ch 0:" in dialog.status.text()


def test_timing_offset_survives_application_settings_round_trip(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    values: dict[str, object] = {}

    class FakeSettings:
        def value(self, key: str, default: object = None, *, type: type | None = None) -> object:
            value = values.get(key, default)
            return type(value) if type is not None else value

        def setValue(self, key: str, value: object) -> None:  # noqa: N802
            values[key] = value

    def fake_setup(window: MainAppWindow) -> None:
        window.ui = SimpleNamespace(
            actionShowSystemLog=QAction(window),
            actionDebugMode=QAction(window),
            actionShowRoi=QAction(window),
            actionLogY=QAction(window),
            mainTabs=SimpleNamespace(
                currentIndex=lambda: 0, count=lambda: 1, setCurrentIndex=Mock()
            ),
        )

    monkeypatch.setattr(app_module, "QSettings", FakeSettings)
    monkeypatch.setattr(dialog_module, "QSettings", FakeSettings)
    monkeypatch.setattr(MainAppWindow, "_setup_ui", fake_setup)
    monkeypatch.setattr(MainAppWindow, "_apply_view_state", lambda _self: None)
    monkeypatch.setattr(MainAppWindow, "_save_developer_settings", lambda _self: None)
    monkeypatch.setattr(
        app_module,
        "MainWindowController",
        lambda *_args, **_kwargs: SimpleNamespace(
            refresh_dma_output_settings=Mock(), shutdown=Mock()
        ),
    )
    window = MainAppWindow(backend="iio")
    qtbot.addWidget(window)

    window.apply_configuration_settings({"timing_channel_b_offset_ns": 40})
    assert window.configuration_settings()["timing_channel_b_offset_ns"] == 40
    dialog = TimingValidationDialog(window)
    qtbot.addWidget(dialog)
    assert dialog.offset_ns.value() == 40
    dialog.offset_ns.setValue(48)
    assert window.configuration_settings()["timing_channel_b_offset_ns"] == 48
