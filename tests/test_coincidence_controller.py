"""Offscreen lifecycle checks; these tests do not contact IIO hardware."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QCheckBox, QSpinBox

from nlab.controllers import coincidence_controller as coincidence_module
from nlab.controllers.coincidence_controller import CoincidenceController
from nlab.controllers.mca_controller import MCAController
from nlab.hardware.digitizer.dma import IIOMcaDmaStreamer
from nlab.hardware.digitizer.mca_capture import McaDmaOutputMode


class _Sync:
    def __init__(self) -> None:
        self.enabled = False
        self.level = 0
        self.source = 0
        self.history: list[tuple[str, int | bool]] = []

    def set_enable(self, value: bool) -> None:
        self.enabled = value
        self.history.append(("enable", value))

    def get_enable(self) -> bool:
        return self.enabled

    def set_sw_trig(self, value: int) -> None:
        self.level = value
        self.history.append(("software", value))

    def get_sw_trig(self) -> int:
        return self.level

    def set_trig_src(self, value: int) -> None:
        self.source = value
        self.history.append(("source", value))

    def get_trig_src(self) -> int:
        return self.source


class _Mca:
    def __init__(self, sync: _Sync) -> None:
        self.sync = sync
        self.ext = False
        self.armed = False
        self.stop = Mock()

    def set_ext_trig_enable(self, value: bool) -> None:
        self.ext = value

    def get_ext_trig_enable(self) -> bool:
        return self.ext

    def get_global_enable(self) -> bool:
        return self.armed


class _McaView(QObject):
    roi_changed = Signal()
    roi_preview_changed = Signal()
    coincidence_ready = Signal(int)
    coincidence_finished = Signal(int)
    coincidence_error = Signal(int, str)
    coincidence_stop_requested = Signal()

    def __init__(
        self,
        channel: int,
        mca: _Mca,
        *,
        fail_start: bool = False,
        auto_ready: bool = True,
    ) -> None:
        super().__init__()
        self.channel = channel
        self._mca = mca
        self.fail_start = fail_start
        self.auto_ready = auto_ready
        self._coincidence_session = False
        self.coincidence_busy = False
        self.coincidence_energy_bin = 2
        self.roi_bounds: tuple[int, int] | None = (100, 200)
        self.coincidence_run_summary = SimpleNamespace(
            continuity="verified", records=0, diagnostics={}, error=None
        )
        self.ui = SimpleNamespace(
            cbDmaEnable=QCheckBox(),
            cbExtTrigger=QCheckBox(),
            spinTimeLimit=QSpinBox(),
        )

    def coincidence_roi(self) -> tuple[int, int] | None:
        return self.roi_bounds

    @property
    def coincidence_active(self) -> bool:
        return self._coincidence_session

    def start_coincidence_capture(self, *_args: object) -> None:
        if self.fail_start:
            raise RuntimeError("simulated arm failure")
        self._coincidence_session = True
        self._mca.armed = True
        if self.auto_ready:
            self.coincidence_ready.emit(self.channel)

    def stop_coincidence_capture(self) -> None:
        self._coincidence_session = False
        self._mca.armed = False
        self.coincidence_finished.emit(self.channel)


def _make_controller(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_second: bool = False,
    delay_second: bool = False,
) -> tuple[CoincidenceController, _Sync, tuple[_McaView, _McaView]]:
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(
        MCAController, "_output_mode", staticmethod(lambda: McaDmaOutputMode.ONLINE)
    )
    sync = _Sync()
    mcas = (_Mca(sync), _Mca(sync))
    views = (
        _McaView(0, mcas[0]),
        _McaView(1, mcas[1], fail_start=fail_second, auto_ready=not delay_second),
    )
    devices = [
        SimpleNamespace(mca=mca, mca_dma=IIOMcaDmaStreamer(Mock(), channel=channel))
        for channel, mca in enumerate(mcas)
    ]
    global_view = SimpleNamespace(set_coincidence_locked=Mock())
    controller = CoincidenceController(devices, list(views), global_view)
    return controller, sync, views


def test_both_dma_readers_ready_before_shared_start(monkeypatch: pytest.MonkeyPatch) -> None:
    controller, sync, views = _make_controller(monkeypatch, delay_second=True)
    controller.start()
    assert controller._state == "arming"
    assert sync.level == 0
    views[1].coincidence_ready.emit(1)
    assert controller._state == "running"
    assert sync.history[:5] == [
        ("enable", False),
        ("software", 0),
        ("source", 0),
        ("enable", True),
        ("software", 1),
    ]
    assert all(view._coincidence_session for view in views)
    controller._begin_stop(None)
    controller.finish_shutdown_sync()
    assert controller._state == "idle"
    assert sync.enabled is False
    assert sync.level == 0


def test_dma_error_during_stop_marks_session_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    controller, _, _ = _make_controller(monkeypatch)
    controller.start()
    controller._begin_stop(None)
    controller._on_channel_error(1, "continuity failed")
    controller.finish_shutdown_sync()
    assert "continuity failed" in controller.status.text()


def test_coincidence_application_settings_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    controller, _, _ = _make_controller(monkeypatch)
    controller.operator.setCurrentText("XOR")
    controller.use_roi[0].setChecked(False)
    controller.low.setValue(-80)
    controller.high.setValue(96)
    controller.offset.setValue(16)
    controller.duration.setValue(30)
    saved = controller.configuration_settings()

    restored, _, _ = _make_controller(monkeypatch)
    restored.apply_configuration_settings(saved)
    assert restored.configuration_settings() == saved


def test_mca_roi_overlay_tracks_selection_and_gate_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _, views = _make_controller(monkeypatch)
    ch0, ch1 = controller._roi_regions
    assert ch0.isVisible() and ch1.isVisible()
    assert ch0.getRegion() == (100, 201)
    assert ch1.getRegion() == (100, 201)
    assert "active gate" in controller.roi_label[0].text()

    views[0].roi_bounds = (210, 260)
    views[0].roi_preview_changed.emit()
    assert ch0.getRegion() == (210, 261)
    assert ch1.getRegion() == (100, 201)

    controller.use_roi[0].setChecked(False)
    assert ch0.isVisible()
    assert "reference only; gate off" in controller.roi_label[0].text()
    assert ch0.lines[0].pen.color().alpha() < ch1.lines[0].pen.color().alpha()

    views[0].roi_bounds = None
    views[0].roi_changed.emit()
    assert not ch0.isVisible()
    assert ch1.isVisible()
    assert "MCA ROI hidden" in controller.roi_label[0].text()


def test_second_arm_failure_disarms_first_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    controller, sync, views = _make_controller(monkeypatch, fail_second=True)
    controller.start()
    controller.finish_shutdown_sync()
    assert controller._state == "idle"
    assert all(not view._coincidence_session for view in views)
    assert sync.level == 0
    assert sync.enabled is False
    assert "simulated arm failure" in controller.status.text()


@pytest.mark.parametrize(
    "mode", [McaDmaOutputMode.BINARY, McaDmaOutputMode.HDF5, McaDmaOutputMode.ROOT]
)
def test_recorded_run_paths_pair_channels_without_overwriting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: McaDmaOutputMode
) -> None:
    controller, _, _ = _make_controller(monkeypatch)
    monkeypatch.setattr(
        coincidence_module,
        "QSettings",
        lambda: SimpleNamespace(value=lambda _key, _default=None: str(tmp_path)),
    )
    controller._session_id = "unit_test"
    first, second = controller._prepare_paths(mode)
    assert first is not None and second is not None
    assert first.name == f"coincidence_unit_test_ch0{mode.extension}"
    assert second.name == f"coincidence_unit_test_ch1{mode.extension}"
    assert controller._session_manifest == tmp_path / "coincidence_unit_test_session.yaml"
    controller._session_manifest.touch()
    with pytest.raises(FileExistsError):
        controller._prepare_paths(mode)
