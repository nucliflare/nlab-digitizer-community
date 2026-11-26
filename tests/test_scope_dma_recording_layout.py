"""Scope refuses an NDMA v1 recording of a mean4 layout before arming DMA."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from pytestqt.qtbot import QtBot

from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer.dma import ScopeDmaGeometry
from nlab.hardware.digitizer.offline import create_offline_digitizer

# .128 IP124 channel A in Periodic on 2026-11-26 (read-only iio_info):
# frame_samples 1024, dma_frame_bytes 520, dma_waveform_samples 255,
# dma_sample_decimation 4, i.e. 2 bytes of alignment padding.
MEAN4 = ScopeDmaGeometry(
    frame_samples=1024, buffer_samples=260, frame_bytes=520,
    waveform_samples=255, sample_decimation=4, padding_bytes=2,
)


@pytest.fixture
def streamer() -> Mock:
    return Mock()


@pytest.fixture
def scope(qtbot: QtBot, streamer: Mock) -> Iterator[ScopeController]:
    device = create_offline_digitizer(0)
    controller = ScopeController(device.scope, scope_dma=streamer)
    qtbot.addWidget(controller)
    controller.ui.cbDmaEnable.setChecked(True)
    try:
        yield controller
    finally:
        controller._refresh_timer.stop()
        controller._measurement_timer.stop()
        controller.stop_dma_sync()
        device.close()


def test_mean4_recording_is_refused_with_the_reason_before_arming(
    scope: ScopeController, streamer: Mock, monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamer.capture_geometry.return_value = MEAN4
    start = Mock()
    monkeypatch.setattr(scope, "_start_with_dma", start)
    scope.ui.btnStart.setChecked(True)

    scope._on_start()

    start.assert_not_called()
    assert scope._dma_worker is None
    assert not scope.ui.btnStart.isChecked()
    assert scope.ui.btnStart.isEnabled()
    assert scope.ui.cbDmaEnable.isEnabled()
    text = scope.status.text()
    assert "4-sample means (8 ns per point)" in text
    assert "NDMA v1" in text
    assert "Any above, Any below or an edge trigger" in text
    assert "raw 2 ns frames" in text


def test_raw_layout_still_starts_dma(
    scope: ScopeController, streamer: Mock, monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamer.capture_geometry.return_value = ScopeDmaGeometry.legacy(1024)
    start = Mock()
    monkeypatch.setattr(scope, "_start_with_dma", start)

    scope._on_start()

    start.assert_called_once_with()


def test_current_online_dma_has_no_file_and_is_not_refused(
    scope: ScopeController, streamer: Mock,
) -> None:
    streamer.capture_geometry.return_value = MEAN4
    scope._dma_online_current = True
    assert not scope._refuse_unrecordable_dma()
    scope._dma_online_current = False
    assert scope._refuse_unrecordable_dma()


def test_unreadable_geometry_is_left_to_the_dma_worker(
    scope: ScopeController, streamer: Mock,
) -> None:
    streamer.capture_geometry.side_effect = OSError(110, "timed out")
    assert not scope._refuse_unrecordable_dma()
