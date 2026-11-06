"""Offscreen debug routing against a simulated PetaLinux viewer-mux contract."""

from __future__ import annotations

import errno
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtCore import QSignalBlocker
from pytestqt.qtbot import QtBot

from nlab.controllers.mca_controller import MCAController
from nlab.hardware.digitizer.mca import MultiChannelAnalyzer

if TYPE_CHECKING:
    from nlab.hardware.digitizer.backends.base import MCABackend
    from nlab.workers.dma_workers import IIOMcaDmaWorker
    from nlab.workers.mca_worker import MCAWorker


@dataclass
class _RoutingBackend:
    enabled: bool = False
    dma_active: bool = False
    reject_live_debug: bool = False
    fail_readback: bool = False
    sources: list[int] = field(default_factory=lambda: [0, 1])
    events: list[tuple[str, int | bool]] = field(default_factory=list)

    def get_global_enable(self) -> bool:
        return self.enabled

    def set_global_enable(self, enabled: bool) -> None:
        self.events.append(("enable", enabled))
        self.enabled = enabled

    def mca_dma_active(self) -> bool:
        return self.dma_active

    def _write_source(self, bank: int, value: int) -> None:
        if self.reject_live_debug and (self.enabled or self.dma_active):
            raise OSError(errno.EBUSY, "MCA configuration is locked")
        self.events.append((f"debug{bank}", value))
        self.sources[bank - 1] = value

    def set_mem1_sig_select(self, value: int) -> None:
        self._write_source(1, value)

    def set_mem2_sig_select(self, value: int) -> None:
        self._write_source(2, value)

    def get_mem1_sig_select(self) -> int:
        if self.fail_readback:
            raise OSError(errno.EIO, "Source readback failed")
        return self.sources[0]

    def get_mem2_sig_select(self) -> int:
        if self.fail_readback:
            raise OSError(errno.EIO, "Source readback failed")
        return self.sources[1]


@pytest.fixture
def controller(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[MCAController, _RoutingBackend]]:
    startup_mca = MagicMock()
    startup_mca.get_debug_signal_selectors.return_value = tuple(range(9))
    startup_mca.edge_det_coeff_is_hardware_backed.return_value = False
    monkeypatch.setattr(MCAController, "_load_hardware_state", lambda self: None)
    view = MCAController(startup_mca, channel=0)
    qtbot.addWidget(view)
    backend = _RoutingBackend()
    view._mca = MultiChannelAnalyzer(cast("MCABackend", backend))
    yield view, backend
    view._worker = None
    view._dma_worker = None


@pytest.mark.parametrize("bank", (1, 2))
def test_debug_source_changes_hardware_when_idle(
    controller: tuple[MCAController, _RoutingBackend], bank: int
) -> None:
    view, backend = controller
    combo = getattr(view.ui, f"comboDebug{bank}")
    combo.setCurrentIndex(combo.findData(8))

    assert backend.sources[bank - 1] == 8
    assert backend.events == [(f"debug{bank}", 8)]
    assert backend.enabled is False


@pytest.mark.parametrize("bank", (1, 2))
@pytest.mark.parametrize("dma", (False, True))
def test_debug_source_changes_preserve_accumulation_and_workers(
    controller: tuple[MCAController, _RoutingBackend], bank: int, dma: bool
) -> None:
    view, backend = controller
    backend.enabled = True
    worker = cast("MCAWorker", object())
    view._worker = worker
    dma_worker = cast("IIOMcaDmaWorker", object()) if dma else None
    backend.dma_active = dma
    view._dma_worker = dma_worker
    view._measurement_active = True
    histogram = np.array([1, 4], dtype=np.uint32)
    displayed = histogram.copy()
    view._last_histogram = histogram
    view._display_histogram = displayed
    view._differential_reference_histogram = histogram
    view._last_elapsed_s = 12.0
    view._hist_curve.setData([0, 1, 2], [1, 4])
    view._set_controls_enabled(not dma)
    combo = getattr(view.ui, f"comboDebug{bank}")

    # Selector codes can be sparse: use the stored hardware code, not its row.
    with QSignalBlocker(combo):
        combo.clear()
        for code in (0, 4, 8):
            combo.addItem(str(code), code)
    assert combo.isEnabled()
    combo.setCurrentIndex(combo.findData(8))

    assert backend.sources[bank - 1] == 8
    assert backend.events == [(f"debug{bank}", 8)]
    assert backend.enabled is True
    assert view._worker is worker
    assert view._dma_worker is dma_worker
    assert view.measurement_active is True
    assert view._last_histogram is histogram
    assert view._display_histogram is displayed
    assert view._differential_reference_histogram is histogram
    assert view._last_elapsed_s == 12.0
    x, y = view._hist_curve.getData()
    np.testing.assert_array_equal(x, [0, 1, 2])
    np.testing.assert_array_equal(y, [1, 4])

    # Repeated selection changes must never toggle the acquisition enable.
    combo.setCurrentIndex(combo.findData(4))
    assert backend.sources[bank - 1] == 4
    assert backend.events == [(f"debug{bank}", 8), (f"debug{bank}", 4)]
    assert backend.enabled is True
    assert view._last_histogram is histogram
    assert view._last_elapsed_s == 12.0


@pytest.mark.parametrize("bank", (1, 2))
@pytest.mark.parametrize("interlock", ("replay", "connection_lost"))
def test_debug_sources_preserve_interlocks_even_for_programmatic_selection(
    controller: tuple[MCAController, _RoutingBackend], bank: int, interlock: str
) -> None:
    view, backend = controller
    if interlock == "replay":
        view._replay_hardware_interlocked = True
    else:
        view._connection_lost = True
    view._set_controls_enabled(True)
    combo = getattr(view.ui, f"comboDebug{bank}")
    assert not combo.isEnabled()

    combo.setCurrentIndex(combo.findData(8))

    assert backend.sources == [0, 1]
    assert backend.events == []


def test_dma_keeps_only_debug_sources_available_in_mca_configuration(
    controller: tuple[MCAController, _RoutingBackend],
) -> None:
    view, _backend = controller
    configuration_widgets = (
        view.ui.comboPulsePolarity,
        view.ui.comboBaseline,
        view.ui.spinPileupWindow,
        view.ui.comboBinning,
        view.ui.cbExtTrigger,
        view.ui.groupSignal,
        view.ui.tabFilters,
        view.ui.spinTimeLimit,
        view.ui.cbDmaEnable,
    )
    view._set_controls_enabled(False)
    assert view.ui.comboDebug1.isEnabled()
    assert view.ui.comboDebug2.isEnabled()
    assert all(not widget.isEnabled() for widget in configuration_widgets)

    view._set_controls_enabled(True)
    assert all(widget.isEnabled() for widget in configuration_widgets)


@pytest.mark.parametrize("bank", (1, 2))
@pytest.mark.parametrize("fail_readback", (False, True))
def test_rejected_source_reports_error_without_restarting_measurement(
    controller: tuple[MCAController, _RoutingBackend], bank: int, fail_readback: bool
) -> None:
    view, backend = controller
    backend.enabled = True
    backend.reject_live_debug = True
    backend.fail_readback = fail_readback
    worker = cast("MCAWorker", object())
    view._worker = worker
    histogram = np.array([1, 4], dtype=np.uint32)
    view._last_histogram = histogram
    view._last_elapsed_s = 12.0
    combo = getattr(view.ui, f"comboDebug{bank}")
    original = combo.currentData()

    combo.setCurrentIndex(combo.findData(8))

    assert backend.sources == [0, 1]
    assert backend.events == []
    assert backend.enabled is True
    assert view._worker is worker
    assert view._last_histogram is histogram
    assert view._last_elapsed_s == 12.0
    assert f"Debug {bank} source update failed:" in view.status.text()
    if fail_readback:
        assert "hardware state unknown" in view.status.text()
    else:
        assert combo.currentData() == original


@pytest.mark.parametrize("bank", (1, 2))
def test_empty_debug_selection_does_not_write(
    controller: tuple[MCAController, _RoutingBackend], bank: int
) -> None:
    view, backend = controller
    combo = getattr(view.ui, f"comboDebug{bank}")
    combo.setCurrentIndex(-1)

    assert backend.events == []
