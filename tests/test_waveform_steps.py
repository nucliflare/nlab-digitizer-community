from __future__ import annotations

import struct
import time
from types import SimpleNamespace
from typing import cast

import numpy as np
import pyqtgraph as pg
import pytest
from pytestqt.qtbot import QtBot

from nlab.controllers.current_monitor_controller import CurrentMonitorController
from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer.current_monitor import ScopeCurrentAccumulator, ScopeCurrentRuntime
from nlab.hardware.digitizer.dma import ScopeDmaGeometry
from nlab.hardware.digitizer.mca import MultiChannelAnalyzer
from nlab.views.plot_viewbox import WAVEFORM_STEP_MODE, DraggableScopeCurve
from nlab.workers.current_monitor_worker import CurrentSample


@pytest.mark.parametrize("draggable", [False, True])
def test_sample_hold_path_preserves_timestamps_and_codes(qtbot: QtBot, draggable: bool) -> None:
    item = DraggableScopeCurve() if draggable else pg.PlotDataItem(stepMode=WAVEFORM_STEP_MODE)
    x = np.array([-16.0, -8.0, 0.0, 8.0])
    y = np.array([10, 20, -5, 7], dtype=np.int16)
    item.setData(x, y)
    actual_x, actual_y = item.getData()
    np.testing.assert_array_equal(actual_x, x)
    np.testing.assert_array_equal(actual_y, y)
    curve = item if draggable else item.curve
    path = curve.getPath()
    points = [(path.elementAt(i).x, path.elementAt(i).y) for i in range(path.elementCount())]
    # Qt/pyqtgraph may retain a duplicate zero-length endpoint.
    points = [
        point for index, point in enumerate(points)
        if not index or point != points[index - 1]
    ]
    assert points == [
        (-16.0, 10.0), (-8.0, 10.0), (-8.0, 20.0), (0.0, 20.0),
        (0.0, -5.0), (8.0, -5.0), (8.0, 7.0),
    ]
    # Updating/clearing does not silently return to diagonal interpolation.
    item.clear()
    item.setData(x, y + 1)
    assert curve.opts["stepMode"] == WAVEFORM_STEP_MODE


@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_single_sample_paths_are_safe(qtbot: QtBot, count: int) -> None:
    curve = DraggableScopeCurve()
    curve.setData(np.arange(count, dtype=float), np.ones(count))
    path = curve.getPath()
    assert path.isEmpty() or path.boundingRect().width() == 0


def test_sample_steps_do_not_bridge_missing_dma_fragments(qtbot: QtBot) -> None:
    item = pg.PlotDataItem(stepMode=WAVEFORM_STEP_MODE, connect="finite")
    item.setData([0, 1, 1, 10, 11], [1, 2, np.nan, 3, 4])
    path = item.curve.getPath()
    for i in range(1, path.elementCount()):
        right, left = path.elementAt(i), path.elementAt(i - 1)
        if not right.isMoveTo():
            assert abs(right.x - left.x) <= 1


def test_current_dma_steps_and_iir_trend_restore_independently(qtbot: QtBot) -> None:
    controller = CurrentMonitorController(
        cast(MultiChannelAnalyzer, SimpleNamespace()), channel=0, auto_start=False,
    )
    qtbot.addWidget(controller)
    geometry = ScopeDmaGeometry.legacy(8)
    accumulator = ScopeCurrentAccumulator()
    accumulator.start_session(geometry, ScopeCurrentRuntime(0, 121, 8, 10))
    for timestamp, samples in ((1, [1, 2, 3, 4]), (100, [5, 6, 7, 8])):
        record = struct.pack("<Q", timestamp) + np.asarray(samples, dtype="<i2").tobytes()
        accumulator.append_frame(record, geometry, time.perf_counter_ns())
    controller._render_dma_plot(accumulator.snapshot(), zero=1, scale=2)
    assert controller._curve.opts["stepMode"] == WAVEFORM_STEP_MODE
    assert controller._curve.opts["connect"] == "finite"
    _x, y = controller._curve.getData()
    np.testing.assert_array_equal(y, [0, 2, 4, 6, np.nan, 8, 10, 12, 14])
    controller._sample_buffer.append(CurrentSample(1, time.perf_counter_ns(), 10, 100))
    controller._render_pending()
    controller._display_timer.stop()
    assert controller._curve.opts["stepMode"] is None
    assert controller._curve.opts["connect"] == "auto"


def test_persistence_uses_sample_holds_and_next_sample_transitions() -> None:
    state = SimpleNamespace(
        persistence=0.5, _display_nx=6, _display_ny=5, _Y_MIN=0, _Y_MAX=4,
        _persistence_buffer=np.zeros((6, 5), dtype=np.float32),
    )
    controller = cast(ScopeController, state)
    ScopeController._rasterize_frame(controller, np.array([1, 3, 2]))
    expected = np.zeros((6, 5))
    expected[0, 1] = 1
    expected[1, 1:4] = 1
    expected[2, 2:4] = 1
    np.testing.assert_array_equal(state._persistence_buffer, expected)
    ScopeController._rasterize_frame(controller, np.array([], dtype=np.int16))
    np.testing.assert_array_equal(state._persistence_buffer, expected * 0.5)
    state._persistence_buffer.fill(0)
    ScopeController._rasterize_frame(controller, np.array([2]))
    assert state._persistence_buffer[0, 2] == 1
    assert np.count_nonzero(state._persistence_buffer) == 1
    # Samples beyond the configured frame do not accumulate in its final column.
    state._persistence_buffer.fill(0)
    ScopeController._rasterize_frame(controller, np.array([1] * 6 + [4]))
    np.testing.assert_array_equal(state._persistence_buffer[:, 1], np.ones(6))
    assert not state._persistence_buffer[:, 2:].any()
