from __future__ import annotations

import numpy as np
from pytestqt.qtbot import QtBot

from nlab.controllers.psd_controller import PSDController
from nlab.hardware.digitizer.dma import _LM_EVENT_DTYPE, McaEventBuffer


def test_controller_drains_dma_batches_into_live_matrix(qtbot: QtBot) -> None:
    event_buffer = McaEventBuffer()
    controller = PSDController(event_buffer, channel=0)
    qtbot.addWidget(controller)
    controller.begin_capture(True)

    events = np.zeros(3, dtype=_LM_EVENT_DTYPE)
    events["trapezoid_energy"] = [1000, 2000, 0]
    events["charge_energy"] = [500, 1000, 0]
    event_buffer.append(events)

    controller.process_pending_events()

    assert controller._accumulator.statistics.received == 3
    assert controller._accumulator.statistics.accepted == 2
    assert controller._accumulator.statistics.zero_total == 1
    assert controller._accumulator.matrix.sum() == 2
    assert "Accepted: 2/3" in controller.ui.lblStatus.text()
    controller.stop_processing()
