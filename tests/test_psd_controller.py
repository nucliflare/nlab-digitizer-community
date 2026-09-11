from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from pytestqt.qtbot import QtBot

from nlab.controllers.psd_controller import PSDController
from nlab.hardware.digitizer.dma import _LM_EVENT_DTYPE, McaEventBuffer
from nlab.views.plot_viewbox import ModifierZoomViewBox


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


def test_bottom_histogram_hover_l_toggles_log_and_all_plots_use_modifier_zoom(
    qtbot: QtBot,
) -> None:
    controller = PSDController(McaEventBuffer(), channel=0)
    qtbot.addWidget(controller)
    controller.show()

    for widget in (controller.ui.plotPsd, controller.ui.plotRatio, controller.ui.plotEnergy):
        assert isinstance(widget.getViewBox(), ModifierZoomViewBox)

    controller.ui.plotEnergy.underMouse = lambda: True
    key = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_L,
        Qt.KeyboardModifier.NoModifier,
    )

    assert controller.eventFilter(controller.ui.plotEnergy, key)
    assert controller._energy_log_y
    assert controller.configuration_settings()["energy_log_y"] is True

    assert controller.eventFilter(controller.ui.plotEnergy, key)
    assert not controller._energy_log_y
    controller.stop_processing()
