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
    assert "Accepted: 2/3" in controller.status.text()
    controller.stop_processing()


def test_live_psd_matrix_makes_zero_and_one_count_bins_transparent(
    qtbot: QtBot,
) -> None:
    controller = PSDController(McaEventBuffer(), channel=0)
    qtbot.addWidget(controller)
    controller._accumulator.matrix[1, 2] = 1
    controller._accumulator.matrix[3, 4] = 2

    controller._render()

    image = np.asarray(controller._psd_image.image)
    assert np.isnan(image[0, 0])
    assert np.isnan(image[2, 1])
    assert image[4, 3] == np.log1p(2)
    controller.stop_processing()


def test_bottom_histogram_hover_l_toggles_log_and_all_plots_use_modifier_zoom(
    qtbot: QtBot,
) -> None:
    controller = PSDController(McaEventBuffer(), channel=0)
    qtbot.addWidget(controller)
    controller.show()

    for widget in (controller.ui.plotPsd, controller.ui.plotRatio, controller.ui.plotEnergy):
        assert isinstance(widget.getViewBox(), ModifierZoomViewBox)
    assert controller.ui.plotPsd.getViewBox().state["defaultPadding"] == 0.0

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


def test_fom_panel_occupies_bottom_right_and_tracks_live_projection(qtbot: QtBot) -> None:
    controller = PSDController(McaEventBuffer(), channel=0)
    qtbot.addWidget(controller)

    energy_position = controller.ui.plotsLayout.getItemPosition(
        controller.ui.plotsLayout.indexOf(controller.ui.plotEnergy)
    )
    fom_position = controller.ui.plotsLayout.getItemPosition(
        controller.ui.plotsLayout.indexOf(controller._fom_widget)
    )

    assert energy_position == (1, 0, 1, 1)
    assert fom_position == (1, 1, 1, 1)
    np.testing.assert_array_equal(
        controller._fom_widget._ratio,
        controller._accumulator.ratio_centers,
    )
    np.testing.assert_array_equal(
        controller._fom_widget._counts,
        np.zeros(controller._accumulator.ratio_bins),
    )
    controller.stop_processing()


def test_session_replay_rebuilds_psd_and_restores_live_processing(qtbot: QtBot) -> None:
    controller = PSDController(McaEventBuffer(), channel=0)
    qtbot.addWidget(controller)
    original_bins = controller.ui.spinEnergyBins.value()
    events = np.zeros(2, dtype=_LM_EVENT_DTYPE)
    events["trapezoid_energy"] = [1000, 2000]
    events["charge_energy"] = [500, 1000]

    controller.begin_session_replay(events, {"energy_bins": 64, "ratio_bins": 32})

    assert not controller._timer.isActive()
    assert controller._accumulator.statistics.received == 2
    assert controller._accumulator.statistics.accepted == 2
    assert "Session replay in progress" in controller.status.text()

    controller.finish_session_replay()

    assert "Session replay complete" in controller.status.text()

    controller.end_session_replay()

    assert controller._timer.isActive()
    assert controller.ui.spinEnergyBins.value() == original_bins
    controller.stop_processing()
