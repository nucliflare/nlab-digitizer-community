from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
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


def test_matrix_scale_switch_changes_colors_only_and_preserves_pending_events(
    qtbot: QtBot,
) -> None:
    event_buffer = McaEventBuffer()
    controller = PSDController(event_buffer, channel=0)
    qtbot.addWidget(controller)
    controller._timer.stop()
    accumulator = controller._accumulator
    accumulator.matrix[1, 2] = 1
    accumulator.matrix[3, 4] = 2
    accumulator.matrix[5, 6] = 100
    controller._render()
    matrix = accumulator.matrix.copy()
    controller._psd_plot.setRange(xRange=(100, 2000), yRange=(-0.2, 0.8), padding=0)
    view_range = controller._psd_plot.viewRange()
    roi = controller._energy_roi.getRegion()
    cut = controller.ui.spinCut.value()
    controller._render_projections = Mock(wraps=controller._render_projections)
    controller._fom_widget.set_projection = Mock(wraps=controller._fom_widget.set_projection)
    controller._fom_widget.clear_fit = Mock(wraps=controller._fom_widget.clear_fit)
    events = np.zeros(2, dtype=_LM_EVENT_DTYPE)
    events["trapezoid_energy"] = 1000
    events["charge_energy"] = 500
    event_buffer.append(events)

    for scale in ("linear", "log", "linear"):
        controller.ui.comboMatrixScale.setCurrentIndex(
            controller.ui.comboMatrixScale.findData(scale)
        )
        image = np.asarray(controller._psd_image.image)
        assert np.isnan(image[0, 0]) and np.isnan(image[2, 1])
        transform = np.log1p if scale == "log" else float
        assert image[4, 3] == transform(2)
        assert image[6, 5] == transform(100)
        np.testing.assert_allclose(controller._psd_image.getLevels(), (0, transform(100)))
        assert controller._accumulator is accumulator
        np.testing.assert_array_equal(accumulator.matrix, matrix)
        np.testing.assert_allclose(controller._psd_plot.viewRange(), view_range)
        assert controller._energy_roi.getRegion() == roi
        assert controller.ui.spinCut.value() == cut
        assert not controller._energy_log_y
    controller._render_projections.assert_not_called()
    controller._fom_widget.set_projection.assert_not_called()
    controller._fom_widget.clear_fit.assert_not_called()
    # Switching scale did not clear the as-yet unprocessed DMA batch.
    controller.process_pending_events()
    assert accumulator.statistics.accepted == 2
    assert accumulator.matrix.sum() == matrix.sum() + 2
    controller.stop_processing()


@pytest.mark.parametrize("settings", [{}, {"matrix_scale": "log"}, {"matrix_scale": "bad"}])
def test_matrix_scale_defaults_and_older_settings_stay_logarithmic(
    qtbot: QtBot, settings: dict[str, object],
) -> None:
    controller = PSDController(McaEventBuffer(), channel=0)
    qtbot.addWidget(controller)
    assert controller.ui.comboMatrixScale.currentData() == "log"
    controller.ui.comboMatrixScale.setCurrentIndex(1)
    controller.apply_configuration_settings(settings)
    assert controller.ui.comboMatrixScale.currentData() == "log"
    assert controller.configuration_settings()["matrix_scale"] == "log"
    # Both empty images have valid levels, with transparent empty bins.
    for index in (0, 1):
        controller.ui.comboMatrixScale.setCurrentIndex(index)
        assert np.isnan(controller._psd_image.image).all()
        np.testing.assert_array_equal(controller._psd_image.getLevels(), (0.0, 1.0))
    controller.stop_processing()


def test_matrix_scale_settings_round_trip_and_replay_restores_previous_scale(qtbot: QtBot) -> None:
    controller = PSDController(McaEventBuffer(), channel=0)
    qtbot.addWidget(controller)
    controller.ui.comboMatrixScale.setCurrentIndex(1)
    controller.set_energy_log_y(True)
    saved = controller.configuration_settings()
    assert saved["matrix_scale"] == "linear"
    controller.ui.comboMatrixScale.setCurrentIndex(0)
    controller.apply_configuration_settings(saved)
    assert controller.ui.comboMatrixScale.currentData() == "linear"
    assert controller._energy_log_y

    controller.begin_session_replay(settings={"matrix_scale": "log"})
    reconfigure = Mock()
    controller.use_session_replay_worker(reconfigure)
    accumulator = controller.session_replay_accumulator()
    accumulator.matrix[3, 4] = 10
    controller.display_session_replay_snapshot(accumulator, True)
    controller.ui.comboMatrixScale.setCurrentIndex(1)
    assert controller._psd_image.image[4, 3] == 10
    reconfigure.assert_not_called()  # No replay/rebin job for a color change.
    controller.ui.comboMatrixScale.setCurrentIndex(0)
    assert controller._psd_image.image[4, 3] == np.log1p(10)
    controller.end_session_replay()
    assert controller.ui.comboMatrixScale.currentData() == "linear"
    assert controller._energy_log_y
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


def test_worker_replay_defers_hidden_psd_and_rebins_asynchronously(qtbot: QtBot) -> None:
    controller = PSDController(McaEventBuffer(), channel=0)
    qtbot.addWidget(controller)
    controller.begin_session_replay(settings={"energy_bins": 64, "ratio_bins": 32})
    callback = Mock()
    controller.use_session_replay_worker(callback)
    accumulator = controller.session_replay_accumulator()
    events = np.zeros(2, dtype=_LM_EVENT_DTYPE)
    events["trapezoid_energy"] = [1000, 2000]
    events["charge_energy"] = [500, 1000]
    accumulator.add_events(events)
    render = Mock(wraps=controller._render)
    controller._render = render

    controller.display_session_replay_snapshot(accumulator.snapshot(), True)

    render.assert_not_called()
    assert controller._accumulator.statistics.received == 2
    controller.show()
    assert render.call_count == 1
    controller.hide()
    controller.ui.spinEnergyBins.setValue(128)
    controller._on_analysis_settings_changed()
    callback.assert_called_once()
    requested = callback.call_args.args[0]
    assert requested.energy_bins == 128
    assert requested.statistics.received == 0
    # Reject the old configuration while worker re-binning is pending.
    controller.display_session_replay_snapshot(accumulator.snapshot(), True)
    assert controller._accumulator.energy_bins == 128
    requested.add_events(events)
    controller.display_session_replay_snapshot(requested.snapshot(), True)
    assert controller._accumulator.statistics.received == 2
    controller.end_session_replay()
    assert controller._session_replay_reconfigure is None
    controller.stop_processing()
