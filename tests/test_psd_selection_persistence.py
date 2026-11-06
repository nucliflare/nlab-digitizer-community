"""PSD guide persistence with generated events and offscreen Qt widgets."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import Mock

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from nlab.controllers.psd_controller import PSDController
from nlab.hardware.digitizer.dma import _LM_EVENT_DTYPE, McaEventBuffer
from nlab.views.offline_psd_plot import OfflinePsdPlot

_SELECTION_KEYS = (
    "ratio_cut",
    "energy_roi",
    "fom_fit_roi",
    "fom_centroid_seeds",
    "fom_guides_visible",
    "fom_use_background",
    "energy_log_y",
)
_ANALYSIS_CONTROLS = (
    "spinEnergyBins",
    "spinRatioBins",
    "spinEnergyShift",
    "spinRatioMin",
    "spinRatioMax",
)


@pytest.fixture
def controller(qtbot: QtBot) -> Iterator[PSDController]:
    view = PSDController(McaEventBuffer(), channel=0)
    qtbot.addWidget(view)
    view._timer.stop()
    view.ui.spinCut.setValue(0.2345)
    view._energy_roi.setRegion((5000.0, 20000.0))
    view._fom_widget.fit_region.setRegion((-0.25, 0.7))
    view._fom_widget.centroid_lines[0].setPos(0.05)
    view._fom_widget.centroid_lines[1].setPos(0.4)
    view._fom_widget.guides_checkbox.setChecked(False)
    view.set_energy_log_y(True)
    yield view
    view.stop_processing()


def _assert_selections(view: PSDController, expected: dict[str, object]) -> None:
    current = view.configuration_settings()
    for key in _SELECTION_KEYS:
        assert current[key] == pytest.approx(expected[key]), key
    assert view._cut_line.value() == pytest.approx(expected["ratio_cut"])
    assert view._ratio_cut_line.value() == pytest.approx(expected["ratio_cut"])


def _events() -> np.ndarray:
    events = np.zeros(2, dtype=_LM_EVENT_DTYPE)
    events["trapezoid_energy"] = [10000, 20000]
    events["charge_energy"] = [6000, 10000]
    return events


@pytest.mark.parametrize(
    "control,value",
    [
        ("spinEnergyBins", 2048),
        ("spinRatioBins", 512),
        ("spinEnergyShift", 1),
        ("spinRatioMin", -0.5),
        ("spinRatioMax", 0.9),
    ],
)
def test_settings_edits_preserve_cut_roi_and_fit_guides(
    controller: PSDController, control: str, value: float
) -> None:
    expected = controller.configuration_settings()
    widget = getattr(controller.ui, control)
    widget.setValue(value)
    widget.editingFinished.emit()

    _assert_selections(controller, expected)


def test_unchanged_field_commits_preserve_accumulated_events(controller: PSDController) -> None:
    controller._event_buffer.append(_events())
    controller.process_pending_events()
    accumulator = controller._accumulator
    matrix = accumulator.matrix.copy()
    expected = controller.configuration_settings()
    for name in _ANALYSIS_CONTROLS:
        getattr(controller.ui, name).editingFinished.emit()

    assert controller._accumulator is accumulator
    assert accumulator.statistics.received == 2
    np.testing.assert_array_equal(accumulator.matrix, matrix)
    _assert_selections(controller, expected)


@pytest.mark.parametrize("control,value", (("spinEnergyBins", 2048), ("spinRatioBins", 512)))
def test_rebinning_preserves_matrix_zoom(
    controller: PSDController, control: str, value: int
) -> None:
    controller._psd_plot.setRange(xRange=(1000, 25000), yRange=(-0.3, 0.8), padding=0)
    expected = controller._psd_plot.viewRange()
    widget = getattr(controller.ui, control)
    widget.setValue(value)
    widget.editingFinished.emit()

    np.testing.assert_allclose(controller._psd_plot.viewRange(), expected)


@pytest.mark.parametrize("settings", ({}, {"energy_bins": 2048}, {"matrix_scale": "linear"}))
def test_partial_configuration_preserves_unspecified_selections(
    controller: PSDController, settings: dict[str, object]
) -> None:
    controller.apply_configuration_settings(
        {"ratio_range": [-0.5, 0.9], "energy_right_shift": 1, "ratio_bins": 512}
    )
    expected = controller.configuration_settings()

    controller.apply_configuration_settings(settings)

    _assert_selections(controller, expected)
    assert controller._accumulator.ratio_range == (-0.5, 0.9)
    assert controller._accumulator.energy_right_shift == 1
    assert controller._accumulator.ratio_bins == 512


def test_narrower_domains_clamp_positions_instead_of_resetting(controller: PSDController) -> None:
    controller.apply_configuration_settings(
        {"ratio_range": [0.1, 0.3], "energy_right_shift": 3}
    )

    assert controller.ui.spinCut.value() == pytest.approx(0.2345)
    assert controller._cut_line.value() == pytest.approx(0.2345)
    assert controller._energy_roi.getRegion() == pytest.approx((5000, 8192))
    assert controller._fom_widget.fit_region.getRegion() == pytest.approx((0.1, 0.3))
    assert [line.value() for line in controller._fom_widget.centroid_lines] == pytest.approx(
        (0.1, 0.3)
    )


def test_color_only_configuration_updates_image_without_rebinning(
    controller: PSDController,
) -> None:
    accumulator = controller._accumulator
    accumulator.matrix[1, 2] = 4
    controller._render()
    expected = controller.configuration_settings()

    controller.apply_configuration_settings({"matrix_scale": "linear"})

    assert controller._psd_image.image[2, 1] == 4
    assert controller._accumulator is accumulator
    _assert_selections(controller, expected)


def test_outside_selections_stay_at_boundary_when_domain_expands(controller: PSDController) -> None:
    controller.ui.spinCut.setValue(0.7)
    controller._fom_widget.fit_region.setRegion((0.5, 0.8))
    controller._fom_widget.centroid_lines[0].setPos(0.6)
    controller._fom_widget.centroid_lines[1].setPos(0.7)
    controller.apply_configuration_settings({"ratio_range": [-0.2, 0.2]})

    assert controller._fom_widget.fit_region.getRegion() == pytest.approx((0.2, 0.2))
    assert [line.value() for line in controller._fom_widget.centroid_lines] == pytest.approx(
        (0.2, 0.2)
    )
    expected = controller.configuration_settings()
    controller.apply_configuration_settings({"ratio_range": [-1.0, 1.0]})
    _assert_selections(controller, expected)


def test_explicit_saved_positions_restore_after_new_bounds(controller: PSDController) -> None:
    controller.apply_configuration_settings(
        {
            "ratio_range": [2.0, 4.0],
            "ratio_cut": 2.75,
            "energy_roi": [200, 700],
            "fom_fit_roi": [2.2, 3.8],
            "fom_centroid_seeds": [2.6, 3.3],
        }
    )

    assert controller.ui.spinCut.value() == pytest.approx(2.75)
    assert controller._cut_line.value() == pytest.approx(2.75)
    assert controller._energy_roi.getRegion() == pytest.approx((200, 700))
    assert controller._fom_widget.fit_region.getRegion() == pytest.approx((2.2, 3.8))
    assert [line.value() for line in controller._fom_widget.centroid_lines] == pytest.approx(
        (2.6, 3.3)
    )
    saved = controller.configuration_settings()
    controller.apply_configuration_settings({"ratio_range": [-4.0, -2.0]})
    controller.apply_configuration_settings(saved)
    _assert_selections(controller, saved)


def test_replay_rebin_and_delayed_snapshots_keep_selected_positions(
    controller: PSDController,
) -> None:
    expected = controller.configuration_settings()
    controller.begin_session_replay()
    previous = controller.session_replay_accumulator()
    reconfigure = Mock()
    controller.use_session_replay_worker(reconfigure)
    controller.ui.spinRatioBins.setValue(512)
    controller.ui.spinRatioBins.editingFinished.emit()
    reconfigure.assert_called_once()
    controller.display_session_replay_snapshot(previous.snapshot(), True)
    assert controller._accumulator.ratio_bins == 512
    updated = reconfigure.call_args.args[0]
    updated.add_events(_events())
    controller.display_session_replay_snapshot(updated.snapshot(), True)
    controller._render()

    assert controller._accumulator.statistics.received == 2
    _assert_selections(controller, expected)
    controller.end_session_replay()
    _assert_selections(controller, expected)


def test_readback_refresh_retains_fit_guides_and_clamps_roi(qtbot: QtBot) -> None:
    plot = OfflinePsdPlot()
    qtbot.addWidget(plot)
    plot.set_data(np.ones((64, 64)), energy_range=(0, 65536), ratio_range=(-1, 1))
    plot.cut_spin.setValue(0.2345)
    plot.energy_region.setRegion((5000, 20000))
    plot.fom_widget.fit_region.setRegion((-0.25, 0.7))
    plot.fom_widget.centroid_lines[0].setPos(0.05)
    plot.fom_widget.centroid_lines[1].setPos(0.4)

    plot.set_data(np.ones((128, 128)), energy_range=(0, 32768), ratio_range=(-0.5, 0.9))

    assert plot.cut_spin.value() == pytest.approx(0.2345)
    assert plot.energy_region.getRegion() == pytest.approx((5000, 20000))
    assert plot.fom_widget.fit_region.getRegion() == pytest.approx((-0.25, 0.7))
    assert [line.value() for line in plot.fom_widget.centroid_lines] == pytest.approx((0.05, 0.4))

    plot.set_data(np.ones((128, 128)), energy_range=(0, 4096), ratio_range=(-0.5, 0.9))
    assert plot.energy_region.getRegion() == pytest.approx((4096, 4096))



def test_independent_fit_ranges_and_continuous_option_round_trip(controller: PSDController) -> None:
    widget = controller._fom_widget
    widget.mode_combo.setCurrentIndex(widget.mode_combo.findData("separate"))
    widget.peak_regions[0].setRegion((-0.1, 0.15))
    widget.peak_regions[1].setRegion((0.25, 0.65))
    widget.continuous_checkbox.setChecked(True)
    widget.background_checkbox.setChecked(False)
    saved = controller.configuration_settings()
    controller.apply_configuration_settings({"energy_bins": 2048})
    assert controller.configuration_settings()["fom_peak_rois"] == saved["fom_peak_rois"]
    widget.mode_combo.setCurrentIndex(widget.mode_combo.findData("double"))
    widget.background_checkbox.setChecked(True)
    widget.continuous_checkbox.setChecked(False)
    widget.peak_regions[0].setRegion((0.0, 0.1))
    controller.apply_configuration_settings(saved)
    current = controller.configuration_settings()
    assert current["fom_fit_mode"] == "separate"
    assert current["fom_peak_rois"] == saved["fom_peak_rois"]
    assert current["fom_continuous"] is True
    assert current["fom_use_background"] is False
    widget.continuous_checkbox.setChecked(False)
