from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QGroupBox, QLabel, QPushButton, QToolBar
from pytestqt.qtbot import QtBot

from nlab.analysis.psd_fom import fit_psd_fom_lm
from nlab.help.context import resolve_context
from nlab.views.offline_psd_plot import OfflinePsdPlot
from nlab.views.psd_readback_dialog import PsdReadbackDialog
from nlab.views.waveform_analysis_dialog import WaveformAnalysisDialog


@pytest.mark.parametrize(("view_type", "title"), [
    (PsdReadbackDialog, "PSD Event Readback"),
    (WaveformAnalysisDialog, "Waveform Analysis Workbench"),
])
def test_workbench_file_identity_lives_in_window_title(
    qtbot: QtBot, view_type: type[PsdReadbackDialog] | type[WaveformAnalysisDialog], title: str,
) -> None:
    dialog = view_type()
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == f"{title} — No file loaded"
    assert dialog.windowFilePath() == ""
    path = Path("D:/") / ("long_directory_" * 20) / "capture.h5"
    dialog._set_current_file(path)
    assert dialog.windowTitle() == f"{title} — capture.h5"
    assert dialog.windowFilePath() == str(path)
    assert not any(label.text().startswith("Current file:")
                   for label in dialog.findChildren(QLabel))
    dialog._set_current_file(None)
    assert dialog.windowTitle() == f"{title} — No file loaded"
    assert dialog.windowFilePath() == ""


def test_waveform_instructions_are_available_without_permanent_banners(qtbot: QtBot) -> None:
    dialog = WaveformAnalysisDialog()
    qtbot.addWidget(dialog)
    labels = [label.text() for label in dialog.findChildren(QLabel)]
    assert not any("Waveforms are read" in text or "Blue region:" in text for text in labels)
    assert not any("Exhaustive search can take" in text for text in labels)
    assert "without changing live" in dialog.toolTip()
    assert "Blue region: baseline" in dialog.waveform_widget.toolTip()
    assert resolve_context(dialog.waveform_widget).topic == "waveform-gates"
    assert dialog.auto_recalculate.text() == "Auto preview"
    assert "20th" in dialog.auto_recalculate.toolTip()
    assert "Releasing a marker" in dialog.auto_recalculate.toolTip()
    assert "confirmation is required" in " ".join(dialog.exhaustive_search.toolTip().split())


def test_waveform_source_controls_live_in_scrollable_left_sidebar(qtbot: QtBot) -> None:
    dialog = WaveformAnalysisDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    assert not any(button.text() in ("Load waveform file...", "Close file")
                   for button in dialog.findChildren(QPushButton))
    for width, height in ((1100, 720), (1450, 1000)):
        dialog.resize(width, height)
        qtbot.wait(10)
        plot_origin = dialog.waveform_widget.mapTo(dialog, QPoint())
        assert dialog.settings_scroll.horizontalScrollBar().maximum() == 0
        for field in (dialog.source_combo, dialog.frame_spin, dialog.sample_period):
            assert dialog.source_settings.isAncestorOf(field)
            assert dialog.settings_panel.isAncestorOf(field)
            assert field.isVisibleTo(dialog)
            assert field.mapTo(dialog, QPoint(field.width(), 0)).x() < plot_origin.x()
            label = next(label for label in dialog.findChildren(QLabel) if label.buddy() is field)
            assert resolve_context(field).topic == "waveform-source"
            assert resolve_context(label).topic == "waveform-source"
        assert dialog.source_settings.mapTo(dialog, QPoint()).y() >= plot_origin.y()
        dialog.settings_scroll.ensureWidgetVisible(dialog.cancel_button)
        qtbot.wait(10)
        viewport = dialog.settings_scroll.viewport()
        center = dialog.cancel_button.mapTo(viewport, dialog.cancel_button.rect().center())
        assert viewport.rect().contains(center)
        dialog.settings_scroll.verticalScrollBar().setValue(0)


def test_waveform_legend_stays_at_bottom_right_after_resizing(qtbot: QtBot) -> None:
    dialog = WaveformAnalysisDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    # Synthetic trace data only; no hardware or capture file is opened.
    dialog.raw_curve.setData(np.arange(64), np.zeros(64))
    dialog.oriented_curve.setData(np.arange(64), np.ones(64))
    legend = dialog.waveform_plot.legend
    assert legend is not None
    assert len(legend.items) == 2
    for width, height in ((1100, 850), (1450, 1000)):
        dialog.resize(width, height)

        def anchored() -> bool:
            rect = legend.mapRectToParent(legend.boundingRect())
            parent = legend.parentItem().boundingRect()
            return bool(
                abs(parent.right() - rect.right() - 10) < 1
                and abs(parent.bottom() - rect.bottom() - 10) < 1
            )

        qtbot.waitUntil(anchored)
        assert legend.opts["offset"] == (-10, -10)


@pytest.mark.parametrize(("span_ns", "unit", "ns_per_unit"), [
    (500.0, "ns", 1.0),
    (16_000.0, "\N{MICRO SIGN}s", 1_000.0),
    (2_000_000.0, "ms", 1_000_000.0),
    (2_000_000_000.0, "s", 1_000_000_000.0),
])
def test_waveform_axes_use_readable_time_units_without_changing_samples_or_gates(
    qtbot: QtBot, span_ns: float, unit: str, ns_per_unit: float,
) -> None:
    dialog = WaveformAnalysisDialog()
    qtbot.addWidget(dialog)
    dialog.sample_period.setValue(8.0)
    dialog._updating_gates = True
    dialog.baseline_region.setRegion((0, 512))
    dialog.gate_start_line.setPos(528)
    dialog.short_end_line.setPos(1328)
    dialog.long_end_line.setPos(4608)
    dialog._updating_gates = False
    times_ns = np.arange(2048, dtype=np.float64) * 8.0
    samples = np.zeros(2048)
    dialog.raw_curve.setData(times_ns, samples)
    before = dialog._gate_indices()
    for span, expected_unit, factor in ((span_ns, unit, ns_per_unit),
                                        (100.0, "ns", 1.0),
                                        (span_ns, unit, ns_per_unit)):
        dialog.waveform_plot.setXRange(0, span, padding=0)
        bottom = dialog.waveform_plot.getAxis("bottom")
        assert bottom.labelUnits == expected_unit
        assert bottom.label.toPlainText() == f"Time ({expected_unit})"
        for side in ("top", "bottom"):
            axis = dialog.waveform_plot.getAxis(side)
            assert not axis.autoSIPrefix
            assert axis.scale == pytest.approx(1 / factor)
            assert axis.tickStrings([0, 2 * factor], axis.scale, factor) == ["0", "2"]
        x, y = dialog.raw_curve.getData()
        np.testing.assert_array_equal(x, times_ns)
        np.testing.assert_array_equal(y, samples)
        assert dialog._gate_indices() == before == (0, 64, 66, 166, 576)


def test_readback_grid_labels_open_their_parameter_topics(qtbot: QtBot) -> None:
    dialog = PsdReadbackDialog()
    qtbot.addWidget(dialog)
    labels = dialog.findChildren(QLabel)
    assert not any("Read stored long/short" in label.text() for label in labels)
    assert "(Qlong - Qshort) / Qlong" in dialog.toolTip()
    for field, topic in (
        (dialog.energy_bins, "psd-bins"), (dialog.ratio_bins, "psd-bins"),
        (dialog.energy_shift, "psd-shift"), (dialog.ratio_min, "psd-range"),
        (dialog.ratio_max, "psd-range"),
    ):
        label = next(label for label in labels if label.buddy() is field)
        assert resolve_context(label).topic == topic
        assert resolve_context(field).topic == topic


def test_readback_settings_and_display_controls_stay_left_of_full_height_plots(
    qtbot: QtBot,
) -> None:
    dialog = PsdReadbackDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.plot_splitter.orientation() == Qt.Orientation.Horizontal
    assert not dialog.plot_splitter.isCollapsible(0)
    assert not dialog.plot_splitter.isCollapsible(1)
    for width in (1000, 1280):
        dialog.resize(width, 760)
        qtbot.wait(10)
        plot_origin = dialog.plot.mapTo(dialog, QPoint())
        sidebar_origin = dialog.settings_panel.mapTo(dialog, QPoint())
        assert sidebar_origin.y() == plot_origin.y()
        assert dialog.settings_panel.width() <= 300
        assert dialog.histogram_settings.isVisibleTo(dialog)
        for control in (dialog.energy_bins, dialog.ratio_bins, dialog.energy_shift,
                        dialog.ratio_min, dialog.ratio_max, dialog.plot.scale_combo,
                        dialog.plot.cut_spin, dialog.plot.roi_checkbox):
            assert control.isVisibleTo(dialog)
            assert dialog.settings_panel.isAncestorOf(control)
            assert control.mapTo(dialog, QPoint(control.width(), 0)).x() < plot_origin.x()
    # Reparenting display controls must keep their original help topics and
    # plotting connections, not turn them into disconnected sidebar copies.
    assert resolve_context(dialog.plot.scale_combo).topic == "display"
    assert resolve_context(dialog.plot.cut_spin).topic == "psd-cut"
    for control, topic in ((dialog.plot.scale_combo, "display"),
                           (dialog.plot.cut_spin, "psd-cut")):
        label = next(label for label in dialog.findChildren(QLabel) if label.buddy() is control)
        assert resolve_context(label).topic == topic
    dialog.plot.cut_spin.setValue(.25)
    assert dialog.plot.cut_line.value() == pytest.approx(.25)
    dialog.roi_action.trigger()
    assert not dialog.plot.energy_region.isVisible()
    assert not dialog.plot.roi_checkbox.isChecked()


def test_readback_menu_groups_actions_separately_from_histogram_settings(qtbot: QtBot) -> None:
    dialog = PsdReadbackDialog()
    qtbot.addWidget(dialog)
    assert not dialog.findChildren(QToolBar)
    assert [action.text().replace("&", "") for action in dialog.menu_bar.actions()] == [
        "File", "Analysis", "View", "Help",
    ]
    for action in (dialog.load_action, dialog.cancel_action, dialog.report_action,
                   dialog.close_action):
        assert action in dialog.file_menu.actions()
    settings = next(group for group in dialog.findChildren(QGroupBox)
                    if group.title() == "Histogram settings")
    assert not settings.findChildren(QPushButton)
    assert not any(label.text().startswith("Current file:")
                   for label in dialog.findChildren(QLabel))
    assert dialog.load_action.isEnabled()
    assert not dialog.cancel_action.isEnabled()
    assert not dialog.report_action.isEnabled()
    dialog._set_busy(True)
    assert not dialog.load_action.isEnabled()
    assert dialog.cancel_action.isEnabled()
    assert not dialog.report_action.isEnabled()
    dialog._set_busy(False)
    assert dialog.load_action.isEnabled() and not dialog.cancel_action.isEnabled()


def test_readback_menu_uses_existing_file_cancel_report_and_close_handlers(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    choose, cancel, report = Mock(), Mock(), Mock()
    monkeypatch.setattr(PsdReadbackDialog, "_choose_file", choose)
    monkeypatch.setattr(PsdReadbackDialog, "_cancel", cancel)
    monkeypatch.setattr(PsdReadbackDialog, "_download_report", report)
    dialog = PsdReadbackDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.activateWindow()
    dialog.raise_()
    qtbot.waitUntil(dialog.isActiveWindow)
    dialog.load_action.trigger()
    assert choose.call_count == 1
    qtbot.keyClick(dialog, Qt.Key.Key_O, Qt.KeyboardModifier.ControlModifier)
    assert choose.call_count == 2
    assert dialog.load_action.shortcutContext() == Qt.ShortcutContext.WindowShortcut
    dialog._set_busy(True)
    qtbot.keyClick(dialog, Qt.Key.Key_O, Qt.KeyboardModifier.ControlModifier)
    assert choose.call_count == 2
    dialog.cancel_action.trigger()
    assert cancel.call_count == 1
    dialog._set_busy(False)
    dialog.report_action.setEnabled(True)
    dialog.report_action.trigger()
    assert report.call_count == 1
    dialog.close_action.trigger()
    assert not dialog.isVisible()


def test_readback_view_and_analysis_menu_states_stay_synchronized(qtbot: QtBot) -> None:
    dialog = PsdReadbackDialog()
    qtbot.addWidget(dialog)
    for action, control in ((dialog.roi_action, dialog.plot.roi_checkbox),
                            (dialog.guides_action, dialog.plot.fom_widget.guides_checkbox)):
        assert action.isChecked() and control.isChecked()
        action.trigger()
        assert not action.isChecked() and not control.isChecked()
        control.setChecked(True)
        assert action.isChecked()
    assert not dialog.copy_action.isEnabled()
    dialog.plot.fom_widget.copy_button.setEnabled(True)
    assert dialog.copy_action.isEnabled()
    dialog.plot.fom_widget.copy_button.setEnabled(False)
    assert not dialog.copy_action.isEnabled()
    dialog.plot.fom_widget.fit_button.setEnabled(False)
    assert not dialog.fit_action.isEnabled()


def test_readback_help_menu_opens_bundled_guide(qtbot: QtBot) -> None:
    dialog = PsdReadbackDialog()
    qtbot.addWidget(dialog)
    dialog.guide_action.trigger()
    assert dialog._help_dialog is not None and dialog._help_dialog.isVisible()
    assert dialog._help_dialog._history[-1].topic == "psd-readback"
    guide = dialog._help_dialog
    guide.close()
    dialog.guide_action.trigger()
    assert dialog._help_dialog is guide and guide.isVisible()


def test_compact_fom_panel_keeps_results_and_fit_warnings(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plot = OfflinePsdPlot()
    qtbot.addWidget(plot)
    widget = plot.fom_widget
    assert widget.guides_checkbox.text() == "Fit guides"
    assert widget.status_label.text() == "No fit yet."
    assert "snapshot" in widget.status_label.toolTip()
    assert "centroid" in widget.first_label.toolTip()
    assert widget.minimumSizeHint().width() <= widget.maximumWidth()
    ratio = np.linspace(0, 1, 256)
    counts = 2 + 200 * np.exp(-0.5 * ((ratio - 0.3) / 0.04) ** 2)
    counts += 150 * np.exp(-0.5 * ((ratio - 0.7) / 0.05) ** 2)
    fit = fit_psd_fom_lm(ratio, counts, roi=(0.1, 0.9), centroid_seeds=(0.3, 0.7))
    warning_fit = replace(fit, warnings=("Synthetic fit-quality warning",))
    monkeypatch.setattr(
        "nlab.views.psd_fom_widget.fit_psd_fom_lm", Mock(return_value=warning_fit),
    )
    widget.set_projection(ratio, counts)
    assert widget.fit_current_projection() is warning_fit
    assert "Warning: poor convergence" in widget.status_label.text()
    assert "Synthetic fit-quality warning" in widget.status_label.text()
    assert widget.first_label.text() != "—" and widget.fom_label.text() != "—"
