from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtWidgets import QLabel
from pytestqt.qtbot import QtBot

from nlab.analysis.psd_fom import fit_psd_fom_lm
from nlab.help.context import resolve_context
from nlab.views.offline_psd_plot import OfflinePsdPlot
from nlab.views.psd_readback_dialog import PsdReadbackDialog
from nlab.views.waveform_analysis_dialog import WaveformAnalysisDialog


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
    assert "confirmation is required" in dialog.exhaustive_search.toolTip()


def test_waveform_legend_stays_at_top_right_after_resizing(qtbot: QtBot) -> None:
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
                and abs(rect.top() - parent.top() - 10) < 1
            )

        qtbot.waitUntil(anchored)
        assert legend.opts["offset"] == (-10, 10)


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
