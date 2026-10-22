"""Confirm a provisional offline-gate preset before writing an idle MCA."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.waveform_mca_gates import GateTimingProposal, WaveformGateSource
from nlab.hardware.digitizer.mca import McaGateTiming
from nlab.views.tooltips import wrap_widget_tooltips

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class McaGateTransferTarget:
    label: str
    read: Callable[[], McaGateTiming]
    apply: Callable[[McaGateTiming], None]


class WaveformMcaTransferDialog(QDialog):
    preview_changed = Signal(object)

    def __init__(self, source: WaveformGateSource, targets: Sequence[McaGateTransferTarget],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Transfer waveform gates to MCA")
        self.source, self.targets = source, tuple(targets)
        self.proposal: GateTimingProposal | None = None
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Candidate starting preset, not a hardware-equivalent PSD calculation. "
            "Pretrigger = integration start (green) − baseline region end; "
            "window = long end − start; "
            "charge-comparison time = short end − start. Values round UP to supported steps. "
            "Only these timings and charge-comparison enable are changed; baseline, polarity, "
            "trigger/filter settings and all other processing remain unchanged. "
            "Pretrigger uses the gap from the baseline region's right boundary to the green line, "
            "not a manually entered hardware trigger position.", self,
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        form = QFormLayout()
        self.target_combo = QComboBox(self)
        self.target_combo.addItems([target.label for target in self.targets])
        form.addRow("Target channel:", self.target_combo)
        form.addRow("Baseline region end:", QLabel(
            f"{source.baseline_end * source.sample_period_ns:g} ns", self,
        ))
        form.addRow("Integration start (green):", QLabel(
            f"{source.gate_start * source.sample_period_ns:g} ns", self,
        ))
        layout.addLayout(form)
        self.enable_cc = QCheckBox("Enable charge comparison on the target", self)
        self.enable_cc.setChecked(True)
        layout.addWidget(self.enable_cc)
        self.table = QTableWidget(4, 4, self)
        self.table.setHorizontalHeaderLabels(["MCA parameter", "Current", "Requested", "Rounded"])
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)
        hint = QLabel(
            "Dashed markers on the waveform show the rounded candidate boundaries. "
            "They do not move your analysis gates or recalculate FOM: recorded samples "
            "may be coarser than hardware timing. MCA window/integration equivalence "
            "is not qualified. Stop acquisition and fully close DMA before applying.", self,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.status = QLabel(self)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply
                                       | QDialogButtonBox.StandardButton.Cancel, self)
        self.apply_button = self.buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.clicked.connect(self._apply)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.target_combo.currentIndexChanged.connect(self.refresh)
        self.enable_cc.toggled.connect(self.refresh)
        wrap_widget_tooltips(self)
        self.resize(740, 520)
        self.refresh()

    def refresh(self) -> None:
        self.proposal = None
        self.table.clearContents()
        self.apply_button.setEnabled(False)
        try:
            proposal = self.source.propose(self.enable_cc.isChecked())
            self.proposal = proposal
            self._show_values(proposal, None)
            self.preview_changed.emit(proposal)
            if self.target_combo.currentIndex() < 0:
                raise ValueError("No live MCA target. Connect hardware and leave replay first.")
            current = self.targets[self.target_combo.currentIndex()].read()
            self._show_values(proposal, current)
            self.apply_button.setEnabled(True)
            self.status.setText("Ready to apply candidate settings.")
        except Exception as exc:
            if self.proposal is None:
                self.preview_changed.emit(None)
            self.status.setText(str(exc))

    def _show_values(self, proposal: GateTimingProposal, current: McaGateTiming | None) -> None:
        timing = proposal.timing
        old = ((current.pretrigger_ns, current.frame_ns, current.cc_time_ns, current.cc_enabled)
               if current is not None else (None,) * 4)
        rows = (
            ("Pretrigger (ns)", old[0], proposal.desired_ns[0], timing.pretrigger_ns),
            ("Window length (ns)", old[1], proposal.desired_ns[1], timing.frame_ns),
            ("Charge comparison (ns)", old[2], proposal.desired_ns[2], timing.cc_time_ns),
            ("Charge comparison enabled", old[3], timing.cc_enabled, timing.cc_enabled),
        )
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                text = "—" if value is None else str(value)
                if isinstance(value, bool):
                    text = "Yes" if value else "No"
                self.table.setItem(row, column, QTableWidgetItem(text))

    def _apply(self) -> None:
        self.refresh()  # Recheck availability after any nested event loop/delay.
        proposal = self.proposal
        if proposal is None or not self.apply_button.isEnabled():
            return
        target = self.targets[self.target_combo.currentIndex()]
        answer = QMessageBox.question(
            self, "Apply candidate MCA settings?",
            f"Write the displayed rounded settings to {target.label}?\n"
            "This does not start acquisition. Offline and hardware FOM may differ.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            target.apply(proposal.timing)
        except Exception as exc:
            log.exception("Waveform-to-MCA gate transfer failed")
            self.refresh()
            self.status.setText(f"Transfer failed: {exc}")
            QMessageBox.warning(self, "MCA gate transfer failed", str(exc))
            return
        self.accept()
