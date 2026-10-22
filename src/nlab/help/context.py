"""Resolve UI labels and their controls without importing hardware controllers.

Designer object names are scoped to their form: a Scope trigger and an MCA
trigger must never acquire each other's explanation. Programmatic form labels
are resolved through their field widgets, not through translated label text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QBoxLayout,
    QComboBox,
    QDial,
    QFormLayout,
    QLabel,
    QLayoutItem,
    QLineEdit,
    QSlider,
    QWidget,
)


@dataclass(frozen=True)
class HelpContext:
    topic: str
    control: str = ""
    tooltip: str = ""
    specific: bool = True


# Keys are Designer control names without their type prefix. The same entry
# covers a label, slider, spin box, and the spin box's internal line edit.
FORM_TOPICS: dict[str, tuple[str, dict[str, str]]] = {
    "ScopeView": ("scope", {
        "TriggerLevel": "scope-trigger", "TriggerMode": "scope-trigger",
        "DacValue": "scope-dac", "Dac": "scope-dac",
        "Pretrigger": "scope-pretrigger", "FrameSamples": "scope-frame",
        "FrameGap": "scope-gap", "DmaEnable": "scope-dma",
        "DmaFile": "scope-dma", "Time": "duration",
        "Start": "scope-run", "Stop": "scope-run", "AcquireFrame": "scope-run",
        "AutoSetup": "scope-auto", "DisplayMode": "display",
        "Persistence": "display", "PersistenceValue": "display", "RefreshRate": "refresh",
        "Waveform": "scope", "plotWaveform": "scope",
    }),
    "MCAView": ("mca", {
        "Polarity": "mca-polarity", "PulsePolarity": "mca-polarity",
        "Baseline": "mca-baseline", "Debug1": "mca-debug", "Debug2": "mca-debug",
        "Pileups": "mca-pileup", "PileupWindow": "mca-pileup",
        "Binning": "mca-binning", "ExtTrigger": "mca-trigger",
        "DmaEnable": "mca-dma", "DmaFile": "mca-dma",
        "TimeLimit": "duration", "Start": "mca-run", "Stop": "mca-run",
        "RefreshRate": "refresh", "TriggerLevel": "mca-trigger",
        "TriggerSource": "mca-trigger", "EdgeDetCoeff": "mca-trigger",
        "FrameSamples": "mca-window", "Pretrigger": "mca-window",
        "LpPreset": "mca-lowpass", "Crrc2Cdelay": "mca-crrc",
        "Crrc2Fdelay": "mca-crrc", "Crrc2Pzc": "mca-crrc",
        "CfdEnable": "mca-cfd", "CfdFactor": "mca-cfd", "CfdDelay": "mca-cfd",
        "CfdTwLow": "mca-cfd", "CfdTwHigh": "mca-cfd",
        "TrapezEnable": "mca-trapezoid", "TrapR": "mca-trapezoid",
        "TrapM": "mca-trapezoid", "TrapT": "mca-trapezoid",
        "TrapE": "mca-trapezoid", "TrapFt": "mca-trapezoid",
        "CcEnable": "mca-charge", "CcTime": "mca-charge",
        "PsdZcEnable": "mca-zerocross", "PsdZcMode": "mca-zerocross",
        "PsdZcLow": "mca-zerocross", "PsdZcHigh": "mca-zerocross",
        "ClearSpectrum": "mca-run", "ExportCsv": "export",
        "plotDebug": "mca-debug", "plotHistogram": "mca-roi",
        "roiStatsPanel": "mca-roi", "Statistics": "mca-statistics",
    }),
    "PSDView": ("psd", {
        "EnergyBins": "psd-bins", "RatioBins": "psd-bins",
        "EnergyShift": "psd-shift", "RatioMin": "psd-range", "RatioMax": "psd-range",
        "Cut": "psd-cut", "Clear": "psd", "plotPsd": "psd-roi",
        "plotRatio": "fom", "plotEnergy": "psd-cut",
    }),
    "CurrentMonitorView": ("current", {
        "Mode": "current-mode", "DmaGap": "current-mode",
        "Start": "current", "Stop": "current", "ResetScopeDefaults": "current-mode",
        "Zero": "current-calibration", "Scale": "current-calibration",
        "Unit": "current-calibration", "Calibration": "current-calibration",
    }),
    "GlobalView": ("global", {
        "SyncSource": "global-sync", "SyncEnable": "global-sync",
        "PrepareSoftware": "global-sync", "SoftwareStart": "global-sync",
        "SoftwareReset": "global-sync", "SoftwareStartHelp": "global-sync",
        "TempCoeff": "global-temperature", "TempOffset": "global-temperature",
        "TemperatureCorrectionEnabled": "global-temperature",
        "TemperatureInterval": "refresh", "DiagnosticsInterval": "refresh",
    }),
    "PSUView": ("psu", {
        "SipmEnable": "psu-bias", "SipmVoltage": "psu-bias", "HvVoltage": "psu-bias",
        "SipmCompensMode": "psu-compensation", "SipmCompensCt": "psu-compensation",
        "SipmCompensTref": "psu-compensation", "HvCompensMode": "psu-compensation",
        "HvCompensCt": "psu-compensation", "HvCompensTref": "psu-compensation",
        "TempDigitalEnable": "psu", "RefreshRate": "refresh", "TimeRange": "display",
    }),
    "ExternalDeviceView": ("external", {}),
    "ConnectionDialog": ("connection", {
        "Ip": "connection", "Port": "connection", "Channels": "connection",
        "Rescan": "connection", "Offline": "session",
    }),
}

# Attributes, rather than object names, identify programmatically created controls.
# Include future optional controls without requiring a feature-branch merge.
CLASS_TOPICS: dict[str, tuple[str, dict[str, str]]] = {
    "WaveformAnalysisDialog": ("waveform", {
        "source_combo": "waveform-source", "frame_spin": "waveform-source",
        "sample_period": "waveform-source", "baseline_method": "waveform-baseline",
        "polarity_combo": "waveform-baseline", "waveform_widget": "waveform-gates",
        "energy_bins": "psd-bins", "ratio_bins": "psd-bins",
        "energy_max": "psd-range", "ratio_min": "psd-range", "ratio_max": "psd-range",
        "event_stride": "waveform-sampling", "maximum_events": "waveform-sampling",
        "optimization_events": "waveform-optimization",
        "optimization_radius": "waveform-optimization",
        "exhaustive_search": "waveform-optimization", "optimize_button": "waveform-optimization",
        "auto_recalculate": "waveform-preview", "report_button": "export",
        "analyze_button": "waveform", "cancel_button": "waveform",
    }),
    "PsdFomWidget": ("fom", {
        "guides_checkbox": "fom-guides", "fit_button": "fom",
        "copy_button": "export", "deviance_label": "fom-quality",
        "status_label": "fom-quality", "first_label": "fom",
        "second_label": "fom", "fom_label": "fom",
    }),
    "OfflinePsdPlot": ("psd-roi", {
        "scale_combo": "display", "cut_spin": "psd-cut",
        "_matrix_widget": "psd-roi", "_ratio_widget": "fom-guides",
        "_energy_widget": "psd-cut",
    }),
    "GeneralSettingsDialog": ("settings", {
        "scope_display_mode": "display", "scope_persistence": "display",
        "scope_refresh": "refresh", "mca_refresh": "refresh", "psu_refresh": "refresh",
        "psu_history": "display", "dma_folder": "export", "mca_output": "mca-dma",
        "auto_configuration": "settings", "snapshot_path": "settings",
    }),
    "CoincidenceController": ("coincidence", {
        "duration": "duration", "operator": "coincidence-logic",
        "not_ch0": "coincidence-logic", "not_ch1": "coincidence-logic",
        "timing_mode": "coincidence-timing", "low": "coincidence-timing",
        "high": "coincidence-timing", "offset": "coincidence-timing",
        "random_sidebands": "coincidence-matrix", "random_gap": "coincidence-matrix",
        "matrix_mode": "coincidence-matrix", "matrix_scale": "display",
        "matrix_plot": "coincidence-matrix", "btnExportMatrix": "export",
    }),
    "EnergyCalibrationDialog": ("energy-calibration", {}),
    "DoseCalibrationDialog": ("dose-calibration", {}),
    "McaPeakAnalysisDialog": ("peak-analysis", {}),
    "McaRoiHistoryDialog": ("mca-roi", {}),
    "PsdReadbackDialog": ("psd-readback", {
        "energy_bins": "psd-bins", "ratio_bins": "psd-bins",
        "energy_shift": "psd-shift", "ratio_min": "psd-range", "ratio_max": "psd-range",
    }),
    "RootConversionDialog": ("conversion", {}),
    "TimingValidationDialog": ("timing-validation", {}),
    "DmaSettingsDialog": ("mca-dma", {}),
    "AboutDialog": ("about", {}),
    "LicenseDialog": ("about", {}),
}

_PREFIX = re.compile(r"^(label|slider|combo|spin|dial|button|group|btn|lbl|cb)")


def _ancestors(widget: QWidget) -> list[QWidget]:
    result = []
    current: QWidget | None = widget
    while current is not None:
        result.append(current)
        current = current.parentWidget()
    return result


def _item_widgets(item: QLayoutItem | None) -> list[QWidget]:
    if item is None:
        return []
    widget = item.widget()
    if widget is not None:
        return [widget]
    layout = item.layout()
    if layout is not None:
        return [child for i in range(layout.count()) for child in _item_widgets(layout.itemAt(i))]
    return []


def _label_fields(label: QLabel, root: QWidget) -> list[QWidget]:
    buddy = label.buddy()
    if buddy is not None:
        return [buddy]
    for layout in root.findChildren(QFormLayout):
        index = layout.indexOf(label)
        if index >= 0:
            row, role = layout.getItemPosition(index)
            if role == QFormLayout.ItemRole.LabelRole:
                return _item_widgets(layout.itemAt(row, QFormLayout.ItemRole.FieldRole))
    # Toolbar captions and Designer vertical label/field pairs have no buddy.
    # Resolve their structural neighbour, never a guessed meaning of the text.
    for box in root.findChildren(QBoxLayout):
        index = box.indexOf(label)
        if index >= 0 and index + 1 < box.count():
            following = _item_widgets(box.itemAt(index + 1))
            return [item for item in following if isinstance(
                item, (QAbstractSpinBox, QComboBox, QLineEdit, QSlider, QDial),
            )]
    return []


def _description(widget: QWidget) -> tuple[str, str]:
    if isinstance(widget, QLabel):
        return widget.text(), widget.toolTip()
    label = widget.accessibleName()
    if not label and isinstance(widget, QAbstractButton):
        label = widget.text()
    return label or widget.objectName(), widget.toolTip()


def resolve_context(widget: QWidget) -> HelpContext:
    """Prefer explicit metadata, then exact control bindings, then workspace help.

    New custom controls can set the ``nlabHelpTopic`` dynamic property. Unknown
    controls deliberately fall back instead of guessing semantics from their text.
    """
    chain = _ancestors(widget)
    for ancestor in chain:
        explicit = ancestor.property("nlabHelpTopic")
        if isinstance(explicit, str) and explicit:
            return HelpContext(explicit, *_description(widget))
        form = FORM_TOPICS.get(ancestor.objectName())
        spec = CLASS_TOPICS.get(type(ancestor).__name__)
        if form is None and spec is None:
            continue
        selected_spec = form if form is not None else spec
        assert selected_spec is not None
        default, bindings = selected_spec
        candidates = chain[:chain.index(ancestor) + 1]
        if isinstance(widget, QLabel):
            candidates = _label_fields(widget, ancestor) + candidates
        for candidate in candidates:
            if form is not None:
                key = _PREFIX.sub("", candidate.objectName(), count=1)
                topic = bindings.get(key)
            else:
                topic = next((topic for name, topic in bindings.items()
                              if getattr(ancestor, name, None) is candidate), None)
            if topic is not None:
                control, tooltip = _description(widget)
                if not tooltip:
                    tooltip = candidate.toolTip()
                return HelpContext(topic, control, tooltip)
        return HelpContext(default, *_description(widget), specific=False)
    return HelpContext("help", *_description(widget), specific=False)
