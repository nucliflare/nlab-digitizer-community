"""Every persistent QSettings key used by the application, in one place.

Values are part of the user's saved state: never change a string here without
a migration. Module-level constants only, no Qt import, so any layer can use
them. Per-channel keys are built by the functions at the end.
"""

from __future__ import annotations

# Connection dialog
CONNECTION_RECENT_IPS = "connection/recent_ips"
CONNECTION_IIO_PORT = "connection/iio_port"
CONNECTION_LAST_CHANNELS = "connection/last_channels"
LEGACY_CONNECTION_LAST_BACKEND = "connection/last_backend"
LEGACY_CONNECTION_LAST_PORT = "connection/last_port"

# Recording destinations and formats
DMA_SAVE_FOLDER = "dma/save_folder"
DMA_SAVE_FOLDER_DEFAULT = "measurements"
MCA_DMA_OUTPUT_MODE = "dma/mca_output_mode"
RECORDING_TIMESTAMP_CHOSEN_NAMES = "dma/timestamp_chosen_names"
SESSION_LAST_FOLDER = "session/last_folder"
CONFIGURATION_AUTO_SAVE_ENABLED = "configuration/auto_save_enabled"

# Developer and view options
DEVELOPER_SHOW_SYSTEM_LOG = "developer/show_system_log"
DEVELOPER_DEBUG_MODE = "developer/debug_mode"
VIEW_SHOW_ROI = "view/show_roi"
VIEW_DIFFERENTIAL_HISTOGRAM = "view/differential_histogram"
VIEW_LOG_Y = "view/log_y"
VIEW_SCOPE_DISTRIBUTION = "view/scope_distribution"
VIEW_SCOPE_SPECTRUM = "view/scope_spectrum"
MEASUREMENT_SERIES_ENABLED = "measurement/series_enabled"
TIMING_CHANNEL_B_OFFSET_NS = "timing/channel_b_offset_ns"

# Appearance (the theme applies at the next start; waveform drawing immediately)
APPEARANCE_THEME = "appearance/theme"
APPEARANCE_WAVEFORM_DRAWING = "appearance/waveform_drawing"

# Global workspace
GLOBAL_DIAGNOSTICS_INTERVAL_MS = "global/diagnostics_interval_ms"
GLOBAL_TEMPERATURE_INTERVAL_MS = "global/temperature_interval_ms"
GLOBAL_TEMPERATURE_COEFFICIENT = "global/temperature_coefficient"
GLOBAL_TEMPERATURE_OFFSET = "global/temperature_offset"
GLOBAL_TEMPERATURE_ENABLED = "global/temperature_enabled"

# Dock layouts (versioned: bump the version when the layout model changes)
DOCK_STATE_SCOPE = "docks/v2/scope"
DOCK_STATE_CURRENT = "docks/v1/current"
DOCK_STATE_MCA = "docks/v2/mca"
DOCK_STATE_PSD = "docks/v2/psd"
DOCK_STATE_PSU = "docks/v2/psu"
DOCK_STATE_GLOBAL = "docks/v2/global"
DOCK_STATE_EXTERNAL = "docks/v2/external"
DOCK_TOPOLOGY_PREFIX = "docks/topology/v1"

# Analysis dialogs: last-used folders and preferences
ANALYSIS_SPECTRUM_FOLDER = "analysis/spectrum_folder"
ANALYSIS_PEAK_FIT_REPORT_FOLDER = "analysis/peak_fit_report_folder"
ANALYSIS_ENERGY_CALIBRATION_REPORT_FOLDER = "analysis/energy_calibration_report_folder"
ANALYSIS_DOSE_CALIBRATION_FOLDER = "analysis/dose_calibration_folder"
ANALYSIS_PSD_EVENT_FOLDER = "analysis/psd_event_folder"
ANALYSIS_PSD_REPORT_FOLDER = "analysis/psd_report_folder"
ANALYSIS_WAVEFORM_FOLDER = "analysis/waveform_folder"
ANALYSIS_WAVEFORM_REPORT_FOLDER = "analysis/waveform_report_folder"
ANALYSIS_RADIONUCLIDE_FAVOURITES = "analysis/radionuclide_reference_favourites"


def scope_channel(channel: int, name: str) -> str:
    """Per-channel Scope display preference, e.g. ``scope/ch0/refresh_rate``."""
    return f"scope/ch{channel}/{name}"
