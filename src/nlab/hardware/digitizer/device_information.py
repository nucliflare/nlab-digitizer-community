"""Scalar, read-only connection snapshots for the About dialog."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import iio


@dataclass(frozen=True)
class InformationSection:
    title: str
    rows: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class DigitizerInformation:
    uri: str
    channel: int
    board_sections: tuple[InformationSection, ...]
    channel_sections: tuple[InformationSection, ...]


# Identity/transport fields from PetaLinux c546f14 docs/user-api.md. Do not
# confuse the PP sw_version hardware register with a deployed image/build ID.
_CORE_ATTRIBUTES = {
    "Scope": ("ip_version", "trigger_mode_available", "dma_queue_mode",
              "dma_kernel_buffers_max", "dma_kernel_buffers_recommended"),
    "Pulse processor": ("hw_version", "sw_version", "id_number", "channel_index",
                        "parameter_abi_version", "parameter_units",
                        "psd_zero_crossing_mode_available"),
    "List mode": ("ip_version", "channel_index", "frame_records", "frame_bytes",
                  "record_layout", "dma_queue_mode"),
    "Input filter": ("ip_version",),
    "AFE DAC": ("ip_version", "baseline_multiplier", "baseline_max", "sclk_frequency"),
    "Shared START": ("ip_version",),
}
_IMAGE_ATTRIBUTES = (
    "firmware_id", "firmware_version", "firmware", "fw_version", "fw_id", "hw_firmware",
    "build_id", "image_id", "image_version", "bitstream_id", "bitstream_version",
)
_UNKNOWN = "Unknown (not reported by target)"


def collect_device_information(
    context: iio.Context,
    *,
    uri: str,
    channel: int,
    devices: Mapping[str, iio.Device | None],
    dac_output: str | None = None,
    configured_image: str = "",
    producer_identity: Mapping[str, int | str] | None = None,
) -> DigitizerInformation:
    """Read identity only, while the connection owner still owns all handles.

    No Context is created, no scan element or attribute is written, and no
    IIO object is retained in the result. About consumes only this snapshot.
    """
    context_attrs = tuple(sorted((str(key), str(value)) for key, value in context.attrs.items()))
    attributes = dict(context_attrs)
    image = "; ".join(f"{key}: {attributes[key]}" for key in _IMAGE_ATTRIBUTES
                      if attributes.get(key)) or _UNKNOWN
    version = context.version
    backend_version = f"{version[0]}.{version[1]}" + (f" ({version[2]})" if version[2] else "")
    connection_rows = [
        ("Endpoint", uri),
        ("Description", str(context.description) or _UNKNOWN),
        ("IIO backend", str(context.name)),
        ("IIO backend version", backend_version),
        ("Firmware / image reported by target", image),
        ("Captured (UTC)", datetime.now(UTC).isoformat(timespec="seconds")),
    ]
    inventory = tuple((str(device.id), str(device.name or device.id)) for device in context.devices)
    board_sections = (
        InformationSection("Connection and firmware", tuple(connection_rows)),
        InformationSection("Reported board / context attributes", context_attrs),
        InformationSection("Detected IIO devices", inventory),
    )
    channel_sections = []
    if producer_identity is not None:
        channel_sections.append(InformationSection("List-mode producer / decoder", (
            ("Decoder", str(producer_identity.get(
                "qualified_record_schema", "Unknown (raw only)"))),
            ("Selection / qualification", str(producer_identity.get(
                "decoder_selection", _UNKNOWN))),
            ("Decoder image identity", str(producer_identity.get("firmware_id", _UNKNOWN))),
        )))
    if configured_image:
        channel_sections.append(InformationSection("Configured provenance", (
            ("Image identity (user supplied)", configured_image),
        )))
    for role, names in _CORE_ATTRIBUTES.items():
        device = devices.get(role)
        if device is None:
            channel_sections.append(InformationSection(role, (("Availability", "Not present"),)))
            continue
        rows = [("IIO device", f"{device.name or device.id} ({device.id})")]
        if role == "AFE DAC":
            rows.append(("Selected output", dac_output or "Unavailable"))
        for name in names:
            if name not in device.attrs:
                continue
            try:
                value = str(device.attrs[name].value)
            except Exception as exc:
                value = f"Unavailable: {exc}"
            rows.append((name, value))
        channel_sections.append(InformationSection(role, tuple(rows)))
    return DigitizerInformation(uri, channel, board_sections, tuple(channel_sections))
