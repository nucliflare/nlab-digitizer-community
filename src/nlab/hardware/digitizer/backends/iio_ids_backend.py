"""IIO backend for the high-voltage control and temperature hardware.

The device/attribute mapping comes from the small reference clients in
``hw_description/iio_mcp3564.c``, ``iio_ad5686r.c`` and ``iio_tmp117.c``:

* AD5686R output ``voltageN.raw`` programs the HV set point;
* MCP3564(R) channels are selected by their ``label`` attribute and expose
  raw HV feedback and the ADC's internal temperature sensor;
* TMP117 instances are selected by their device label and expose
  ``temp.raw`` plus ``temp.scale``. Channel-specific ``cha_temp`` and
  ``chb_temp`` sensors are preferred but optional; newer boards with only
  the shared ``HAT_temp`` sensor use that as the digital-temperature source.

The legacy IDS service also exposed a SiPM supply. That hardware is no
longer present, so the corresponding abstract-interface methods deliberately
do not attempt to alias another DAC/ADC channel.

Device discovery and all read paths were confirmed live on both channels on
2026-08-10. The optional channel-temperature topology was confirmed on
2026-09-14 against 192.168.10.128 (three TMP117s) and 192.168.10.135 plus
10.7.0.121 (only HAT_temp). DAC writes and software compensation were
boundary/unit tested only; they were not exercised against live HV hardware
during implementation.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import iio

from ..diagnostics import GlobalDiagnosticReading
from .base import IDSBackend

log = logging.getLogger(__name__)

_ADC_DEVICE_NAMES = ("mcp3564", "mcp3564r")
_DAC_DEVICE_NAME = "ad5686r"
_TMP_DEVICE_NAME = "tmp117"
_ADS_DEVICE_NAME = "ads5407"
_XADC_DEVICE_NAME = "xadc"
_CLOCK_DEVICE_NAME = "ltc6951"

_HV_FEEDBACK_LABELS = ("CHA_HV_get_Vout", "CHB_HV_get_Vout")
_TMP_LABELS = ("cha_temp", "chb_temp")
_GLOBAL_TMP_LABEL = "HAT_temp"

# IIO voltage scales are expressed in mV. The board's 2.5 V DAC/ADC span
# represents the legacy IDS interface's 0..1250 V HV span, hence 0.5 V of
# HV per millivolt at the converter pin. Keep this constructor-configurable
# because the three generic converter drivers cannot describe an external
# board-level divider/gain in their channel attributes.
_DEFAULT_HV_GAIN_V_PER_MV = 0.5
_HV_MIN_VOLTAGE = 0.0
_HV_MAX_VOLTAGE = 1250.0
_DAC_MIN_RAW = 0
_DAC_MAX_RAW = 65535

# MCP3561/2/4R data sheet, Equation 5-1 (first-order fit):
# VIN(mV) = 0.2973 * TEMP(degC) + 80.
_MCP_TEMP_OFFSET_MV = 80.0
_MCP_TEMP_SLOPE_MV_PER_C = 0.2973

_SIPM_UNSUPPORTED = (
    "IIO IDS backend: SiPM voltage-control hardware is not present; no "
    "AD5686R/MCP3564/TMP117 attribute is a valid replacement"
)
_COMPENSATION_MODE_UNSUPPORTED = (
    "IIO IDS backend: HV temperature-compensation mode 3 has no documented "
    "sensor mapping in the legacy interface or the three IIO clients"
)


class IIOIDSUnavailableError(RuntimeError):
    """Required IIO devices for one channel's IDS/PSU panel are absent."""


def _find_device(context: iio.Context, *names: str) -> iio.Device | None:
    """Find the first device whose IIO name matches one of *names*."""
    for name in names:
        device = context.find_device(name)
        if device is not None:
            return device
    return None


def _find_labelled_channel(device: iio.Device, label: str) -> iio.Channel | None:
    """Match a channel by the ``label`` attribute used by iio_mcp3564.c."""
    for channel in device.channels:
        attr = channel.attrs.get("label")
        if attr is not None and attr.value.strip() == label:
            return channel
    return None


def _find_labelled_device(
    context: iio.Context,
    name: str,
    label: str,
) -> iio.Device | None:
    """Match same-name devices by their IIO device label, not probe order."""
    matches = [
        device
        for device in context.devices
        if device.name == name and getattr(device, "label", None) == label
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"expected at most one '{name}' device labelled '{label}', "
            f"found {len(matches)}"
        )
    return matches[0] if matches else None


@dataclass(frozen=True)
class _IDSDevices:
    adc: iio.Device
    hv_feedback: iio.Channel
    adc_temperature: iio.Channel
    dac: iio.Device
    hv_output: iio.Channel
    digital_temp_input: iio.Channel | None
    digital_temp_label: str | None
    ads_temp_input: iio.Channel | None


@dataclass(frozen=True)
class _GlobalDevices:
    hat_temp_input: iio.Channel | None
    ads: iio.Device | None
    ads_temp_input: iio.Channel | None
    xadc_temp_input: iio.Channel | None
    xadc_rails: tuple[tuple[str, iio.Channel], ...]
    clock: iio.Device | None


def _discover_global_devices(context: iio.Context) -> _GlobalDevices:
    hat = _find_labelled_device(context, _TMP_DEVICE_NAME, _GLOBAL_TMP_LABEL)
    hat_temp_input = hat.find_channel("temp") if hat is not None else None

    ads = _find_device(context, _ADS_DEVICE_NAME)
    ads_temp_input = ads.find_channel("temp") if ads is not None else None

    # Two xadc devices are present on the currently deployed image despite
    # user-api.md documenting one. Their common rails agree within ADC noise.
    # Select the first context device deterministically and identify rails by
    # their channel label, never by voltageN numbering.
    xadc = _find_device(context, _XADC_DEVICE_NAME)
    xadc_temp_input = xadc.find_channel("temp0") if xadc is not None else None
    xadc_rails: list[tuple[str, iio.Channel]] = []
    if xadc is not None:
        for channel in xadc.channels:
            label_attr = channel.attrs.get("label")
            if label_attr is None or "raw" not in channel.attrs or "scale" not in channel.attrs:
                continue
            label = label_attr.value.strip()
            if label.startswith("vcc"):
                xadc_rails.append((label, channel))

    return _GlobalDevices(
        hat_temp_input=hat_temp_input,
        ads=ads,
        ads_temp_input=ads_temp_input,
        xadc_temp_input=xadc_temp_input,
        xadc_rails=tuple(sorted(xadc_rails, key=lambda item: item[0])),
        clock=_find_device(context, _CLOCK_DEVICE_NAME),
    )


def _discover_devices(
    context: iio.Context,
    channel: int,
    dac_channel: int,
) -> _IDSDevices:
    adc = _find_device(context, *_ADC_DEVICE_NAMES)
    if adc is None:
        raise IIOIDSUnavailableError(
            "IIO IDS backend: no mcp3564/mcp3564r device found"
        )

    feedback_label = _HV_FEEDBACK_LABELS[channel]
    hv_feedback = _find_labelled_channel(adc, feedback_label)
    if hv_feedback is None:
        raise IIOIDSUnavailableError(
            f"IIO IDS backend: {adc.name} has no channel labelled "
            f"'{feedback_label}'"
        )

    adc_temperature = _find_labelled_channel(adc, "temperature")
    if adc_temperature is None:
        raise IIOIDSUnavailableError(
            f"IIO IDS backend: {adc.name} has no channel labelled 'temperature'"
        )

    dac = _find_device(context, _DAC_DEVICE_NAME)
    if dac is None:
        raise IIOIDSUnavailableError(
            f"IIO IDS backend: no {_DAC_DEVICE_NAME} device found"
        )
    hv_output = dac.find_channel(f"voltage{dac_channel}", True)
    if hv_output is None:
        raise IIOIDSUnavailableError(
            f"IIO IDS backend: {_DAC_DEVICE_NAME} has no output channel "
            f"voltage{dac_channel}"
        )

    # Live audit on 2026-09-14: 192.168.10.128 exposes cha_temp/chb_temp and
    # HAT_temp, while 192.168.10.135 and 10.7.0.121 expose only HAT_temp.
    # The HV DAC and ADC paths are complete on all three boards, so a missing
    # per-channel thermometer must not suppress the entire PSU backend.
    preferred_temp_label = _TMP_LABELS[channel]
    digital_temp = _find_labelled_device(
        context, _TMP_DEVICE_NAME, preferred_temp_label,
    )
    digital_temp_label: str | None = preferred_temp_label
    if digital_temp is None:
        digital_temp = _find_labelled_device(
            context, _TMP_DEVICE_NAME, _GLOBAL_TMP_LABEL,
        )
        digital_temp_label = _GLOBAL_TMP_LABEL if digital_temp is not None else None
    digital_temp_input = digital_temp.find_channel("temp") if digital_temp is not None else None
    if digital_temp_input is None:
        digital_temp_label = None

    ads = _find_device(context, _ADS_DEVICE_NAME)
    ads_temp_input = ads.find_channel("temp") if ads is not None else None

    return _IDSDevices(
        adc=adc,
        hv_feedback=hv_feedback,
        adc_temperature=adc_temperature,
        dac=dac,
        hv_output=hv_output,
        digital_temp_input=digital_temp_input,
        digital_temp_label=digital_temp_label,
        ads_temp_input=ads_temp_input,
    )


class IIOIDSBackend(IDSBackend):
    """Implement the non-SiPM IDS interface through network IIO devices.

    ``channel`` is zero-based, matching the IIO digitizer backend. The
    AD5686R driver exposes four unlabelled outputs; by default channel 0/1
    uses voltage0/1. ``dac_channel`` permits an explicit board routing when
    a deployment uses a different output.

    Two IIO contexts are used so periodic PSU readback can run independently
    from GUI-thread writes. This follows the same thread-isolation rule as
    :class:`IIODigitizerBackend`.
    """

    def __init__(
        self,
        channel: int,
        uri: str = "ip:192.168.10.128:30431",
        *,
        dac_channel: int | None = None,
        hv_gain_v_per_mv: float = _DEFAULT_HV_GAIN_V_PER_MV,
    ) -> None:
        if channel not in (0, 1):
            raise ValueError("IIO IDS channel must be 0 or 1")
        if hv_gain_v_per_mv <= 0:
            raise ValueError("hv_gain_v_per_mv must be positive")

        selected_dac_channel = channel if dac_channel is None else dac_channel
        if selected_dac_channel < 0:
            raise ValueError("dac_channel must be non-negative")

        self._ch = channel
        self._uri = uri
        self._hv_gain_v_per_mv = float(hv_gain_v_per_mv)
        self._ctx = iio.Context(uri)
        self._devices = _discover_devices(
            self._ctx, channel, selected_dac_channel,
        )
        self._monitor_ctx = iio.Context(uri)
        self._monitor_devices = _discover_devices(
            self._monitor_ctx, channel, selected_dac_channel,
        )
        # GlobalController polls this context from its own QThread. It must
        # not share either GUI/config I/O or the per-channel PSU monitor's
        # context; remote libiio contexts are not thread-safe.
        self._global_ctx = iio.Context(uri)
        self._global_devices = _discover_global_devices(self._global_ctx)
        # Created lazily by TemperatureCorrectionWorker on its own thread.
        # Remote IIO contexts are not thread-safe, so this ADS5407 read must
        # not reuse either monitoring context above.
        self._temperature_correction_ctx: iio.Context | None = None
        self._temperature_correction_ads: iio.Channel | None = None

        self._state_lock = threading.RLock()
        self._monitor_lock = threading.RLock()
        self._hv_compens_ct = 0.0
        self._hv_compens_tref = 20.0
        self._hv_compens_mode = 0
        self._temp_digital_mode = 0
        self._nominal_hv_voltage = self._read_dac_voltage(self._devices)
        self._compensated_hv_voltage = self._nominal_hv_voltage

        digital_temp_label = self._devices.digital_temp_label
        if digital_temp_label == _GLOBAL_TMP_LABEL:
            log.warning(
                "IIO IDS backend ch%d: no %s TMP117; using shared %s sensor",
                channel,
                _TMP_LABELS[channel],
                _GLOBAL_TMP_LABEL,
            )
        elif digital_temp_label is None:
            log.warning(
                "IIO IDS backend ch%d: no channel or shared TMP117; digital "
                "temperature is unavailable but HV control remains available",
                channel,
            )

        log.info(
            "IIO IDS backend: connected ch%d (DAC voltage%d, %s, %s) to %s",
            channel,
            selected_dac_channel,
            _HV_FEEDBACK_LABELS[channel],
            _TMP_LABELS[channel],
            uri,
        )

    def close(self) -> None:
        # pylibiio Context has no public close() in the supported 0.25/0.26
        # bindings. Owned contexts are released with this backend object.
        log.info("IIO IDS backend: closing ch%d", self._ch)

    # ---- scaled IIO helpers ----

    def _read_dac_voltage(self, devices: _IDSDevices) -> float:
        raw = int(devices.hv_output.attrs["raw"].value)
        scale_mv = float(devices.hv_output.attrs["scale"].value)
        return raw * scale_mv * self._hv_gain_v_per_mv

    def _write_dac_voltage(self, devices: _IDSDevices, voltage: float) -> float:
        bounded = min(max(float(voltage), _HV_MIN_VOLTAGE), _HV_MAX_VOLTAGE)
        scale_mv = float(devices.hv_output.attrs["scale"].value)
        raw = round(bounded / (scale_mv * self._hv_gain_v_per_mv))
        raw = min(max(raw, _DAC_MIN_RAW), _DAC_MAX_RAW)
        devices.hv_output.attrs["raw"].value = str(raw)
        return raw * scale_mv * self._hv_gain_v_per_mv

    def _read_hv_feedback(self, devices: _IDSDevices) -> float:
        raw = int(devices.hv_feedback.attrs["raw"].value)
        scale_mv = float(devices.adc.attrs["scale"].value)
        return raw * scale_mv * self._hv_gain_v_per_mv

    @staticmethod
    def _read_digital_temperature(devices: _IDSDevices) -> float:
        if devices.digital_temp_input is None:
            return float("nan")
        raw = int(devices.digital_temp_input.attrs["raw"].value)
        # The TMP117 IIO scale is millidegrees Celsius per raw count.
        scale_millidegrees = float(
            devices.digital_temp_input.attrs["scale"].value
        )
        return raw * scale_millidegrees / 1000.0

    @staticmethod
    def _read_analog_temperature(devices: _IDSDevices) -> float:
        raw = int(devices.adc_temperature.attrs["raw"].value)
        scale_mv = float(devices.adc.attrs["scale"].value)
        sensor_mv = raw * scale_mv
        return (
            sensor_mv - _MCP_TEMP_OFFSET_MV
        ) / _MCP_TEMP_SLOPE_MV_PER_C

    @staticmethod
    def _read_ads_temperature(devices: _IDSDevices) -> float:
        if devices.ads_temp_input is None:
            raise RuntimeError(
                f"IIO IDS backend: no {_ADS_DEVICE_NAME} temp channel found"
            )
        # The board-specific ADS5407 IIO device exposes an already-converted
        # integer temperature as temp.raw and no scale attribute.
        return float(devices.ads_temp_input.attrs["raw"].value)

    def _temperature_for_compensation(self, devices: _IDSDevices) -> float | None:
        if self._hv_compens_mode == 0:
            return None
        if self._hv_compens_mode == 1:
            if devices.digital_temp_input is None:
                raise RuntimeError(
                    "IIO IDS backend: digital HV temperature compensation "
                    "requires a channel-specific or HAT_temp TMP117 sensor"
                )
            return self._read_digital_temperature(devices)
        if self._hv_compens_mode == 2:
            return self._read_analog_temperature(devices)
        raise NotImplementedError(_COMPENSATION_MODE_UNSUPPORTED)

    def _apply_compensation(self, devices: _IDSDevices) -> float:
        temperature = self._temperature_for_compensation(devices)
        target = self._nominal_hv_voltage
        if temperature is not None:
            target += self._hv_compens_ct * (
                temperature - self._hv_compens_tref
            )
        self._compensated_hv_voltage = self._write_dac_voltage(devices, target)
        return self._compensated_hv_voltage

    # ---- versions ----

    def get_versions(self) -> list[int]:
        """Return no legacy IDS IP versions; these IIO drivers expose none."""
        return []

    def get_ads_temp(self) -> float:
        with self._monitor_lock:
            return self._read_ads_temperature(self._monitor_devices)

    def get_ads_temp_for_correction(self) -> float:
        """Read ADS5407 only through the correction worker's IIO context."""
        if self._temperature_correction_ctx is None:
            context = iio.Context(self._uri)
            ads = _find_device(context, _ADS_DEVICE_NAME)
            ads_temp = ads.find_channel("temp") if ads is not None else None
            if ads_temp is None:
                raise RuntimeError(
                    f"IIO IDS backend: no {_ADS_DEVICE_NAME} temp channel found"
                )
            self._temperature_correction_ctx = context
            self._temperature_correction_ads = ads_temp
        if self._temperature_correction_ads is None:
            raise RuntimeError(
                f"IIO IDS backend: no {_ADS_DEVICE_NAME} temp channel found"
            )
        return float(self._temperature_correction_ads.attrs["raw"].value)

    @staticmethod
    def _read_scaled_temperature(channel: iio.Channel) -> float:
        raw = float(channel.attrs["raw"].value)
        scale = float(channel.attrs["scale"].value)
        correction_attr = channel.attrs.get("offset") or channel.attrs.get("calibbias")
        correction = float(correction_attr.value) if correction_attr is not None else 0.0
        # IIO temperature scales are millidegrees Celsius per raw count.
        return (raw + correction) * scale / 1000.0

    def get_global_diagnostics(self) -> list[GlobalDiagnosticReading]:
        """Read common board diagnostics through the dedicated context.

        Attribute names and units follow user-api.md plus live discovery on
        2026-08-11. ADS5407 temperature remains a raw register code because
        the driver and data sheet expose no conversion function.
        """
        devices = self._global_devices
        readings: list[GlobalDiagnosticReading] = []

        if devices.hat_temp_input is not None:
            readings.append(GlobalDiagnosticReading(
                "hat_temperature", "HAT temperature",
                self._read_scaled_temperature(devices.hat_temp_input), "°C", 1,
            ))
        if devices.xadc_temp_input is not None:
            readings.append(GlobalDiagnosticReading(
                "fpga_temperature", "FPGA temperature",
                self._read_scaled_temperature(devices.xadc_temp_input), "°C", 1,
            ))
        if devices.ads_temp_input is not None:
            readings.append(GlobalDiagnosticReading(
                "ads_temperature_raw", "ADS5407 temperature code",
                int(devices.ads_temp_input.attrs["raw"].value), "raw",
            ))
        if devices.ads is not None and "sampling_frequency" in devices.ads.attrs:
            readings.append(GlobalDiagnosticReading(
                "sampling_frequency", "ADC sampling frequency",
                int(devices.ads.attrs["sampling_frequency"].value), "Hz",
            ))

        if devices.clock is not None:
            for key, label, healthy_value in (
                ("pll_locked", "PLL locked", True),
                ("ref_ok", "Clock reference present", True),
                ("pll_unlock", "PLL unlock indicator", False),
            ):
                if key not in devices.clock.attrs:
                    continue
                value = bool(int(devices.clock.attrs[key].value))
                readings.append(GlobalDiagnosticReading(
                    key, label, value, healthy=value is healthy_value,
                ))

        for label, channel in devices.xadc_rails:
            raw = float(channel.attrs["raw"].value)
            scale_mv = float(channel.attrs["scale"].value)
            readings.append(GlobalDiagnosticReading(
                f"rail_{label}", label.upper(), raw * scale_mv / 1000.0, "V", 3,
            ))

        return readings

    # ---- removed SiPM bias supply ----

    def set_sipm_enable(self, val: int) -> None:
        raise NotImplementedError(_SIPM_UNSUPPORTED)

    def set_sipm_voltage(self, val: float) -> None:
        raise NotImplementedError(_SIPM_UNSUPPORTED)

    def get_sipm_adc_voltage(self) -> float:
        raise NotImplementedError(_SIPM_UNSUPPORTED)

    def get_sipm_adc_current(self) -> float:
        raise NotImplementedError(_SIPM_UNSUPPORTED)

    def get_sipm_overload(self) -> int:
        # HVSupply.sipm_available() treats only 0/1 as present. A sentinel is
        # preferable to aliasing an unrelated raw channel and lets the GUI
        # hide the removed controls without handling an exception.
        return -1

    def set_sipm_compens_ct(self, val: float) -> None:
        raise NotImplementedError(_SIPM_UNSUPPORTED)

    def set_sipm_compens_tref(self, val: float) -> None:
        raise NotImplementedError(_SIPM_UNSUPPORTED)

    def set_sipm_compens_mode(self, val: int) -> None:
        raise NotImplementedError(_SIPM_UNSUPPORTED)

    def get_sipm_compens_output(self) -> float:
        raise NotImplementedError(_SIPM_UNSUPPORTED)

    # ---- HV bias supply ----

    def set_hv_voltage(self, val: float) -> None:
        with self._state_lock:
            self._nominal_hv_voltage = float(val)
            self._apply_compensation(self._devices)

    def get_hv_adc_voltage(self) -> float:
        with self._monitor_lock:
            return self._read_hv_feedback(self._monitor_devices)

    # ---- software HV temperature compensation ----

    def set_hv_compens_ct(self, val: float) -> None:
        with self._state_lock:
            self._hv_compens_ct = float(val)
            self._apply_compensation(self._devices)

    def set_hv_compens_tref(self, val: float) -> None:
        with self._state_lock:
            self._hv_compens_tref = float(val)
            self._apply_compensation(self._devices)

    def set_hv_compens_mode(self, val: int) -> None:
        if val not in (0, 1, 2):
            raise NotImplementedError(_COMPENSATION_MODE_UNSUPPORTED)
        with self._state_lock:
            self._hv_compens_mode = int(val)
            self._apply_compensation(self._devices)

    def get_hv_compens_output(self) -> float:
        # The IIO drivers expose no autonomous compensation register. Refresh
        # and apply the software correction during the PSU worker's normal
        # polling cycle, using its dedicated context.
        with self._state_lock, self._monitor_lock:
            return self._apply_compensation(self._monitor_devices)

    # ---- temperature sensors ----

    def get_temp_analog(self) -> float:
        with self._monitor_lock:
            return self._read_analog_temperature(self._monitor_devices)

    def set_temp_digital_enable(self, val: int) -> None:
        if val not in (0, 1, 2):
            raise ValueError("temp_digital_enable must be 0, 1 or 2")
        # iio_tmp117.c exposes only raw and scale: the sensor continuously
        # converts and has no enable/mode attribute. Retain the requested
        # legacy mode as software state while leaving the hardware untouched.
        self._temp_digital_mode = int(val)

    def get_temp_digital_status(self) -> int:
        # iio_tmp117.c exposes no communication-status attribute. Presence
        # of a usable labelled temp channel is the only available status.
        return int(self._monitor_devices.digital_temp_input is not None)

    def get_temp_digital(self) -> float:
        with self._monitor_lock:
            return self._read_digital_temperature(self._monitor_devices)

    def get_safe_shutdown_voltage(self) -> float:
        """Return the IIO DAC's real off state for HVSupply.safe_shutdown()."""
        return 0.0
