"""Opt-in, bounded live smoke test for ABI2 and Scope periodic sums.

Usage: python examples/verify_iio_update.py --uri ip:192.168.10.128:30431
       --exercise --output-dir <new-output-directory>

Discovery is read-only unless --exercise is given. Records original state,
keeps measurements stopped, closes all workers/buffers, restores valid fields,
and reports reset-only values that cannot legally be reapplied. It does not
qualify pulse response, channel wiring, completion fencing, or fault recovery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend
from nlab.hardware.digitizer.digitizer import Digitizer
from nlab.hardware.digitizer.mca_parameters import (
    DEFAULT_PARAMETERS,
    PARAMETERS,
    validate_parameters,
)
from nlab.hardware.digitizer.scope import TriggerMode
from nlab.hardware.digitizer.scope_sums import ScopeSumCapture, read_sum_capture


def verify_channel(uri: str, channel: int, output: Path, exercise: bool) -> dict:
    backend = IIODigitizerBackend(channel, uri)
    device = Digitizer(backend)
    scope, mca = device.scope, device.mca
    report: dict = {
        "channel": channel,
        "contract": mca.parameter_contract(),
        "scope_identity": backend.scope_sum_identity(),
        "checks": [],
        "restore": [],
    }
    original = {}
    scope_original = {}
    ack = None
    changed = False
    debug_original = {}
    try:
        original = mca.parameter_snapshot()
        scope_original = dict(
            frame=scope.get_frame_samples(),
            pre=scope.get_pretrigger_samples(),
            mode=int(scope.get_trigger_mode()),
            gap=scope.get_frame_period_cycles(),
        )
        ack = backend._pp_attr_get("parameter_abi_ack")
        report["original_mca"] = original
        report["original_scope"] = scope_original
        debug_original = {
            "debug_signal1": mca.get_mem1_sig_select(),
            "debug_signal2": mca.get_mem2_sig_select(),
        }
        report["original_ack"] = ack
        report["scope_modes"] = [int(mode) for mode in scope.supported_trigger_modes()]
        if not exercise:
            return report
        if (
            mca.get_global_enable()
            or mca.dma_active()
            or scope.get_enable()
            or backend.scope_dma_active()
        ):
            raise RuntimeError("Target is owned by an acquisition; refusing the smoke test")
        mca.stop()
        mca.prepare_configuration()
        changed = True
        # Boundary checks use native quantities, never client register encodings.
        for field in (
            "pretrigger_samples",
            "frame_samples",
            "cfd_delay_samples",
            "crrc2_cdelay",
            "trapezoid_r_samples",
            "trapezoid_beta_raw",
            "cfd_factor_raw",
        ):
            low, high, step = PARAMETERS[field]
            for value in (low, high):
                backend._pp_attr_set(field, str(value))
                actual = int(backend._pp_attr_get(field))
                if actual != value:
                    raise AssertionError((field, value, actual))
            report["checks"].append({"parameter": field, "bounds": [low, high], "step": step})
        mca.configure_parameters(dict(DEFAULT_PARAMETERS))
        mca.start()
        mca.set_mem1_sig_select(1)
        mca.set_mem2_sig_select(2)
        if not mca.get_global_enable():
            raise AssertionError("Live debug selection unexpectedly stopped MCA")
        mca.stop()
        report["checks"].append("ABI2 START/STOP and live debug routing; no completion claim")
        if TriggerMode.PERIODIC_SUM in scope.supported_trigger_modes():
            scope.stop()
            scope.set_pretrigger_samples(0)
            scope.set_frame_samples(256)
            scope.set_frame_period_cycles(1000)
            scope.set_trigger_mode(TriggerMode.PERIODIC_SUM)
            path = output / f"channel-{channel}-sums.h5"
            result = ScopeSumCapture(backend, channel).run(path, max_packets=2)
            result.pop("points", None)
            reopened = sum(len(packet.records) for packet in read_sum_capture(path))
            if reopened != result["windows"]:
                raise AssertionError("Recorded/reopened window counts differ")
            report["checks"].append({"sum_capture": result, "reopened_windows": reopened})
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if changed:
            for label, stop in (("scope", scope.stop), ("mca", mca.stop)):
                try:
                    stop()
                except Exception as exc:
                    report["restore"].append(f"{label} stop: {exc}")
            try:
                backend.close_scope_sum_capture()
            except Exception as exc:
                report["restore"].append(f"Scope close: {exc}")
            if original:
                # A reset-only R=4 snapshot is not a valid ABI2 preset.
                # Disable algorithms while restoring each legal original field.
                enables = [name for name in PARAMETERS if name.endswith("_enable")]
                for name in enables:
                    try:
                        backend._pp_attr_set(name, "0")
                    except Exception as exc:
                        report["restore"].append(f"{name}: {exc}")
                for name, value in original.items():
                    try:
                        validate_parameters({name: value})
                        backend._pp_attr_set(name, str(value))
                    except Exception as exc:
                        report["restore"].append(f"{name}={value} not restored: {exc}")
                for name, value in debug_original.items():
                    try:
                        backend._pp_attr_set(name, str(value))
                    except Exception as exc:
                        report["restore"].append(f"{name}: {exc}")
                if ack is not None:
                    try:
                        backend._pp.attrs["parameter_abi_ack"].value = ack
                    except Exception as exc:
                        report["restore"].append(f"ABI ack: {exc}")
            if scope_original:
                for field, setter in (
                    ("mode", backend.set_edge),
                    ("frame", scope.set_frame_samples),
                    ("gap", scope.set_frame_period_cycles),
                    ("pre", scope.set_pretrigger_samples),
                ):
                    try:
                        setter(scope_original[field])
                    except Exception as exc:
                        report["restore"].append(
                            f"Scope {field}={scope_original[field]} not restored: {exc}"
                        )
            report["final_mca_enable"] = mca.get_global_enable()
            report["final_scope_enable"] = scope.get_enable()
            report["final_lm_active"] = mca.dma_active()
            report["final_scope_dma_active"] = backend.scope_dma_active()
        device.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="ip:192.168.10.128:30431")
    parser.add_argument("--channel", type=int, choices=(0, 1), action="append")
    parser.add_argument("--exercise", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "uri": args.uri,
        "driver_reference": "bb8c57e",
        "live": True,
        "exercise": args.exercise,
        "channels": [],
    }
    for channel in args.channel or (0, 1):
        report["channels"].append(verify_channel(args.uri, channel, args.output_dir, args.exercise))
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if any("error" in result for result in report["channels"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
