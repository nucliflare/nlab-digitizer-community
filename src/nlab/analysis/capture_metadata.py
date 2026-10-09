"""Read optional digitizer identity metadata stored beside or inside captures."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import yaml

log = logging.getLogger(__name__)


def capture_digitizer_info(path: Path) -> dict[str, object]:
    """Return available connection metadata without guessing absent identity fields."""
    configuration = _capture_configuration(Path(path))
    connection = configuration.get("connection")
    if not isinstance(connection, Mapping):
        return {"device endpoint": "Not embedded in source file"}

    result: dict[str, object] = {}
    host = connection.get("ip")
    port = connection.get("port")
    if host is not None:
        endpoint = str(host)
        if port is not None:
            endpoint = f"{endpoint}:{port}"
        result["device endpoint"] = endpoint
    else:
        result["device endpoint"] = "Not embedded in source file"
    if connection.get("backend") is not None:
        result["backend"] = connection["backend"]
    if connection.get("channels") is not None:
        result["configured channels"] = connection["channels"]
    return result


def _capture_configuration(path: Path) -> Mapping[str, Any]:
    if path.suffix.lower() in {".h5", ".hdf5"}:
        try:
            with h5py.File(path, "r", swmr=True) as file:
                if "configuration_yaml" in file:
                    value = file["configuration_yaml"][()]
                    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    return _yaml_mapping(text, path)
        except (OSError, ValueError):
            log.warning("Could not read embedded capture metadata from %s", path, exc_info=True)
    elif path.suffix.lower() == ".root":
        try:
            import uproot

            with uproot.open(path) as file:
                if "configuration_yaml" in file:
                    return _yaml_mapping(str(file["configuration_yaml"]), path)
        except (OSError, ValueError):
            log.warning("Could not read embedded capture metadata from %s", path, exc_info=True)

    candidates = (path.with_suffix(".yaml"), path.parent / "configuration.yaml")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            return _yaml_mapping(candidate.read_text(encoding="utf-8"), candidate)
        except OSError:
            log.warning("Could not read capture metadata from %s", candidate, exc_info=True)
    return {}


def _yaml_mapping(text: str, source: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        log.warning("Ignoring invalid capture metadata YAML from %s", source, exc_info=True)
        return {}
    return value if isinstance(value, Mapping) else {}
