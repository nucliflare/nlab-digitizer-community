import sys as _sys
import os as _os

_grpc_path = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "grpc", "generated")
)
if _os.path.isdir(_grpc_path) and _grpc_path not in _sys.path:
    _sys.path.insert(0, _grpc_path)

from .digitizer import Digitizer
from .scope import (
    Scope,
    ScopeParam,
    TriggerMode,
    RangeSpec,
    ListSpec,
    ParameterSpec,
    ScopeSettingEntry,
    PARAMETER_SPECS,
)
from .mca import (
    MultiChannelAnalyzer,
    MCAParam,
    MCASettingEntry,
    MCA_PARAMETER_SPECS,
)
from .hv import (
    HVSupply,
    HVParam,
    HVSettingEntry,
    HV_PARAMETER_SPECS,
)
from .backends import ScopeBackend, MCABackend, DigitizerBackend, GrpcDigitizerBackend, IIODigitizerBackend

__all__ = [
    "Digitizer",
    "Scope",
    "ScopeParam",
    "TriggerMode",
    "RangeSpec",
    "ListSpec",
    "ParameterSpec",
    "ScopeSettingEntry",
    "PARAMETER_SPECS",
    "MultiChannelAnalyzer",
    "MCAParam",
    "MCASettingEntry",
    "MCA_PARAMETER_SPECS",
    "HVSupply",
    "HVParam",
    "HVSettingEntry",
    "HV_PARAMETER_SPECS",
    "ScopeBackend",
    "MCABackend",
    "DigitizerBackend",
    "GrpcDigitizerBackend",
    "IIODigitizerBackend",
]
