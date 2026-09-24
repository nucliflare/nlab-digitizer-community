from __future__ import annotations

from abc import ABC, abstractmethod


class CurrentMonitorClient(ABC):
    """Small worker-owned connection for the live FPGA IIR readout.

    The client deliberately exposes only the one operation needed by the
    current monitor.  Keeping it separate from ``Digitizer`` prevents a
    background polling thread from sharing an IIO context or gRPC channel
    with GUI/configuration calls.
    """

    @abstractmethod
    def read_raw(self) -> int:
        """Return the current signed IIR output code."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release the worker-owned transport."""
        ...
