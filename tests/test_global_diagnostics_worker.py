from types import SimpleNamespace
from unittest.mock import Mock

from nlab.workers.global_diagnostics_worker import GlobalDiagnosticsWorker


def test_shutdown_requested_before_tick_skips_hardware_read() -> None:
    device = SimpleNamespace(get_global_diagnostics=Mock())
    worker = GlobalDiagnosticsWorker(device)  # type: ignore[arg-type]
    finished = Mock()
    worker.finished.connect(finished)

    worker.request_shutdown()
    worker._tick()

    device.get_global_diagnostics.assert_not_called()
    finished.assert_called_once_with()


def test_shutdown_requested_during_read_finishes_without_emitting_readback() -> None:
    device = SimpleNamespace()
    worker = GlobalDiagnosticsWorker(device)  # type: ignore[arg-type]
    finished = Mock()
    readback = Mock()
    worker.finished.connect(finished)
    worker.readback.connect(readback)

    def read_and_request_shutdown() -> list[object]:
        worker.request_shutdown()
        return [object()]

    device.get_global_diagnostics = Mock(side_effect=read_and_request_shutdown)

    worker._tick()

    finished.assert_called_once_with()
    readback.assert_not_called()
