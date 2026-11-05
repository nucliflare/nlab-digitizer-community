"""Exercise GUI collection ownership using cycles and CPU-only worker threads."""

import gc
import threading

import pytest
from PySide6.QtCore import QObject

from nlab.views.gui_gc import install_gui_garbage_collector


@pytest.fixture
def fresh_policy(qapp):
    original_enabled = gc.isenabled()
    original_thresholds = gc.get_threshold()
    previous = getattr(qapp, "_nlab_gui_garbage_collector", None)
    was_active = previous is not None and not previous.closed
    if was_active:
        previous.close()
    gc.enable()
    try:
        yield
    finally:
        collector = getattr(qapp, "_nlab_gui_garbage_collector", None)
        if collector is not None:
            collector.close()
        gc.set_threshold(*original_thresholds)
        gc.collect()
        if was_active:
            gc.enable()
            install_gui_garbage_collector()
        elif original_enabled:
            gc.enable()
        else:
            gc.disable()


def test_worker_allocations_cannot_finalize_qt_cycles_off_gui_thread(fresh_policy):
    collector = install_gui_garbage_collector()
    assert collector is not None
    collector.timer.stop()  # Deliberately hold collection until the worker has allocated.
    gc.set_threshold(10, 1, 1)
    finalized = []

    class Probe(QObject):
        def __del__(self):
            finalized.append(threading.get_ident())

    probe = Probe()
    probe.cycle = probe
    del probe

    def allocate():
        for _ in range(1000):
            cycle = []
            cycle.append(cycle)

    worker = threading.Thread(target=allocate)
    worker.start()
    worker.join(5)
    assert not worker.is_alive()
    assert finalized == []
    collector.check()
    assert finalized == [threading.get_ident()]


def test_install_is_idempotent_and_close_restores_automatic_collection(fresh_policy, qapp):
    collector = install_gui_garbage_collector()
    assert collector is not None
    assert install_gui_garbage_collector() is collector
    assert collector.parent() is qapp and collector.timer.isActive()
    assert not gc.isenabled()
    collector.close()
    collector.close()
    assert gc.isenabled() and not collector.timer.isActive()
    replacement = install_gui_garbage_collector()
    assert replacement is not collector


def test_close_preserves_preexisting_disabled_collection(fresh_policy):
    gc.disable()
    collector = install_gui_garbage_collector()
    assert collector is not None
    collector.close()
    assert not gc.isenabled()


@pytest.mark.parametrize("operation", ["install", "check", "close"])
def test_collection_policy_rejects_worker_thread_calls(fresh_policy, operation):
    collector = install_gui_garbage_collector()
    assert collector is not None
    errors = []

    def invoke():
        try:
            if operation == "install":
                install_gui_garbage_collector()
            else:
                getattr(collector, operation)()
        except RuntimeError as error:
            errors.append(str(error))

    worker = threading.Thread(target=invoke)
    worker.start()
    worker.join(5)
    assert not worker.is_alive()
    assert len(errors) == 1 and "GUI thread" in errors[0]
    assert not collector.closed and not gc.isenabled()
