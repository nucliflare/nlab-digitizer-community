"""Static guards for Qt worker/thread ownership conventions."""

from __future__ import annotations

import re
from pathlib import Path

_SELF_DELETION_CONNECTION = re.compile(
    r"(?P<owner>(?:self\.)?[A-Za-z_][A-Za-z0-9_]*)\.finished\.connect\("
    r"\s*(?P=owner)\.deleteLater\s*\)"
)

_GUI_THREAD_FINISH_CONNECTION = re.compile(
    r"(?P<owner>(?:self\.)?[A-Za-z_][A-Za-z0-9_]*thread)\.finished\.connect\("
    r"\s*self\._[A-Za-z0-9_]*finished(?P<arguments>.*?)\)",
    re.DOTALL,
)


def test_workers_are_deleted_by_their_finished_threads() -> None:
    source_root = Path(__file__).parents[1] / "src" / "nlab"
    offenders: list[str] = []
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in _SELF_DELETION_CONNECTION.finditer(text):
            owner = match.group("owner")
            if owner.rsplit(".", 1)[-1].endswith("worker"):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(source_root)}:{line}")

    assert not offenders, (
        "workers must connect thread.finished to worker.deleteLater; unsafe "
        f"worker self-deletion connections found at {', '.join(offenders)}"
    )


def test_gui_thread_finish_handlers_are_explicitly_queued() -> None:
    source_root = Path(__file__).parents[1] / "src" / "nlab"
    offenders: list[str] = []
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in _GUI_THREAD_FINISH_CONNECTION.finditer(text):
            if "Qt.ConnectionType.QueuedConnection" not in match.group("arguments"):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(source_root)}:{line}")

    assert not offenders, (
        "QThread.finished GUI cleanup handlers must be explicitly queued; "
        f"unsafe connections found at {', '.join(offenders)}"
    )
