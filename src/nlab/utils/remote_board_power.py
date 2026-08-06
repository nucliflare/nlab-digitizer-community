"""Restricted SSH command construction for remote board power control."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

BoardPowerCommand = Literal["reboot", "shutdown"]

_EXPECTED_DISCONNECT_MESSAGES = (
    "closed by remote host",
    "connection closed by",
    "connection reset by",
    "broken pipe",
)


def ssh_arguments(host: str, key_path: Path, command: str) -> list[str]:
    """Build an OpenSSH invocation without involving a command shell.

    The matching Dropbear ``authorized_keys`` entry forces the board-side
    dispatcher, so only ``check``, ``reboot``, and ``shutdown`` can succeed.
    These client options also make GUI use strictly non-interactive.
    """
    if not host.strip():
        raise ValueError("Remote board host is empty")
    if command not in {"check", "reboot", "shutdown"}:
        raise ValueError(f"Unsupported remote board command: {command}")

    return [
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "ConnectionAttempts=1",
        "-i",
        str(key_path),
        f"nlab-reboot@{host}",
        command,
    ]


def power_command_was_delivered(exit_code: int, stderr: str) -> bool:
    """Accept a clean SSH exit or the disconnect caused by board power-off.

    SysV ``reboot``/``poweroff`` can tear down Dropbear before it sends the
    remote exit status. OpenSSH then returns 255 even though the command was
    accepted. We only accept 255 when its diagnostic specifically describes
    that expected connection loss; authentication and transport errors remain
    failures. The caller performs a successful ``check`` immediately before
    sending the destructive command.
    """
    if exit_code == 0:
        return True
    if exit_code != 255:
        return False
    message = stderr.casefold()
    return any(fragment in message for fragment in _EXPECTED_DISCONNECT_MESSAGES)
