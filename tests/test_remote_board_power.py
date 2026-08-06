from pathlib import Path

import pytest

from nlab.utils.remote_board_power import power_command_was_delivered, ssh_arguments


def test_ssh_arguments_are_noninteractive_and_use_restricted_account() -> None:
    key_path = Path(r"C:\Users\tester\.ssh\nlab_board_power_ed25519")

    args = ssh_arguments("192.168.10.128", key_path, "reboot")

    assert args == [
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
        "nlab-reboot@192.168.10.128",
        "reboot",
    ]


@pytest.mark.parametrize("command", ["check", "reboot", "shutdown"])
def test_ssh_arguments_accept_only_dispatcher_commands(command: str) -> None:
    assert ssh_arguments("board.local", Path("key"), command)[-1] == command


def test_ssh_arguments_reject_invalid_input() -> None:
    with pytest.raises(ValueError, match="host is empty"):
        ssh_arguments(" ", Path("key"), "check")
    with pytest.raises(ValueError, match="Unsupported"):
        ssh_arguments("board.local", Path("key"), "whoami")


@pytest.mark.parametrize(
    ("exit_code", "stderr"),
    [
        (0, ""),
        (255, "Connection to 192.168.10.128 closed by remote host."),
        (255, "client_loop: send disconnect: Broken pipe"),
        (255, "Connection reset by 192.168.10.128 port 22"),
    ],
)
def test_power_command_accepts_clean_exit_or_expected_disconnect(
    exit_code: int, stderr: str,
) -> None:
    assert power_command_was_delivered(exit_code, stderr)


@pytest.mark.parametrize(
    ("exit_code", "stderr"),
    [
        (64, "Denied. Allowed commands: check, reboot, shutdown."),
        (255, "Permission denied (publickey)."),
        (255, "ssh: connect to host 192.168.10.128 port 22: Connection timed out"),
        (1, "unspecified error"),
    ],
)
def test_power_command_rejects_command_authentication_and_transport_failures(
    exit_code: int, stderr: str,
) -> None:
    assert not power_command_was_delivered(exit_code, stderr)
