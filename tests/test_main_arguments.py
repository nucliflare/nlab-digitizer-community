from nlab.main import _int_value, _parse_arguments, _string_value


def test_gui_command_line_accepts_config_ip_and_port() -> None:
    args, qt_args = _parse_arguments(
        [
            "--config",
            "experiment.yaml",
            "--ip",
            "192.0.2.10",
            "--port",
            "30431",
            "-platform",
            "offscreen",
        ]
    )

    assert str(args.config) == "experiment.yaml"
    assert args.ip == "192.0.2.10"
    assert args.port == 30431
    assert qt_args == ["-platform", "offscreen"]


def test_yaml_connection_values_are_type_checked() -> None:
    assert _string_value("board.local") == "board.local"
    assert _string_value(123) is None
    assert _int_value(30431) == 30431
    assert _int_value(True) is None
