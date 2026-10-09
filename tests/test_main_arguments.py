import logging

from nlab.main import _configure_logging, _int_value, _parse_arguments, _string_value


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


def test_logging_configuration_suppresses_matplotlib_font_debug_records() -> None:
    font_logger = logging.getLogger("matplotlib.font_manager")
    original_level = font_logger.level
    try:
        font_logger.setLevel(logging.NOTSET)

        _configure_logging()

        assert font_logger.level == logging.WARNING
    finally:
        font_logger.setLevel(original_level)
