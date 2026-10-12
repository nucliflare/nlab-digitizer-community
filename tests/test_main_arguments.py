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


def test_logging_configuration_suppresses_report_renderer_debug_records() -> None:
    logger_names = (
        "matplotlib",
        "matplotlib.colorbar",
        "matplotlib.font_manager",
        "PIL",
        "PIL.PngImagePlugin",
    )
    loggers = [logging.getLogger(name) for name in logger_names]
    original_levels = [logger.level for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(logging.NOTSET)

        _configure_logging()

        assert logging.getLogger("matplotlib").level == logging.WARNING
        assert logging.getLogger("PIL").level == logging.WARNING
        for logger in loggers:
            assert logger.getEffectiveLevel() == logging.WARNING
    finally:
        for logger, original_level in zip(loggers, original_levels, strict=True):
            logger.setLevel(original_level)
