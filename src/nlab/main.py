import argparse
import logging
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from nlab import __version__
from nlab.app import MainAppWindow
from nlab.utils.settings_io import connection_settings, read_configuration
from nlab.utils.windows_icon import apply_taskbar_icon
from nlab.views.connection_dialog import ConnectionDialog

# Registers compiled Qt resources (icons, images) with the Qt resource system.
# Must happen before any QPixmap(":/...") call.
# `del _rc` removes only the local name; the module stays in sys.modules and
# the Qt registration remains active.
try:
    from nlab.ui import resources_rc as _rc

    del _rc
except ImportError:
    pass  # not compiled yet — run scripts/build_ui.py


log = logging.getLogger(__name__)


def _parse_arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Nuclear Lab Digitizer GUI")
    parser.add_argument("--config", type=Path, help="YAML settings file to apply after connecting")
    parser.add_argument("--ip", help="Initial device address in the connection dialog")
    parser.add_argument("--port", type=int, help="Initial device port")
    # Preserve Qt's own switches (for example -platform) for QApplication.
    return parser.parse_known_args(argv)


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_value(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args, qt_args = _parse_arguments(sys.argv[1:])
    _set_windows_app_user_model_id()
    app = QApplication([sys.argv[0], *qt_args])
    _icon = QIcon(":/icons/ewt.ico")
    log.info(
        "App icon from qrc resource: isNull=%s sizes=%s", _icon.isNull(), _icon.availableSizes()
    )
    app.setWindowIcon(_icon)
    app.setApplicationName("Nuclear Lab Digitizer")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("EWT")
    app.setOrganizationDomain("ewt.local")  # scopes QSettings on all platforms

    config: dict[str, object] = {}
    if args.config is not None:
        config = read_configuration(args.config)
    connection = connection_settings(config)
    config_ip = _string_value(connection.get("ip"))
    config_port = _int_value(connection.get("port"))
    config_backend = _string_value(connection.get("backend"))
    config_channels = _int_value(connection.get("channels"))

    splash = _show_splash()

    dialog = ConnectionDialog(
        ip=args.ip if args.ip is not None else config_ip,
        port=args.port if args.port is not None else config_port,
        backend=config_backend,
        channels=config_channels,
    )
    if splash is not None:
        splash.finish(dialog)  # splash closes as soon as the dialog is shown

    if dialog.exec() != ConnectionDialog.DialogCode.Accepted:
        sys.exit(0)

    log.info(
        "Application starting — v%s, backend=%s, host=%s, port=%d, channels=%d",
        __version__,
        dialog.backend,
        dialog.ip,
        dialog.port,
        dialog.channels,
    )
    window = MainAppWindow(
        backend=dialog.backend,
        host=dialog.ip,
        port=dialog.port,
        channels=dialog.channels,
        config_path=args.config,
    )

    window.show()
    # Apply the native taskbar icon after the event loop starts so Qt has
    # fully settled the QMainWindow's native HWND (dock layout, DWM
    # composition, etc.) before we target it with WM_SETICON. The in-__init__
    # call targets a provisional handle that may be replaced on first show().
    QTimer.singleShot(0, lambda: apply_taskbar_icon(window))
    sys.exit(app.exec())


def _set_windows_app_user_model_id() -> None:
    """Give this process its own taskbar identity on Windows.

    Without this, Windows groups the taskbar button under python.exe/
    pythonw.exe and shows *its* icon there instead of ours — even though
    setWindowIcon() already makes the titlebar icon correct, since that's
    purely a Qt-side concern. Must run before any window is shown.
    """
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "EWT.NuclearLabDigitizer.Community"
        )
    except (AttributeError, OSError):
        log.warning("Could not set Windows AppUserModelID — taskbar icon may be wrong")


def _show_splash() -> QSplashScreen | None:
    pixmap = QPixmap(":/ewt.png")
    if pixmap.isNull():
        return None
    splash = QSplashScreen(pixmap, Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    QApplication.processEvents()
    return splash


if __name__ == "__main__":
    main()
