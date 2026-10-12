from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QEventLoop, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen

from nlab import __version__
from nlab.utils.windows_icon import apply_taskbar_icon

if TYPE_CHECKING:
    from nlab.app import MainAppWindow as _MainAppWindow
    from nlab.views.connection_dialog import ConnectionDialog as _ConnectionDialog

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
LIBIIO_V026_RELEASE_URL = "https://github.com/analogdevicesinc/libiio/releases/tag/v0.26"
LIBIIO_V026_WINDOWS_INSTALLER = "libiio-0.26.ga0eca0d2-setup.exe"

# Loaded only after QApplication exists and the native libiio runtime has been
# checked. Keeping these imports out of module scope lets frozen builds replace
# PyInstaller's raw missing-DLL traceback with an actionable Qt dialog.
MainAppWindow: type[_MainAppWindow] | None = None
ConnectionDialog: type[_ConnectionDialog] | None = None


class LibiioRuntimeError(RuntimeError):
    """The pylibiio wrapper could not load a usable native runtime."""


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Debug mode is intended for NLab diagnostics. Report rendering otherwise
    # exposes verbose Matplotlib internals and Pillow PNG chunk parsing through
    # the root logger when developer debug mode is enabled.
    for logger_name in ("matplotlib", "PIL"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _verify_libiio_runtime() -> None:
    try:
        module = importlib.import_module("iio")
    except Exception as exc:
        raise LibiioRuntimeError(str(exc) or type(exc).__name__) from exc
    version = getattr(module, "version", None)
    if (
        isinstance(version, tuple)
        and version
        and isinstance(version[0], int)
        and version[0] >= 1
    ):
        display_version = ".".join(str(component) for component in version[:2])
        raise LibiioRuntimeError(
            f"libiio {display_version} uses the incompatible 1.x API; libiio v0.26 is required"
        )
    log.info("Loaded native libiio runtime: %s", version if version is not None else "unknown")


def _load_runtime_ui() -> None:
    global ConnectionDialog, MainAppWindow
    if MainAppWindow is None:
        from nlab.app import MainAppWindow as RuntimeMainAppWindow

        MainAppWindow = RuntimeMainAppWindow
    if ConnectionDialog is None:
        from nlab.views.connection_dialog import ConnectionDialog as RuntimeConnectionDialog

        ConnectionDialog = RuntimeConnectionDialog


def _show_libiio_runtime_error(error: LibiioRuntimeError) -> None:
    box = QMessageBox()
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle("libiio Runtime Required")
    box.setText("A supported Analog Devices libiio runtime is not available.")
    if sys.platform == "win32":
        instructions = (
            f"Install the supported libiio v0.26 Windows package "
            f"({LIBIIO_V026_WINDOWS_INSTALLER}), then restart Nuclear Lab Digitizer."
        )
    else:
        instructions = (
            "Install the libiio v0.26 native runtime using your system package manager or "
            "the official release assets, then restart Nuclear Lab Digitizer."
        )
    box.setInformativeText(
        f"{instructions}\n\nlibiio 1.x uses a different API and is not yet a validated native "
        "runtime for this application."
    )
    box.setDetailedText(
        f"Official release: {LIBIIO_V026_RELEASE_URL}\n\nStartup error:\n{error}"
    )
    box.setStandardButtons(QMessageBox.StandardButton.Close)
    box.setDefaultButton(QMessageBox.StandardButton.Close)
    open_button = box.addButton(
        "Open libiio v0.26 download page",
        QMessageBox.ButtonRole.ActionRole,
    )
    box.exec()
    if box.clickedButton() is open_button:
        if not QDesktopServices.openUrl(QUrl(LIBIIO_V026_RELEASE_URL)):
            log.warning("Could not open the libiio release page in the default browser")


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
    _configure_logging()
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

    try:
        _verify_libiio_runtime()
    except LibiioRuntimeError as exc:
        log.error("Native libiio runtime is unavailable: %s", exc)
        _show_libiio_runtime_error(exc)
        sys.exit(1)

    _load_runtime_ui()
    assert MainAppWindow is not None
    assert ConnectionDialog is not None
    from nlab.utils.settings_io import (
        connection_settings,
        read_configuration,
        validate_connection_transport,
    )

    config: dict[str, object] = {}
    if args.config is not None:
        config = read_configuration(args.config)
        validate_connection_transport(config)
    connection = connection_settings(config)
    config_ip = _string_value(connection.get("ip"))
    config_port = _int_value(connection.get("port"))
    config_channels = _int_value(connection.get("channels"))

    splash = _show_splash()

    dialog = ConnectionDialog(
        ip=args.ip if args.ip is not None else config_ip,
        port=args.port if args.port is not None else config_port,
        channels=config_channels,
    )
    splash.finish(dialog)  # splash closes as soon as the dialog is shown

    if dialog.exec() != ConnectionDialog.DialogCode.Accepted:
        sys.exit(0)

    offline = bool(getattr(dialog, "offline", False))
    if offline:
        log.info(
            "Application starting — v%s, offline analysis mode, channels=%d",
            __version__,
            dialog.channels,
        )
    else:
        log.info(
            "Application starting — v%s, IIO host=%s, port=%d, channels=%d",
            __version__,
            dialog.ip,
            dialog.port,
            dialog.channels,
        )
    connection_splash = _show_connection_splash()
    if offline:
        _update_splash(connection_splash, "Preparing offline analysis workspace...")
    try:
        window = MainAppWindow(
            host="" if offline else dialog.ip,
            port=dialog.port,
            channels=dialog.channels,
            config_path=args.config,
            on_progress=lambda message: _update_splash(connection_splash, message),
            offline=offline,
        )
        window.show()
        QApplication.processEvents()
    except Exception as exc:
        connection_splash.close()
        log.exception("Device connection or setup failed")
        QMessageBox.critical(None, "Connection Failed", str(exc))
        sys.exit(1)
    connection_splash.finish(window)
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


def _show_splash() -> QSplashScreen:
    return _create_splash("Launching Application", "Launching application")


def _show_connection_splash() -> QSplashScreen:
    """Show visible progress before synchronous device setup begins."""
    return _create_splash("Connecting to Device", "Connecting to device...")


def _create_splash(title: str, message: str) -> QSplashScreen:
    pixmap = QPixmap(520, 320)
    pixmap.fill(QColor("#20252b"))
    logo = QPixmap(":/ewt.png")
    painter = QPainter(pixmap)
    try:
        if logo.isNull():
            painter.setPen(QColor("#e9f0f6"))
            painter.setFont(QFont("Sans Serif", 22, QFont.Weight.Bold))
            painter.drawText(
                0,
                35,
                pixmap.width(),
                210,
                Qt.AlignmentFlag.AlignCenter,
                "Nuclear Lab Digitizer",
            )
        else:
            scaled = logo.scaled(
                220,
                220,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap((pixmap.width() - scaled.width()) // 2, 24, scaled)
    finally:
        painter.end()
    splash = QSplashScreen(pixmap, Qt.WindowType.WindowStaysOnTopHint)
    splash.setWindowTitle(title)
    splash.show()
    _update_splash(splash, message)
    return splash


def _update_splash(splash: QSplashScreen, message: str) -> None:
    splash.showMessage(
        message,
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
        QColor("#e9f0f6"),
    )
    QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)


if __name__ == "__main__":
    main()
