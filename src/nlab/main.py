import logging
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from nlab import __version__
from nlab.app import MainAppWindow
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _set_windows_app_user_model_id()
    app = QApplication(sys.argv)
    _icon = QIcon(":/icons/ewt.ico")
    log.info("App icon from qrc resource: isNull=%s sizes=%s",
              _icon.isNull(), _icon.availableSizes())
    app.setWindowIcon(_icon)
    app.setApplicationName("Nuclear Lab Digitizer")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("EWT")
    app.setOrganizationDomain("ewt.local")  # scopes QSettings on all platforms

    splash = _show_splash()

    dialog = ConnectionDialog()
    if splash is not None:
        splash.finish(dialog)  # splash closes as soon as the dialog is shown

    if dialog.exec() != ConnectionDialog.DialogCode.Accepted:
        sys.exit(0)

    log.info("Application starting — v%s, backend=%s, host=%s, port=%d, channels=%d",
             __version__, dialog.backend, dialog.ip, dialog.port, dialog.channels)
    window = MainAppWindow(
        backend=dialog.backend, host=dialog.ip, port=dialog.port, channels=dialog.channels,
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
            "EWT.NuclearLabDigitizer.Community")
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
