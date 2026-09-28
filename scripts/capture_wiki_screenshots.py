#!/usr/bin/env python
"""Capture hardware-independent wiki screenshots without measurement plots."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QWidget

from nlab.views.connection_dialog import ConnectionDialog


def _save_widget(app: QApplication, widget: QWidget, path: Path) -> None:
    widget.resize(560, 300)
    widget.show()
    app.processEvents()
    image = widget.grab()
    if not image.save(str(path), "PNG"):
        raise RuntimeError(f"could not save {path}")
    widget.close()
    app.processEvents()


def _load_documentation_font(app: QApplication) -> None:
    """Load one explicit font when an offscreen platform finds no system fonts."""
    candidates = (
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(str(candidate))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0], 9))
            return
    if not QFontDatabase.families():
        raise RuntimeError("Qt could not discover a usable font for screenshots")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "images",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nlab-wiki-screenshots-") as settings_dir:
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(
            QSettings.Format.IniFormat,
            QSettings.Scope.UserScope,
            settings_dir,
        )
        app = QApplication.instance() or QApplication([])
        app.setStyle("Fusion")
        _load_documentation_font(app)
        dialog = ConnectionDialog(
            ip="192.168.10.128",
            port=30_431,
            backend="iio",
            channels=2,
        )
        _save_widget(app, dialog, args.output_dir / "connection-dialog.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
