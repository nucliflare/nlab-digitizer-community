"""Offline display-math support for the Qt Markdown help browser."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from hashlib import sha256
from io import BytesIO

from PySide6.QtCore import QUrl
from PySide6.QtGui import QImage, QPalette, QTextDocument, QTextFormat, QTextImageFormat
from PySide6.QtWidgets import QTextBrowser

log = logging.getLogger(__name__)
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)$")


def _math_placeholders(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace display math outside code fences, leaving other Markdown intact."""
    lines = markdown.splitlines()
    output: list[str] = []
    formulas: list[tuple[str, str]] = []
    prefix = "NLABMATH" + sha256(markdown.encode()).hexdigest()[:16]
    fence = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        marker = _FENCE.fullmatch(line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                if not marker[2].strip():
                    fence = ""
        elif marker:
            fence = marker[1]
        else:
            stripped = line.strip()
            expression = ""
            end = index
            if line.startswith(("    ", "\t")):
                output.append(line)
                index += 1
                continue
            if stripped == "$$":
                end = next(
                    (i for i in range(index + 1, len(lines)) if lines[i].strip() == "$$"),
                    index,
                )
                expression = " ".join(lines[index + 1:end]).strip()
            elif stripped.startswith("$$") and stripped.endswith("$$"):
                expression = stripped[2:-2].strip()
            if expression:
                token = f"{prefix}X{len(formulas)}END"
                formulas.append((token, expression))
                output.extend(("", token, ""))
                index = end + 1
                continue
        output.append(line)
        index += 1
    return "\n".join(output), formulas


@lru_cache(maxsize=128)
def _formula_image(expression: str, color: str, font_size: float, scale: float) -> QImage:
    # Import only when a math topic is opened. Explicitly disabling usetex
    # keeps rendering offline even if another Matplotlib user enables TeX.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(1, 0.5), dpi=96 * scale)
    FigureCanvasAgg(figure)
    figure.text(
        0, 0, f"${expression}$", color=color, fontsize=font_size,
        fontfamily="DejaVu Sans", math_fontfamily="dejavusans", usetex=False,
    )
    stream = BytesIO()
    try:
        figure.savefig(
            stream, format="png", dpi=96 * scale, transparent=True,
            bbox_inches="tight", pad_inches=0.04,
        )
    finally:
        figure.clear()
    image = QImage.fromData(stream.getvalue())
    if image.isNull():
        raise RuntimeError("Could not render the help formula")
    return image


def set_help_markdown(browser: QTextBrowser, markdown: str) -> None:
    """Render Markdown plus $$ display math, retaining TeX as image alt text."""
    converted, formulas = _math_placeholders(markdown)
    browser.setMarkdown(converted)
    if not formulas:
        return
    document = browser.document()
    color = browser.palette().color(QPalette.ColorRole.Text).name()
    point_size = browser.font().pointSizeF()
    if point_size <= 0:
        point_size = max(browser.font().pixelSize(), 12) * 72 / 96
    font_size = max(12.0, point_size + 2)
    scale = max(2.0, browser.devicePixelRatioF())
    for token, expression in formulas:
        cursor = document.find(token)
        if cursor.isNull():
            continue
        try:
            image = _formula_image(expression, color, font_size, scale)
        except (OSError, RuntimeError, ValueError):
            log.warning("Could not render a help formula: %s", expression, exc_info=True)
            cursor.insertText(f"$$ {expression} $$")
            continue
        digest = sha256(f"{expression}|{color}|{font_size}|{scale}".encode()).hexdigest()
        url = QUrl(f"nlab-math:{digest}")
        document.addResource(QTextDocument.ResourceType.ImageResource, url, image)
        image_format = QTextImageFormat()
        image_format.setName(url.toString())
        image_format.setWidth(image.width() / scale)
        image_format.setHeight(image.height() / scale)
        image_format.setProperty(QTextFormat.Property.ImageAltText, expression)
        image_format.setToolTip(expression)
        cursor.insertImage(image_format)
