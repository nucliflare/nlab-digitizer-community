"""Readable plain-text tooltips without changing their meaning or F1 help."""

from __future__ import annotations

import textwrap

from PySide6.QtWidgets import QWidget

TOOLTIP_LINE_LENGTH = 68


def wrap_tooltip(text: str) -> str:
    """Wrap prose and long paths, preserving explicit line/paragraph breaks.

    Keep plain text rather than HTML so F1 explanations do not expose markup.
    Character-based wrapping follows the user's font scaling without forcing a
    fixed-pixel tooltip size. Short hints and empty tooltips remain unchanged.
    """
    return "\n".join(
        textwrap.fill(line, width=TOOLTIP_LINE_LENGTH, break_on_hyphens=False,
                      replace_whitespace=False)
        for line in text.split("\n")
    )


def wrap_widget_tooltips(root: QWidget) -> None:
    """Apply at the end of UI construction; safe for already-wrapped children."""
    for widget in (root, *root.findChildren(QWidget)):
        if widget.toolTip():
            widget.setToolTip(wrap_tooltip(widget.toolTip()))
