"""Load the same Markdown parameter reference included by the project wiki."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files

_HEADING = re.compile(r"^## (.+) \{#([a-z0-9-]+)\}$", re.MULTILINE)


@dataclass(frozen=True)
class HelpTopic:
    key: str
    title: str
    markdown: str


def load_topics() -> dict[str, HelpTopic]:
    """Read package data, including inside a wheel or frozen application."""
    reference = files("nlab").joinpath("data", "help", "parameters.md")
    return parse_topics(reference.read_text(encoding="utf-8"))


def parse_topics(markdown: str) -> dict[str, HelpTopic]:
    headings = list(_HEADING.finditer(markdown))
    topics: dict[str, HelpTopic] = {}
    for index, match in enumerate(headings):
        title, key = match.groups()
        if key in topics:
            raise ValueError(f"Duplicate help topic: {key}")
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        topics[key] = HelpTopic(key, title, markdown[match.end():end].strip())
    if "help" not in topics:
        raise ValueError("Help reference has no home topic")
    return topics
