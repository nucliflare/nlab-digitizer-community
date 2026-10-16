"""Collect ROOT behaviors selected by name at runtime, not by static imports."""

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("uproot.behaviors")
