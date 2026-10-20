from __future__ import annotations

import os
import runpy
import tomllib
from pathlib import Path

import pytest

from nlab.help.content import load_topics

ROOT = Path(__file__).resolve().parents[1]


def test_all_executable_builders_include_offline_help(tmp_path: Path) -> None:
    reviewed = runpy.run_path(str(ROOT / "scripts/build_pyinstaller_reviewed.py"))
    builder = reviewed["_build_arguments"]
    builder.__globals__["WORK"] = tmp_path
    fallback = runpy.run_path(str(ROOT / "scripts/build_pyinstaller.py"))
    nuitka = runpy.run_path(str(ROOT / "scripts/build_nuitka.py"))
    data = ROOT / "src/nlab/data/help"
    option = f"--add-data={data}{os.pathsep}nlab/data/help"
    assert option in builder(clean=False, version="0.7.4")
    assert option in fallback["args"]
    assert f"--include-data-dir={data}=nlab/data/help" in nuitka["args"]
    assert data.joinpath("parameters.md").is_file()


def test_wheel_data_and_cwd_independent_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    config = tomllib.loads(ROOT.joinpath("pyproject.toml").read_text(encoding="utf-8"))
    assert "data/help/*.md" in config["tool"]["setuptools"]["package-data"]["nlab"]
    monkeypatch.chdir(tmp_path)
    assert "scope-trigger" in load_topics()
