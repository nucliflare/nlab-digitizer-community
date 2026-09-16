"""Keep downloadable Windows artifacts to one archive layer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml


@pytest.mark.parametrize(
    "workflow_path",
    [".gitea/workflows/build.yaml", ".github/workflows/build.yaml"],
)
def test_windows_artifact_uploads_executable_not_nested_zip(workflow_path: str) -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / workflow_path).read_text(encoding="utf-8"))
    build = workflow["jobs"]["build"]
    windows = next(
        item for item in build["strategy"]["matrix"]["include"] if item["name"] == "windows"
    )
    assert windows["package"] == "dist/nlab.exe"

    upload = next(step for step in build["steps"] if "upload-artifact" in step.get("uses", ""))
    assert upload["with"]["path"] == "${{ matrix.package }}"
    assert not any(step.get("name") == "Package build (Windows)" for step in build["steps"])

    release_steps = workflow["jobs"]["release"]["steps"]
    package_index = next(
        index
        for index, step in enumerate(release_steps)
        if step.get("name") == "Package Windows release archive"
    )
    release_index = next(
        index
        for index, step in enumerate(release_steps)
        if step.get("name", "").startswith("Create ")
    )
    assert package_index < release_index
    assert release_steps[package_index]["run"].splitlines() == [
        "cd dist/nlab-windows",
        "python3 -m zipfile -c nlab-windows.zip nlab.exe",
    ]


def test_release_zip_recipe_contains_executable_at_archive_root(tmp_path: Path) -> None:
    (tmp_path / "nlab.exe").write_bytes(b"test executable")
    subprocess.run(
        [sys.executable, "-m", "zipfile", "-c", "nlab-windows.zip", "nlab.exe"],
        cwd=tmp_path,
        check=True,
    )
    with ZipFile(tmp_path / "nlab-windows.zip") as archive:
        assert archive.namelist() == ["nlab.exe"]
