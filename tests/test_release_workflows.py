"""Guard release workflows against obsolete steps and packaging regressions."""

from __future__ import annotations

import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml


@pytest.mark.parametrize(
    "workflow_path",
    [".gitea/workflows/build.yaml", ".github/workflows/build.yaml"],
)
def test_workflow_does_not_generate_removed_grpc_stubs(workflow_path: str) -> None:
    root = Path(__file__).resolve().parents[1]
    workflow_text = (root / workflow_path).read_text(encoding="utf-8")

    assert "Generate gRPC stubs" not in workflow_text
    assert "generate_proto.py" not in workflow_text


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


def test_github_windows_installs_pinned_libiio_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load(
        (root / ".github/workflows/build.yaml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["build"]["steps"]

    install = next(
        step
        for step in steps
        if step.get("name") == "Install native libiio runtime (Windows only)"
    )
    assert install["if"] == "matrix.name == 'windows'"
    assert install["shell"] == "pwsh"
    assert "releases/download/v0.26/Windows.zip" in install["run"]
    assert "4AD4A8C6B3F7145922C122DCFC51D693F0D3F9A0054FB2228814CE58C538A6CA" in install["run"]
    assert "Windows-VS-2022-x64" in install["run"]
    assert "$env:GITHUB_PATH" in install["run"]

    verify = next(step for step in steps if step.get("name") == "Verify native libiio runtime")
    assert "if" not in verify
    assert "import iio" in verify["run"]


def test_github_release_collects_lmde_artifact_from_same_run() -> None:
    """A separate build must finish before its archive is collected for release."""
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load(
        (root / ".github/workflows/build.yaml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    lmde_job = jobs["build-lmde7"]
    lmde_workflow = yaml.safe_load(
        (root / lmde_job["uses"].removeprefix("./")).read_text(encoding="utf-8")
    )
    lmde_steps = lmde_workflow["jobs"]["build"]["steps"]
    upload = next(step for step in lmde_steps if "upload-artifact" in step.get("uses", ""))
    archive_path = Path(upload["with"]["path"])
    assert archive_path.name == "nlab-lmde7-x86_64.tar.gz"

    release = jobs["release"]
    assert {"build", "build-lmde7"} <= set(release["needs"])
    download = next(
        step for step in release["steps"] if "download-artifact" in step.get("uses", "")
    )
    assert "run-id" not in download["with"]
    assert "name" not in download["with"]
    downloaded_archive = (
        Path(download["with"]["path"]) / upload["with"]["name"] / archive_path.name
    )
    publish = next(
        step for step in release["steps"] if "action-gh-release" in step.get("uses", "")
    )
    assert any(
        fnmatch(downloaded_archive.as_posix(), pattern)
        for pattern in publish["with"]["files"].splitlines()
    )
