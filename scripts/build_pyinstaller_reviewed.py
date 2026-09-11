"""Build a standalone executable using PyInstaller.

This is the reviewed successor to ``build_pyinstaller.py``.  The original
script is intentionally retained as a simple fallback while this version is
validated in normal development and release workflows.

Prerequisites::

    pip install -e ".[dev]"

Usage::

    python scripts/build_pyinstaller_reviewed.py
    python scripts/build_pyinstaller_reviewed.py --clean

Output: ``dist/nlab.exe`` on Windows, or ``dist/nlab`` elsewhere.

The build intentionally retains the runtime SciPy, Matplotlib, and Pillow
stacks.  Only development and interactive-computing packages that can leak in
from the ``dev`` extra are excluded.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRY = ROOT / "src" / "nlab" / "main.py"
ICON = ROOT / "resources" / "icons" / "ewt.ico"
VERSION_SOURCE = ROOT / "src" / "nlab" / "_version.py"
GENERATED_PROTO_DIR = ROOT / "src" / "nlab" / "hardware" / "grpc" / "generated"
DIST = ROOT / "dist"
WORK = ROOT / "build" / "pyinstaller-reviewed"
SPEC = WORK / "spec"
OUTPUT_NAME = "nlab"

REQUIRED_PROTO_FILES = (
    "__init__.py",
    "base_pb2.py",
    "base_pb2_grpc.py",
    "IDS_pb2.py",
    "IDS_pb2_grpc.py",
    "settings_pb2.py",
    "settings_pb2_grpc.py",
)

# These packages are installed by the development extra, or are optional
# interactive integrations discovered by dependency hooks.  The application
# neither imports nor invokes them at runtime.  SciPy, Matplotlib, and Pillow
# are deliberately absent from this list.
EXCLUDED_MODULES = (
    "pytest",
    "_pytest",
    "IPython",
    "ipykernel",
    "ipywidgets",
    "jupyter",
    "jupyter_client",
    "jupyter_core",
    "nbclient",
    "nbconvert",
    "nbformat",
    "notebook",
    "jedi",
    "grpc_tools",
    "setuptools",
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build nlab with PyInstaller")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="clear PyInstaller's cached analysis before building",
    )
    return parser.parse_args()


def _read_version() -> str:
    """Read ``__version__`` without importing the application package."""
    tree = ast.parse(VERSION_SOURCE.read_text(encoding="utf-8"), filename=str(VERSION_SOURCE))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in targets
        ):
            continue
        if node.value is None:
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, str) and value:
            return value
    raise RuntimeError(f"Could not find a string __version__ assignment in {VERSION_SOURCE}")


def _numeric_version(version: str) -> tuple[int, int, int, int]:
    """Convert a PEP 440-style leading release segment to a Windows tuple."""
    release = version.split("+", maxsplit=1)[0].split("-", maxsplit=1)[0]
    components: list[int] = []
    for component in release.split("."):
        digits = ""
        for character in component:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        components.append(int(digits))
        if len(components) == 4:
            break
    if not components:
        raise RuntimeError(f"Version {version!r} has no numeric release segment")
    padded = (components + [0] * 4)[:4]
    return padded[0], padded[1], padded[2], padded[3]


def _write_windows_version_file(version: str) -> Path:
    """Create the version resource consumed by PyInstaller on Windows."""
    version_file = WORK / "nlab-version-info.txt"
    version_file.parent.mkdir(parents=True, exist_ok=True)
    numeric = _numeric_version(version)
    version_file.write_text(
        f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={numeric!r},
    prodvers={numeric!r},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0),
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '040904B0',
        [
          StringStruct('CompanyName', 'EWT'),
          StringStruct('FileDescription', 'Nuclear Lab Digitizer'),
          StringStruct('FileVersion', {version!r}),
          StringStruct('InternalName', 'nlab'),
          StringStruct('LegalCopyright', 'Eastern Wall Technologies, Sp. z o. o.'),
          StringStruct('OriginalFilename', 'nlab.exe'),
          StringStruct('ProductName', 'Nuclear Lab Digitizer'),
          StringStruct('ProductVersion', {version!r}),
        ],
      )
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])]),
  ],
)
""",
        encoding="utf-8",
    )
    return version_file


def _preflight() -> None:
    required_paths = [
        ENTRY,
        ICON,
        VERSION_SOURCE,
        *(GENERATED_PROTO_DIR / name for name in REQUIRED_PROTO_FILES),
    ]
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(f"Required build inputs are missing:\n{formatted}")
    if importlib.util.find_spec("PyInstaller") is None:
        raise RuntimeError(
            f"PyInstaller is not installed for {sys.executable}. "
            'Install the development dependencies with: pip install -e ".[dev]"'
        )


def _build_arguments(*, clean: bool, version: str) -> list[str]:
    arguments = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--name",
        OUTPUT_NAME,
        f"--icon={ICON}",
        f"--distpath={DIST}",
        f"--workpath={WORK}",
        f"--specpath={SPEC}",
        f"--paths={GENERATED_PROTO_DIR}",
        f"--add-data={ICON}{os.pathsep}resources/icons",
    ]
    if clean:
        arguments.append("--clean")
    if sys.platform == "win32":
        arguments.append(f"--version-file={_write_windows_version_file(version)}")
    for module in EXCLUDED_MODULES:
        arguments.extend(("--exclude-module", module))
    arguments.append(str(ENTRY))
    return arguments


def _display_command(arguments: list[str]) -> str:
    if sys.platform == "win32":
        return subprocess.list2cmdline(arguments)
    return shlex.join(arguments)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    options = _parse_arguments()
    _preflight()

    DIST.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    SPEC.mkdir(parents=True, exist_ok=True)

    version = _read_version()
    arguments = _build_arguments(clean=options.clean, version=version)
    print(f"Building nlab {version} with PyInstaller...", flush=True)
    print(f"Command: {_display_command(arguments)}", flush=True)

    started = time.monotonic()
    subprocess.run(arguments, check=True)
    elapsed = time.monotonic() - started

    executable = DIST / (f"{OUTPUT_NAME}.exe" if sys.platform == "win32" else OUTPUT_NAME)
    if not executable.is_file():
        raise RuntimeError(f"PyInstaller completed but did not create {executable}")

    print("\nBuild complete.")
    print(f"Output:  {executable}")
    print(f"Size:    {executable.stat().st_size / (1024 * 1024):.2f} MiB")
    print(f"SHA-256: {_sha256(executable)}")
    print(f"Elapsed: {elapsed:.1f} seconds")

    warnings_file = WORK / OUTPUT_NAME / f"warn-{OUTPUT_NAME}.txt"
    if warnings_file.is_file() and warnings_file.stat().st_size:
        print(f"Warnings: {warnings_file}")


if __name__ == "__main__":
    main()
