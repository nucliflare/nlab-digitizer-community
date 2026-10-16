"""Build a standalone executable using PyInstaller.

Prerequisites (all in [dev] extras):
    pip install -e ".[dev]"

Run from the project root with the venv active:
    python scripts/build_pyinstaller.py

Output: dist/nlab.exe  (Windows)  /  dist/nlab  (Linux/macOS)

Flags used
----------
--onefile       single executable (slower startup than --onedir)
--windowed      no console window on Windows/macOS
--icon          application icon
--name          output binary name

PyInstaller vs Nuitka
---------------------
PyInstaller bundles the Python interpreter + bytecode — faster to build,
easier to debug hidden-import issues, but larger binary and easier to
reverse-engineer.  Use Nuitka (scripts/build_nuitka.py) for release builds.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRY = ROOT / "src" / "nlab" / "main.py"
ICON = ROOT / "resources" / "icons" / "ewt.ico"
RADIONUCLIDE_CATALOG = ROOT / "src" / "nlab" / "data" / "radionuclide_lines.sqlite3"
N42_DATA = ROOT / "src" / "nlab" / "data"
DIST = ROOT / "dist"
WORK = ROOT / "build" / "pyinstaller"

args = [
    sys.executable, "-m", "PyInstaller",
    "--onefile",
    "--windowed",
    "--name", "nlab",
    f"--icon={ICON}",
    f"--distpath={DIST}",
    f"--workpath={WORK}",
    f"--specpath={ROOT / 'build' / 'pyinstaller'}",
    f"--additional-hooks-dir={ROOT / 'scripts' / 'pyinstaller_hooks'}",
    # nlab.utils.windows_icon's native taskbar-icon override needs a real
    # on-disk .ico at runtime (relative to the onefile extraction dir) — see
    # the matching note in scripts/build_nuitka.py.
    f"--add-data={ICON}{os.pathsep}resources/icons",
    f"--add-data={RADIONUCLIDE_CATALOG}{os.pathsep}nlab/data",
    f"--add-data={N42_DATA / 'n42.xsd'}{os.pathsep}nlab/data",
    f"--add-data={N42_DATA / 'N42SchemaAddin-2020.xsd'}{os.pathsep}nlab/data",
    f"--add-data={N42_DATA / 'n42.sch'}{os.pathsep}nlab/data",
    str(ENTRY),
]


def main() -> None:
    DIST.mkdir(exist_ok=True)
    print("Building with PyInstaller...")
    print("Command:", " ".join(str(a) for a in args))
    subprocess.run(args, check=True)
    print(f"\nBuild complete. Output in {DIST}/")


if __name__ == "__main__":
    main()
