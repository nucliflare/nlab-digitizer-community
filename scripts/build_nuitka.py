"""Build a single-file executable using Nuitka.

Prerequisites (all in [dev] extras):
    pip install -e ".[dev]"

Run from the project root with the venv active:
    python scripts/build_nuitka.py

Output: dist/nlab.exe  (Windows)  /  dist/nlab  (Linux/macOS)

Nuitka notes
------------
- --onefile:           single self-contained binary; on first launch it
                       self-extracts to a temp dir then runs. Use
                       --standalone for a faster-launching folder dist.
- --enable-plugin=pyside6: required for PySide6 apps
- --windows-console-mode=disable: no console window on Windows (ignored on Linux)
- --include-module=PySide6.QtOpenGL(Widgets): pyqtgraph imports these for
                       optional GPU acceleration; Nuitka's static analyzer
                       doesn't discover them on its own.
- nlab.utils.windows_icon bundles resources/icons/ewt.ico as a real file
                       so its native LoadImage/WM_SETICON override works at
                       runtime inside the self-extraction temp dir.
- Compilation takes several minutes on first run; subsequent runs are faster
  (Nuitka caches compiled C objects between runs via ccache).

Alternative: PyInstaller
------------------------
    python scripts/build_pyinstaller.py
PyInstaller is faster to build and easier to debug hidden-import issues.
Nuitka produces smaller, faster-starting binaries and is used for releases.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRY = ROOT / "src" / "nlab" / "main.py"
ICON = ROOT / "resources" / "icons" / "ewt.ico"
RADIONUCLIDE_CATALOG = ROOT / "src" / "nlab" / "data" / "radionuclide_lines.sqlite3"
DIST = ROOT / "dist"

# On Linux, the executable has no .exe extension so it would be named "nlab"
# (a file), which conflicts with the "nlab/" directory Nuitka creates for the
# compiled package. Using a distinct name avoids that. Windows is fine because
# the exe gets .exe and "nlab/" can coexist with "nlab.exe".
OUTPUT_NAME = "nlab" if sys.platform == "win32" else "nlab-app"

args = [
    sys.executable, "-m", "nuitka",
    "--standalone",     # folder dist; onefile triggers Windows Smart App Control on unsigned builds
    "--enable-plugin=pyside6",
    f"--output-dir={DIST}",
    f"--output-filename={OUTPUT_NAME}",
    "--include-package=nlab",
    "--include-module=PySide6.QtOpenGL",
    "--include-module=PySide6.QtOpenGLWidgets",
    f"--include-data-files={ICON}=resources/icons/ewt.ico",
    f"--include-data-files={RADIONUCLIDE_CATALOG}=nlab/data/radionuclide_lines.sqlite3",
]

if sys.platform == "win32":
    args += [
        "--windows-console-mode=disable",
        f"--windows-icon-from-ico={ICON}",
    ]


def main() -> None:
    DIST.mkdir(exist_ok=True)
    print("Building with Nuitka — this may take a few minutes...")
    subprocess.run([*args, str(ENTRY)], check=True)
    print(f"\nBuild complete. Output in {DIST}/")


if __name__ == "__main__":
    main()
