from __future__ import annotations

import runpy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from pytestqt.qtbot import QtBot

from nlab import main as main_module
from nlab.utils import root_conversion_check

_ROOT = Path(__file__).resolve().parents[1]


def test_both_pyinstaller_builders_use_project_hooks(tmp_path: Path) -> None:
    reviewed = runpy.run_path(str(_ROOT / "scripts/build_pyinstaller_reviewed.py"))
    builder = reviewed["_build_arguments"]
    builder.__globals__["WORK"] = tmp_path
    fallback = runpy.run_path(str(_ROOT / "scripts/build_pyinstaller.py"))
    hook_option = f"--additional-hooks-dir={_ROOT / 'scripts' / 'pyinstaller_hooks'}"
    assert hook_option in builder(clean=False, version="0.7.3")
    assert hook_option in fallback["args"]


def test_awkward_hook_collects_package_relative_native_kernels() -> None:
    hook = runpy.run_path(str(_ROOT / "scripts/pyinstaller_hooks/hook-awkward_cpp.py"))
    kernels = [(Path(src), Path(dst)) for src, dst in hook["binaries"] if "cpu-kernels" in src]
    assert kernels, "The ctypes-loaded CPU kernel library must be bundled"
    assert all(src.is_file() and dst.parts[-2:] == ("awkward_cpp", "lib") for src, dst in kernels)
    assert any("awkward" in Path(src).name and "cpu-kernels" not in src
               for src, _ in hook["binaries"])


def test_uproot_hook_collects_dynamically_loaded_tree_behaviors() -> None:
    hook = runpy.run_path(str(_ROOT / "scripts/pyinstaller_hooks/hook-uproot.py"))
    assert {"uproot.behaviors.TTree", "uproot.behaviors.TBranch"}.issubset(hook["hiddenimports"])


def test_offline_check_exercises_real_dialogs_and_both_root_trees(qtbot: QtBot) -> None:
    root_conversion_check.check_root_conversion()


@pytest.mark.parametrize("failure", [False, True])
def test_cli_check_exits_without_hardware_or_runtime_ui(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, failure: bool,
) -> None:
    application = Mock()
    monkeypatch.setattr(main_module, "QApplication", Mock(return_value=application))
    monkeypatch.setattr(main_module, "_set_windows_app_user_model_id", Mock())
    monkeypatch.setattr(sys, "argv", ["nlab", "--check-root-conversion", "-platform", "offscreen"])
    hardware = Mock(side_effect=AssertionError("must not probe hardware runtime"))
    runtime_ui = Mock(side_effect=AssertionError("must not open connection UI"))
    monkeypatch.setattr(main_module, "_verify_libiio_runtime", hardware)
    monkeypatch.setattr(main_module, "_load_runtime_ui", runtime_ui)
    check = Mock(side_effect=RuntimeError("missing DLL") if failure else None)
    monkeypatch.setattr(root_conversion_check, "check_root_conversion", check)
    with pytest.raises(SystemExit) as exit_info:
        main_module.main()
    assert exit_info.value.code == (1 if failure else 0)
    check.assert_called_once_with()
    hardware.assert_not_called()
    runtime_ui.assert_not_called()
