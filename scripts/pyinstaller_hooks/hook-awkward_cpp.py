"""Include Awkward's ctypes-loaded kernels at their package-relative paths."""

from PyInstaller.utils.hooks import collect_delvewheel_libs_directory, collect_dynamic_libs

# cpu_kernels.py/libawkward.py construct these paths using importlib.resources,
# so PyInstaller's literal ctypes-call analysis cannot reliably discover them.
binaries = collect_dynamic_libs("awkward_cpp")
datas, binaries = collect_delvewheel_libs_directory("awkward_cpp", binaries=binaries)
