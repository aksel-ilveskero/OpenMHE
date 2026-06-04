"""Load Acados shared libraries and configure Acados environment variables."""

import ctypes
import os
import sys

def _guess_acados_root() -> str:
    """Resolve Acados source tree without printing template warnings."""
    template_dir = os.path.dirname(
        os.path.abspath(sys.modules["acados_template"].__file__)
    )
    return os.path.realpath(os.path.join(template_dir, "..", "..", ".."))


def acados_root() -> str:
    """Resolved Acados installation root."""
    env = os.environ.get("ACADOS_SOURCE_DIR")
    if env:
        return os.path.realpath(os.path.expanduser(env))
    return _guess_acados_root()


def ensure_acados_environment() -> str:
    """Set ``ACADOS_SOURCE_DIR`` if missing and preload Acados shared libraries.

    Call once before building or loading an :class:`~acados_template.AcadosOcpSolver`
    to avoid repeated ``ACADOS_SOURCE_DIR`` warnings from ``acados_template``.
    """
    if not os.environ.get("ACADOS_SOURCE_DIR"):
        os.environ["ACADOS_SOURCE_DIR"] = acados_root()
    return ensure_acados_runtime_lib_path()


def ensure_acados_runtime_lib_path() -> str:
    """Preload Acados shared libraries so generated solvers can ``dlopen`` them.

    Updates ``LD_LIBRARY_PATH`` on Linux and returns the Acados ``lib`` directory.
    """
    lib_dir = os.path.join(acados_root(), "lib")
    if sys.platform == "win32":
        os.add_dll_directory(lib_dir)
        return lib_dir

    path_entries = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if lib_dir not in path_entries:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{lib_dir}:{os.environ['LD_LIBRARY_PATH']}"
            if os.environ.get("LD_LIBRARY_PATH")
            else lib_dir
        )

    for lib_name in ("libblasfeo", "libhpipm", "libqpOASES_e", "libacados"):
        lib_path = os.path.join(lib_dir, f"{lib_name}.so")
        if os.path.isfile(lib_path):
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)

    return lib_dir
