"""Resolve directories for Acados JSON specs and generated C code."""

from __future__ import annotations

import os
from pathlib import Path

MHE_JSON_SUBDIR = "mhe_json"
CODEGEN_SUBDIR = "c_generated_code"


def _find_repo_root(start: Path) -> Path | None:
    """Walk parents from *start* looking for pyproject.toml (editable dev layout)."""
    for directory in (start, *start.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    return None


def get_data_root() -> Path:
    """Root directory for OpenMHE runtime artifacts.

    Resolution order:

    1. ``OPENMHE_DATA_DIR`` environment variable
    2. Repository root (if ``pyproject.toml`` is found above the installed package)
    3. ``<cwd>/.openmhe/``
    """
    env = os.environ.get("OPENMHE_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()

    package_dir = Path(__file__).resolve().parent
    repo = _find_repo_root(package_dir.parent)
    if repo is not None:
        return repo

    return (Path.cwd() / ".openmhe").resolve()


def get_mhe_json_dir() -> Path:
    """Directory for Acados OCP JSON problem descriptions."""
    path = get_data_root() / MHE_JSON_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_codegen_dir() -> Path:
    """Directory for Acados-generated C solver code."""
    path = get_data_root() / CODEGEN_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def mhe_json_path(filename: str) -> str:
    """Absolute path for a solver JSON file under :data:`get_mhe_json_dir`."""
    return str(get_mhe_json_dir() / filename)
