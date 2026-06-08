"""Run sliding-window MHE via the compiled C driver (``libopenmhe_mhe_run.so``)."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from openmhe.builder.input_regs import unmeasured_regulator_indices
from openmhe.builder.solver import WindowStep, _precompute_yrefs
from openmhe.frontend.acados_runtime import acados_root, ensure_acados_environment
from openmhe.paths import get_codegen_dir

_C_SOLVER_DIR = Path(__file__).resolve().parent.parent / "c_solver"
_RUN_LIB = _C_SOLVER_DIR / "libopenmhe_mhe_run.so"

_RUN_LIB_HANDLE: ctypes.CDLL | None = None
_OCP_LIB_HANDLE: ctypes.CDLL | None = None
_LOADED_HEADER_MTIME: float | None = None


class _RunConfig(ctypes.Structure):
    _fields_ = [
        ("n_steps", ctypes.c_int),
        ("N", ctypes.c_int),
        ("nx", ctypes.c_int),
        ("nx_base", ctypes.c_int),
        ("nu", ctypes.c_int),
        ("nu_ocp", ctypes.c_int),
        ("nu_ctrl", ctypes.c_int),
        ("ny_stage", ctypes.c_int),
        ("ny0", ctypes.c_int),
        ("has_arrival", ctypes.c_int),
        ("arrival_off", ctypes.c_int),
        ("dynamic_arrival", ctypes.c_int),
        ("n_pin", ctypes.c_int),
        ("n_u_extract", ctypes.c_int),
        ("n_rw", ctypes.c_int),
        ("n_fd", ctypes.c_int),
        ("n_sd", ctypes.c_int),
        ("n_unmeasured", ctypes.c_int),
    ]


def _build_run_lib(*, force: bool = False) -> Path:
    """Compile ``libopenmhe_mhe_run.so`` if missing or stale."""
    codegen = get_codegen_dir()
    header = codegen / "acados_solver_openmhe_mhe.h"
    if not header.is_file():
        raise FileNotFoundError(
            f"No generated solver at {header}. Call build_mhe_solver() first."
        )

    run_src = _C_SOLVER_DIR / "run_loop.c"
    filter_src = _C_SOLVER_DIR / "filter_arrival.c"
    run_obj = _C_SOLVER_DIR / "run_loop.o"
    stale_obj = run_obj.is_file() and run_obj.stat().st_mtime < header.stat().st_mtime
    if (
        not force
        and not stale_obj
        and _RUN_LIB.is_file()
        and _RUN_LIB.stat().st_mtime >= header.stat().st_mtime
        and _RUN_LIB.stat().st_mtime >= run_src.stat().st_mtime
        and _RUN_LIB.stat().st_mtime >= filter_src.stat().st_mtime
    ):
        return _RUN_LIB

    if force or stale_obj:
        run_obj.unlink(missing_ok=True)
        _RUN_LIB.unlink(missing_ok=True)

    ensure_acados_environment()
    env = os.environ.copy()
    env["CODEGEN_DIR"] = str(codegen)
    env["ACADOS_DIR"] = acados_root()
    subprocess.run(
        ["make", "-C", str(_C_SOLVER_DIR), f"CODEGEN_DIR={codegen}"],
        check=True,
        env=env,
    )
    return _RUN_LIB


def _unload_shared_lib(lib: ctypes.CDLL | None) -> None:
    """Drop a ctypes-loaded shared library so a rebuilt ``.so`` can be mapped."""
    if lib is None:
        return
    handle = getattr(lib, "_handle", None)
    if not handle:
        return
    if sys.platform == "linux":
        try:
            libdl = ctypes.CDLL("libdl.so.2")
        except OSError:
            libdl = ctypes.CDLL("libdl.so")
        libdl.dlclose.argtypes = [ctypes.c_void_p]
        libdl.dlclose.restype = ctypes.c_int
        libdl.dlclose(ctypes.c_void_p(handle))
    elif sys.platform == "darwin":
        libc = ctypes.CDLL("libc.dylib")
        libc.dlclose.argtypes = [ctypes.c_void_p]
        libc.dlclose.restype = ctypes.c_int
        libc.dlclose(ctypes.c_void_p(handle))


def _load_run_lib(*, rebuild: bool = False) -> ctypes.CDLL:
    """Load (or reload) the C sliding-window driver and Acados OCP library."""
    global _RUN_LIB_HANDLE, _OCP_LIB_HANDLE, _LOADED_HEADER_MTIME

    ensure_acados_environment()
    codegen = get_codegen_dir()
    header = codegen / "acados_solver_openmhe_mhe.h"
    header_mtime = header.stat().st_mtime if header.is_file() else 0.0

    lib_path = _build_run_lib(force=rebuild)

    need_reload = (
        rebuild
        or _RUN_LIB_HANDLE is None
        or _LOADED_HEADER_MTIME is None
        or header_mtime > _LOADED_HEADER_MTIME
    )
    if need_reload and _RUN_LIB_HANDLE is not None:
        _unload_shared_lib(_RUN_LIB_HANDLE)
        _unload_shared_lib(_OCP_LIB_HANDLE)
        _RUN_LIB_HANDLE = None
        _OCP_LIB_HANDLE = None
        _LOADED_HEADER_MTIME = None

    ocp_lib = codegen / "libacados_ocp_solver_openmhe_mhe.so"
    if _OCP_LIB_HANDLE is None and ocp_lib.is_file():
        _OCP_LIB_HANDLE = ctypes.CDLL(str(ocp_lib), mode=ctypes.RTLD_GLOBAL)

    if _RUN_LIB_HANDLE is None:
        _RUN_LIB_HANDLE = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        _LOADED_HEADER_MTIME = header_mtime

    return _RUN_LIB_HANDLE


def _u_extract_specs(solver) -> np.ndarray:
    """Packed triples ``(out_ui, from_u, src_idx)`` for the C extractor."""
    controlled_idx = list(solver._controlled_idx)
    controlled_map = solver._controlled_map
    specs: list[tuple[int, int, int]] = []
    for ui in controlled_idx:
        specs.append((ui, 1, controlled_map[ui]))
    for ui, col in solver._load_state_col.items():
        specs.append((ui, 0, col))
    for ui, col in solver._col_sd1.items():
        if ui not in controlled_map:
            specs.append((ui, 0, col))
    for ui, col in solver._col_fd1.items():
        if ui not in controlled_map and ui not in solver._load_state_col:
            specs.append((ui, 0, col))
    flat = np.array(specs, dtype=np.int32).reshape(-1)
    return flat


def _index_array(mapping: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if not mapping:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    keys = sorted(mapping.keys())
    return (
        np.asarray(keys, dtype=np.int32),
        np.asarray([mapping[k] for k in keys], dtype=np.int32),
    )


def _precompute_arrival(
    arrival_cost,
    y: np.ndarray,
    u: np.ndarray | None,
    n_est: int,
    nx_base: int,
    W0_template: np.ndarray,
    arrival_w_slice: slice,
) -> tuple[np.ndarray, np.ndarray]:
    """Python arrival pass (EKF / UKF dynamic weights for the C driver)."""
    x_bar_stage = np.zeros((n_est, nx_base), dtype=np.float64, order="C")
    ny0 = W0_template.shape[0]
    W0_stage = np.zeros((n_est, ny0, ny0), dtype=np.float64, order="F")
    for idx in range(n_est):
        x_bar, P = arrival_cost.window_prior(idx, y, u)
        x_bar_stage[idx, :] = x_bar
        W0 = W0_template.copy()
        W0[arrival_w_slice, arrival_w_slice] = np.linalg.inv(P)
        W0_stage[idx, :, :] = W0
    return x_bar_stage, W0_stage


def run_c_solver(solver, y, u, post_steps=None, *, rebuild: bool = False):
    """C implementation of :func:`~openmhe.run_solver` (same return values).

    ``post_steps`` are applied in Python after the C loop (same semantics as
    :func:`~openmhe.run_solver`).
    """
    from openmhe.mhe_strategies.arrival_cost import UKFArrivalCost

    arrival_cost = getattr(solver, "_arrival_cost", None)
    if isinstance(arrival_cost, UKFArrivalCost):
        raise TypeError(
            "run_c_solver does not support UKFArrivalCost yet; use EKFArrivalCost "
            "or run_solver() for UKF."
        )

    lib = _load_run_lib(rebuild=rebuild)
    lib.openmhe_mhe_acados_create_capsule.restype = ctypes.c_void_p
    lib.openmhe_mhe_acados_free_capsule.argtypes = [ctypes.c_void_p]
    _c_double_p = ctypes.POINTER(ctypes.c_double)
    _c_int_p = ctypes.POINTER(ctypes.c_int)
    lib.openmhe_mhe_run_sliding.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_RunConfig),
        _c_double_p,
        _c_double_p,
        _c_double_p,
        _c_double_p,
        _c_int_p,
        _c_int_p,
        _c_int_p,
        _c_int_p,
        _c_int_p,
        _c_int_p,
        _c_int_p,
        _c_int_p,
        _c_int_p,
        _c_int_p,
        _c_double_p,
        _c_double_p,
        _c_double_p,
        _c_double_p,
    ]
    lib.openmhe_mhe_run_sliding.restype = ctypes.c_int

    y = np.asarray(y, dtype=np.float64, order="C")
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n_steps = y.shape[1]
    nu = solver._nu
    nx_base = solver._nx_base
    N = solver._N
    n_est = n_steps - N
    if n_est <= 0:
        raise ValueError(
            f"Measurement sequence has {n_steps} samples but horizon is N={N}; "
            "need at least N+1 samples."
        )

    builder = solver._builder
    nw = solver._nw
    yrefs = _precompute_yrefs(builder, y, u, nx_base, nu, nw, n_steps)
    ny_stage = yrefs.shape[1]

    u_hat_rm = np.full((n_est, nu), np.nan, order="C")
    u_raw_rm = np.full((n_est, nu), np.nan, order="C")
    x_hat_rm = np.full((n_est, nx_base), np.nan, order="C")

    if arrival_cost is not None:
        arrival_cost.reset()

    has_arrival = solver._arrival_slice is not None

    pin_vals = None
    if solver._pin_u_idx and u is not None:
        known_inputs = np.asarray(solver._known_inputs, dtype=int)
        pin_vals = np.ascontiguousarray(u[known_inputs, :].T)

    rw_idx, rw_col = _index_array(solver._load_state_col)
    fd_idx, fd_col = _index_array(solver._col_fd1)
    sd_idx, sd1_col = _index_array(solver._col_sd1)
    _, sd2_col = _index_array(solver._col_sd2)
    unmeasured = sorted(
        unmeasured_regulator_indices(
            builder, solver._rw_indices, solver._fd_indices, solver._sd_indices
        )
    )
    unmeasured_ui = np.asarray(unmeasured, dtype=np.int32)

    x_bar_stage = None
    W0_stage = None
    if (
        has_arrival
        and arrival_cost is not None
        and arrival_cost.is_dynamic
        and solver._W0_template is not None
    ):
        W0_template = np.asfortranarray(solver._W0_template, dtype=np.float64)
        x_bar_stage, W0_stage = _precompute_arrival(
            arrival_cost,
            y,
            u,
            n_est,
            nx_base,
            W0_template,
            solver._arrival_W_slice,
        )

    nu_ocp = solver._nu_ctrl + solver._nw
    cfg = _RunConfig(
        n_steps=n_steps,
        N=N,
        nx=solver._nx,
        nx_base=nx_base,
        nu=nu,
        nu_ocp=nu_ocp,
        nu_ctrl=solver._nu_ctrl,
        ny_stage=ny_stage,
        ny0=solver._n_residual_0 if has_arrival else ny_stage,
        has_arrival=int(has_arrival),
        arrival_off=int(solver._arrival_slice.start) if has_arrival else 0,
        dynamic_arrival=int(W0_stage is not None),
        n_pin=len(solver._pin_u_idx) if pin_vals is not None else 0,
        n_u_extract=len(_u_extract_specs(solver)) // 3,
        n_rw=len(rw_idx),
        n_fd=len(fd_idx),
        n_sd=len(sd_idx),
        n_unmeasured=len(unmeasured_ui),
    )

    u_meas = None
    if u is not None:
        u_meas = np.ascontiguousarray(u.T if u.ndim > 1 else u.reshape(1, -1).T)

    def _f64(arr):
        if arr is None:
            return None
        return np.ascontiguousarray(arr, dtype=np.float64)

    def _i32(arr):
        return np.ascontiguousarray(arr, dtype=np.int32)

    capsule = lib.openmhe_mhe_acados_create_capsule()
    try:
        status = lib.openmhe_mhe_run_sliding(
            capsule,
            ctypes.byref(cfg),
            _f64(yrefs).ctypes.data_as(_c_double_p),
            _f64(x_bar_stage).ctypes.data_as(_c_double_p)
            if x_bar_stage is not None
            else None,
            _f64(W0_stage).ctypes.data_as(_c_double_p)
            if W0_stage is not None
            else None,
            _f64(pin_vals).ctypes.data_as(_c_double_p) if pin_vals is not None else None,
            _i32(solver._controlled_idx).ctypes.data_as(_c_int_p),
            _u_extract_specs(solver).ctypes.data_as(_c_int_p),
            rw_idx.ctypes.data_as(_c_int_p),
            rw_col.ctypes.data_as(_c_int_p),
            fd_idx.ctypes.data_as(_c_int_p),
            fd_col.ctypes.data_as(_c_int_p),
            sd_idx.ctypes.data_as(_c_int_p),
            sd1_col.ctypes.data_as(_c_int_p),
            sd2_col.ctypes.data_as(_c_int_p),
            unmeasured_ui.ctypes.data_as(_c_int_p),
            u_meas.ctypes.data_as(_c_double_p) if u_meas is not None else None,
            _f64(u_hat_rm).ctypes.data_as(_c_double_p),
            _f64(x_hat_rm).ctypes.data_as(_c_double_p),
            _f64(u_raw_rm).ctypes.data_as(_c_double_p),
        )
    finally:
        lib.openmhe_mhe_acados_free_capsule(capsule)

    if status == -3:
        raise RuntimeError(
            "C MHE driver dimensions do not match the current Acados codegen "
            f"(solver N={N}, nx={solver._nx}, nu_ocp={nu_ocp}, ny0={cfg.ny0}). "
            "After changing horizon or objective terms, call "
            "run_c_solver(..., rebuild=True). If the error persists in Jupyter, "
            "restart the kernel (ctypes cannot reload a stale in-memory .so)."
        )
    if status != 0:
        raise RuntimeError(f"C MHE driver failed with status {status}")

    u_hat = u_hat_rm.T
    x_hat = x_hat_rm.T

    _post_steps = list(post_steps) if post_steps is not None else []
    for step in _post_steps:
        if callable(getattr(step, "reset", None)):
            step.reset()

    if _post_steps:
        for idx, k in enumerate(range(N, n_steps)):
            if np.any(np.isnan(u_hat[:, idx])):
                continue
            ctx = WindowStep(
                idx=idx,
                t_start=k - N,
                k=k,
                N=N,
                dt=solver._dt,
                nx_base=nx_base,
                nu=nu,
                x_hat=x_hat,
                u_hat=u_hat,
                x_full=None,
                y=y,
                u=u,
            )
            for step in _post_steps:
                step(ctx)

    return u_hat, x_hat
