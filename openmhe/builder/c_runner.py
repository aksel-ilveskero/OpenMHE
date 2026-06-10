"""Run sliding-window MHE via the compiled C driver (``libopenmhe_mhe_run.so``)."""

from __future__ import annotations

import ctypes
import time
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from openmhe.builder.input_regs import unmeasured_regulator_indices
from openmhe.builder.solver import WindowStep, _lti_fast_enabled, _precompute_yrefs
from openmhe.frontend.acados_runtime import (
    acados_root,
    blasfeo_target_define,
    ensure_acados_environment,
)
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
        ("n_arrival", ctypes.c_int),
        ("has_arrival", ctypes.c_int),
        ("arrival_off", ctypes.c_int),
        ("dynamic_arrival", ctypes.c_int),
        ("n_pin", ctypes.c_int),
        ("n_u_extract", ctypes.c_int),
        ("n_rw", ctypes.c_int),
        ("n_fd", ctypes.c_int),
        ("n_sd", ctypes.c_int),
        ("n_unmeasured", ctypes.c_int),
        ("lti_linear_ls_fast", ctypes.c_int),
        ("linear_ls", ctypes.c_int),
    ]


class _FilterSetup(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("nx_base", ctypes.c_int),
        ("ny", ctypes.c_int),
        ("nu", ctypes.c_int),
        ("n_arrival", ctypes.c_int),
        ("arrival_w_off", ctypes.c_int),
        ("A", ctypes.c_void_p),
        ("B", ctypes.c_void_p),
        ("C", ctypes.c_void_p),
        ("D", ctypes.c_void_p),
        ("Q", ctypes.c_void_p),
        ("R", ctypes.c_void_p),
        ("W0_template", ctypes.c_void_p),
    ]


_OPENMHE_FILTER_EKF = 2


def _build_filter_setup(solver, has_arrival: bool):
    """Build ctypes filter setup for in-C EKF arrival, or ``None``."""
    if not has_arrival or getattr(solver, "_filter_kind", None) != "ekf":
        return None
    w0_template = getattr(solver, "_W0_template_f", None)
    if w0_template is None and solver._W0_template is not None:
        w0_template = np.asfortranarray(solver._W0_template, dtype=np.float64)
        solver._W0_template_f = w0_template
    if w0_template is None:
        return None

    def _ptr(arr):
        return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data_as(
            ctypes.c_void_p
        )

    return _FilterSetup(
        kind=_OPENMHE_FILTER_EKF,
        nx_base=int(solver._nx_base),
        ny=int(solver._system.ny),
        nu=int(solver._system.nu),
        n_arrival=int(solver._n_arrival),
        arrival_w_off=int(solver._arrival_slice.start),
        A=_ptr(solver._filter_A),
        B=_ptr(solver._filter_B),
        C=_ptr(solver._filter_C),
        D=_ptr(solver._filter_D),
        Q=_ptr(solver._filter_Q),
        R=_ptr(solver._filter_R),
        W0_template=_ptr(w0_template),
    )


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
    env["BLASFEO_TARGET_DEFINE"] = blasfeo_target_define()
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
    """Convert a dictionary of indices to arrays."""
    if not mapping:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    keys = sorted(mapping.keys())
    return (
        np.asarray(keys, dtype=np.int32),
        np.asarray([mapping[k] for k in keys], dtype=np.int32),
    )


def run_c_solver(
    solver,
    y,
    u,
    post_steps=None,
    *,
    rebuild: bool = False,
    lti_linear_ls_fast: bool | None = None,
):
    """Run the C based MHE solver.

    Returns the estimated control and state sequences over the entire measurement.

    Parameters
    ----------
    solver : Solver
        The solver object to use.
    y : array_like
        The measurement sequence.
    u : array_like
        The control sequence.
    post_steps : list of callable, optional
        Post-processing steps to apply after the C loop.
    rebuild : bool, optional
        Force recompilation of ``libopenmhe_mhe_run.so`` when codegen or C
        sources changed.
    lti_linear_ls_fast : bool or None, optional
        Override ``solver._lti_linear_ls_fast``.  When enabled (default for
        all-L2 builds with ``SQP_RTI``), window 0 runs a full Acados solve and
        condenses the QP left-hand side; later windows refresh QP vectors only.
        Requires ``nlp_solver_type='SQP_RTI'``; otherwise a warning is emitted and the full solve is used.  See ``openmhe/c_solver/README.md``.
    
    Returns
    -------
    u_hat : array_like
        The estimated control sequence.
    x_hat : array_like
        The estimated state sequence.

    Raises
    ------
    TypeError
        If the arrival cost is a UKFArrivalCost.
    RuntimeError
        If the C MHE driver fails.
    """

    
    from openmhe.mhe_strategies.arrival_cost import UKFArrivalCost

    arrival_cost = getattr(solver, "_arrival_cost", None)
    if isinstance(arrival_cost, UKFArrivalCost):
        raise TypeError(
            "run_c_solver does not support UKFArrivalCost yet; use EKFArrivalCost "
            "or run_solver() for UKF."
        )

    # Setup libraries, functions and types
    lib = _load_run_lib(rebuild=rebuild)
    lib.openmhe_mhe_acados_create_capsule.restype = ctypes.c_void_p
    lib.openmhe_mhe_acados_free_capsule.argtypes = [ctypes.c_void_p]
    lib.openmhe_mhe_init_solver.argtypes = [ctypes.c_void_p]
    lib.openmhe_mhe_init_solver.restype = ctypes.c_int
    lib.openmhe_mhe_free_solver.argtypes = [ctypes.c_void_p]
    lib.openmhe_mhe_free_solver.restype = ctypes.c_int
    _c_double_p = ctypes.POINTER(ctypes.c_double)
    _c_int_p = ctypes.POINTER(ctypes.c_int)

    # Define the arguments for the C function
    lib.openmhe_mhe_run_sliding.argtypes = [
        ctypes.c_void_p, # capsule
        ctypes.POINTER(_RunConfig), # cfg
        _c_double_p, # yref
        _c_double_p, # x_bar_pre
        _c_double_p, # W0_stage_pre
        _c_double_p, # pin_vals
        _c_int_p, # controlled_idx
        _c_int_p, # u_extract_raw
        _c_int_p, # rw_idx
        _c_int_p, # rw_col
        _c_int_p, # fd_idx
        _c_int_p, # fd_col
        _c_int_p, # sd_idx
        _c_int_p, # sd1_col
        _c_int_p, # sd2_col
        _c_int_p, # unmeasured_ui
        _c_int_p, # arrival_state_idx
        _c_double_p, # u_meas
        _c_double_p, # y_meas
        ctypes.POINTER(_FilterSetup), # filter_setup
        _c_double_p, # u_hat
        _c_double_p, # x_hat
        _c_double_p, # u_raw_hat
    ]

    # C function return type is int status code
    lib.openmhe_mhe_run_sliding.restype = ctypes.c_int

    # Convert input arrays to numpy arrays
    y = np.asarray(y, dtype=np.float64, order="C")
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Set basic dimensions
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
    
    # Precompute measured state references
    yrefs = _precompute_yrefs(builder, y, u, nx_base, nu, nw, n_steps)
    ny_stage = yrefs.shape[1]

    u_hat_rm = np.full((n_est, nu), np.nan, order="C")
    u_raw_rm = np.full((n_est, nu), np.nan, order="C")
    x_hat_rm = np.full((n_est, nx_base), np.nan, order="C")

    # Check for arrival cost
    has_arrival = solver._arrival_slice is not None
    dynamic_arrival = (
        has_arrival
        and getattr(solver, "_filter_kind", None) == "ekf"
        and solver._W0_template is not None
    )

    # Build arrival cost
    filter_setup = _build_filter_setup(solver, has_arrival)

    # Convert measurement array to contiguous array
    y_meas = np.ascontiguousarray(y.T, dtype=np.float64)

    # Check for pinned (known) inputs
    pin_vals = None
    if solver._pin_u_idx and u is not None:
        known_inputs = np.asarray(solver._known_inputs, dtype=int)
        pin_vals = np.ascontiguousarray(u[known_inputs, :].T)

    # Build index arrays for different input regulator types
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
        n_arrival=int(solver._n_arrival) if has_arrival else 0,
        has_arrival=int(has_arrival),
        arrival_off=int(solver._arrival_slice.start) if has_arrival else 0,
        dynamic_arrival=int(dynamic_arrival),
        n_pin=len(solver._pin_u_idx) if pin_vals is not None else 0,
        n_u_extract=len(_u_extract_specs(solver)) // 3,
        n_rw=len(rw_idx),
        n_fd=len(fd_idx),
        n_sd=len(sd_idx),
        n_unmeasured=len(unmeasured_ui),
        lti_linear_ls_fast=int(
            _lti_fast_enabled(
                solver._lti_linear_ls_fast
                if lti_linear_ls_fast is None
                else lti_linear_ls_fast,
                getattr(solver, "_linear_ls", solver._cost_mode == "ls"),
                getattr(solver, "_nlp_solver_type", "SQP_RTI"),
                stacklevel=2,
            )
        ),
        linear_ls=int(getattr(solver, "_linear_ls", solver._cost_mode == "ls")),
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
    status = -1
    try:
        t_mhe_start = time.perf_counter()
        status = lib.openmhe_mhe_init_solver(capsule)
        if status != 0:
            raise RuntimeError(
                f"Acados solver init failed with status {status}"
            )
        status = lib.openmhe_mhe_run_sliding(
            capsule,
            ctypes.byref(cfg),
            _f64(yrefs).ctypes.data_as(_c_double_p),
            None,
            None,
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
            _i32(solver._arrival_state_idx).ctypes.data_as(_c_int_p)
            if solver._arrival_state_idx is not None
            else None,
            u_meas.ctypes.data_as(_c_double_p) if u_meas is not None else None,
            y_meas.ctypes.data_as(_c_double_p),
            ctypes.byref(filter_setup) if filter_setup is not None else None,
            _f64(u_hat_rm).ctypes.data_as(_c_double_p),
            _f64(x_hat_rm).ctypes.data_as(_c_double_p),
            _f64(u_raw_rm).ctypes.data_as(_c_double_p),
        )
        t_mhe_end = time.perf_counter()
        print(f"MHE time: {t_mhe_end - t_mhe_start:.6f} seconds")
    finally:
        lib.openmhe_mhe_free_solver(capsule)
        lib.openmhe_mhe_acados_free_capsule(capsule)

    if status == -4:
        raise RuntimeError(
            "C MHE driver called before Acados solver init (internal error)."
        )
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
