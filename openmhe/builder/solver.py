"""Build and run Acados sliding-window MHE solvers from :class:`~openmhe.ObjectiveBuilder`."""

import os
from dataclasses import dataclass

import casadi as ca
import numpy as np
import scipy.signal as signal
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from openmhe.frontend.system import SystemModel
from openmhe.mhe_strategies import ObjectiveBuilder
from openmhe.frontend.acados_runtime import acados_root, ensure_acados_environment
from openmhe.paths import get_codegen_dir, mhe_json_path
from openmhe.builder.cost import (
    build_conl_cost,
    build_linear_ls_cost,
    cost_mode_tag,
    needs_conl,
    validate_term_penalties,
)
from openmhe.builder.input_regs import (
    collect_fd_indices,
    collect_rw_indices,
    collect_sd_indices,
    cost_terms,
    merge_process_weight,
    seed_reg_state_prior,
    term_kind,
    unmeasured_regulator_indices,
    validate_input_models,
    validate_input_partition,
)


def _configure_acados_codegen(ocp: AcadosOcp) -> None:
    """Point Acados codegen at the local install and OpenMHE output directories."""
    acados_path = acados_root()
    lib_dir = os.path.join(acados_path, "lib")
    opts = ocp.code_gen_opts
    opts.acados_lib_path = lib_dir
    opts._AcadosCodeGenOpts__acados_include_path = os.path.join(
        acados_path, "include"
    ).replace(os.sep, "/")
    opts.code_export_directory = str(get_codegen_dir()).replace(os.sep, "/")


def _u_expr(ui, controlled_idx, u_ctrl, x, col_rw):
    """CasADi expression for physical input ``ui`` (control or random-walk state)."""
    if ui in controlled_idx:
        return u_ctrl[controlled_idx.index(ui)]
    return x[col_rw[ui]]


def _build_yref_stack(builder, y_meas, u_known, nx_base, nu, nw):
    """Stack per-term measurement references for one horizon stage."""
    parts = []
    for term in builder.terms:
        kind = term_kind(term)
        if kind == "MEASUREMENT":
            parts.append(np.asarray(y_meas).ravel())
        elif kind == "PROCESS":
            parts.append(np.zeros(nw))
        elif kind == "INPUT_TRACKING":
            idx = np.asarray(term.target_idx, dtype=int)
            ref = np.zeros(len(idx))
            for i, ui in enumerate(idx):
                if getattr(term, "reference", "measured") == "measured":
                    ref[i] = u_known[ui]
                elif term.reference == "zero":
                    ref[i] = 0.0
                else:
                    ref[i] = float(term.reference)
            parts.append(ref)
        elif kind == "INPUT_REG":
            parts.append(np.zeros(len(term.target_idx)))
        elif kind in ("INPUT_RANDOM_WALK", "KNOWN_INPUT"):
            continue
        else:
            raise ValueError(f"Unsupported term type: {kind}")
    return np.concatenate(parts)


def _precompute_yrefs(builder, y, u, nx_base, nu, nw, n_steps):
    """Build the stacked ``yref`` vector for every time index once."""
    y_2d = y if y.ndim > 1 else y.reshape(1, -1)
    u_2d = u if u.ndim > 1 else u.reshape(1, -1)
    first = _build_yref_stack(builder, y_2d[:, 0], u_2d[:, 0], nx_base, nu, nw)
    yrefs = np.empty((n_steps, first.size))
    yrefs[0] = first
    for t_idx in range(1, n_steps):
        yrefs[t_idx] = _build_yref_stack(
            builder, y_2d[:, t_idx], u_2d[:, t_idx], nx_base, nu, nw
        )
    return yrefs


def _configure_nlp_solver(
    ocp: AcadosOcp,
    *,
    nlp_solver_type: str,
    nlp_solver_max_iter: int | None,
    qp_solver: str,
    N_horizon: int,
) -> None:
    """Apply NLP/QP options tuned for warm-started sliding-window MHE."""
    ocp.solver_options.nlp_solver_type = nlp_solver_type
    if nlp_solver_max_iter is None:
        nlp_solver_max_iter = 1 if nlp_solver_type == "SQP_RTI" else 50
    ocp.solver_options.nlp_solver_max_iter = nlp_solver_max_iter
    ocp.solver_options.qp_solver = qp_solver
    if qp_solver == "PARTIAL_CONDENSING_HPIPM":
        ocp.solver_options.hpipm_mode = "SPEED_ABS"
        ocp.solver_options.qp_solver_cond_N = N_horizon
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.nlp_solver_warm_start_first_qp = True
    else:
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.nlp_solver_warm_start_first_qp_from_nlp = False
    ocp.solver_options.nlp_solver_tol_stat = 1e-4
    ocp.solver_options.nlp_solver_tol_eq = 1e-4
    ocp.solver_options.nlp_solver_tol_ineq = 1e-4


@dataclass
class WindowStep:
    """Context passed to each post-step callable after a window solve.

    Fields
    ------
    idx : int
        Column index into ``x_hat`` / ``u_hat`` for the current window.
    t_start : int
        Time index of the first stage in this window (``k - N``).
    k : int
        Time index one past the last stage (``k - 1`` is the last stage).
    N : int
        Horizon length.
    dt : float
        Sample period in seconds.
    nx_base : int
        Number of plant state variables (excludes augmented input states).
    nu : int
        Total number of physical inputs.
    x_hat : np.ndarray
        Full ``(nx_base, n_est)`` state estimate array, **mutable in place**.
    u_hat : np.ndarray
        Full ``(nu, n_est)`` input estimate array, **mutable in place**.
    x_full : np.ndarray
        Augmented state vector at the last horizon stage (length ``nx``).
    y : np.ndarray
        Raw measurement sequence ``(ny, n_steps)``.
    u : np.ndarray
        Raw known-input sequence ``(nu, n_steps)``.
    """

    idx: int
    t_start: int
    k: int
    N: int
    dt: float
    nx_base: int
    nu: int
    x_hat: np.ndarray
    u_hat: np.ndarray
    x_full: np.ndarray
    y: np.ndarray
    u: np.ndarray


def build_mhe_solver(
    mhe_system: SystemModel,
    N_horizon,
    builder: ObjectiveBuilder,
    dt: float = 0.001,
    P_arrival: np.ndarray | None = None,
    already_discrete: bool = False,
    input_as_state: list[int] | None = None,
    nlp_solver_type: str = "SQP_RTI",
    nlp_solver_max_iter: int | None = None,
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM",
):
    """Build an Acados MHE solver from an objective.

    Uses ``LINEAR_LS`` when every term has :class:`~openmhe.L2Penalty`; otherwise
    ``CONVEX_OVER_NONLINEAR`` for L1, Huber, or dead-zone penalties.

    Unknown inputs can be modeled with :class:`~openmhe.InputRandomWalk` (preferred)
    or the legacy ``input_as_state`` keyword.

    Each column of ``B`` must be assigned exactly once via regulated terms
    (RW/FD/SD), :class:`~openmhe.KnownInput`, or :class:`~openmhe.InputTrackingTerm`.

    Arrival cost at the first stage of each window can be supplied either as a
    fixed matrix ``P_arrival`` or via ``builder.arrival_cost``
    (:class:`~openmhe.SteadyStateArrivalCost`, :class:`~openmhe.EKFArrivalCost`,
    :class:`~openmhe.UKFArrivalCost`). Pass only one of ``P_arrival`` or
    ``builder.arrival_cost``.
    """
    arrival_cost = builder.arrival_cost
    if P_arrival is not None and arrival_cost is not None:
        raise ValueError("Pass only one of P_arrival or builder.arrival_cost.")
    if arrival_cost is not None:
        P_arrival = arrival_cost.initial_covariance(mhe_system, builder)
        if arrival_cost.is_dynamic and needs_conl(builder):
            raise ValueError(
                "Dynamic arrival costs require LINEAR_LS (all L2 penalties)."
            )
    validate_term_penalties(builder)
    ensure_acados_environment()

    A = np.asarray(mhe_system.A, dtype=float)
    B = np.asarray(mhe_system.B, dtype=float)
    C = np.asarray(mhe_system.C, dtype=float)
    D = np.asarray(mhe_system.D, dtype=float) if mhe_system.D is not None else np.zeros((C.shape[0], B.shape[1]))

    nx_base = A.shape[0]
    nu = B.shape[1]
    ny = C.shape[0]

    if already_discrete:
        A_d, B_d, C_d, D_d = A, B, C, D
    else:
        A_d, B_d, C_d, D_d, _ = signal.cont2discrete((A, B, C, D), dt=dt)

    rw_indices, rw_lambdas = collect_rw_indices(builder, input_as_state)
    fd_indices = collect_fd_indices(builder)
    sd_indices = collect_sd_indices(builder)
    validate_input_models(rw_indices, fd_indices, sd_indices)
    known_inputs = validate_input_partition(
        builder, nu, rw_indices, fd_indices, sd_indices
    )

    n_rw = len(rw_indices)
    n_fd = len(fd_indices)
    n_sd = len(sd_indices)
    input_as_state = rw_indices
    controlled_idx = [i for i in range(nu) if i not in input_as_state]
    nu_ctrl = len(controlled_idx)

    col_rw = {ui: nx_base + j for j, ui in enumerate(rw_indices)}
    col_fd1 = {ui: nx_base + n_rw + j for j, ui in enumerate(fd_indices)}
    col_sd1 = {ui: nx_base + n_rw + n_fd + j for j, ui in enumerate(sd_indices)}
    col_sd2 = {
        ui: nx_base + n_rw + n_fd + n_sd + j for j, ui in enumerate(sd_indices)
    }

    nw = nx_base + n_rw
    process_terms = [t for t in builder.terms if term_kind(t) == "PROCESS"]
    if len(process_terms) != 1:
        raise ValueError("Objective must contain exactly one ProcessTerm.")
    if n_rw:
        W_proc = merge_process_weight(
            process_terms[0], nx_base, rw_indices, rw_lambdas
        )
        process_terms[0].weight.W = W_proc
    elif process_terms[0].weight.dim != nx_base:
        raise ValueError(
            f"ProcessTerm weight dim {process_terms[0].weight.dim} must equal "
            f"nx={nx_base} when no InputRandomWalk terms are present."
        )
    process_terms[0].weight.dim = nw

    nx = nx_base + n_rw + n_fd + 2 * n_sd
    x_base = ca.SX.sym("x_base", nx_base)
    parts = [x_base]
    if n_rw:
        parts.append(ca.SX.sym("x_rw", n_rw))
    if n_fd:
        parts.append(ca.SX.sym("x_fd1", n_fd))
    if n_sd:
        parts.append(ca.SX.sym("x_sd1", n_sd))
        parts.append(ca.SX.sym("x_sd2", n_sd))
    x = ca.vertcat(*parts) if len(parts) > 1 else x_base

    nu_err = nu_ctrl + nw
    u = ca.SX.sym("u", nu_err)
    u_ctrl = u[:nu_ctrl]
    w = u[nu_ctrl:nu_err]

    dyn_base = A_d @ x_base
    for ui in controlled_idx:
        dyn_base += B_d[:, ui] * u_ctrl[controlled_idx.index(ui)]
    for ui in rw_indices:
        dyn_base += B_d[:, ui] * x[col_rw[ui]]
    dyn_base += w[:nx_base]

    dyn_parts = [dyn_base]
    if n_rw:
        dyn_parts.append(
            x[nx_base : nx_base + n_rw] + w[nx_base : nx_base + n_rw]
        )
    if n_fd:
        fd_next = [
            _u_expr(ui, controlled_idx, u_ctrl, x, col_rw) for ui in fd_indices
        ]
        dyn_parts.append(ca.vertcat(*fd_next))
    if n_sd:
        sd_d1_next = [
            _u_expr(ui, controlled_idx, u_ctrl, x, col_rw) for ui in sd_indices
        ]
        sd_d2_next = [x[col_sd1[ui]] for ui in sd_indices]
        dyn_parts.append(ca.vertcat(*sd_d1_next, *sd_d2_next))

    disc_dyn_expr = ca.vertcat(*dyn_parts) if len(dyn_parts) > 1 else dyn_parts[0]

    model = AcadosModel()
    model.name = "openmhe_mhe"
    model.x = x
    model.u = u
    model.disc_dyn_expr = disc_dyn_expr

    def _residual_dim(term):
        """Row count for one term in the stacked LS / CONL residual."""
        if term_kind(term) == "PROCESS":
            return nw
        return term.weight.dim

    n_residual = sum(_residual_dim(t) for t in cost_terms(builder))
    Vx = np.zeros((n_residual, nx))
    Vu = np.zeros((n_residual, nu_err))
    row = 0

    for term in builder.terms:
        kind = term_kind(term)
        if kind == "MEASUREMENT":
            Vx[row : row + ny, :nx_base] = C_d
            if D_d is not None and np.any(D_d != 0):
                for ui in range(nu):
                    d_col = D_d[:, ui]
                    if not np.any(d_col != 0):
                        continue
                    if ui in controlled_idx:
                        Vu[row : row + ny, controlled_idx.index(ui)] = d_col
                    elif ui in col_rw:
                        Vx[row : row + ny, col_rw[ui]] = d_col
                    else:
                        raise ValueError(
                            f"D[:, {ui}] is nonzero but input {ui} is neither "
                            "controlled nor modeled with InputRandomWalk."
                        )
            row += ny
        elif kind == "PROCESS":
            Vu[row : row + nw, nu_ctrl : nu_ctrl + nw] = np.eye(nw)
            row += nw
        elif kind == "INPUT_TRACKING":
            idx = np.asarray(term.target_idx, dtype=int)
            for i, ui in enumerate(idx):
                if ui in input_as_state:
                    raise ValueError(
                        f"Cannot track input {ui} as measured; it is modeled as a state."
                    )
                if ui in known_inputs:
                    raise ValueError(
                        f"Cannot track input {ui}; it is declared as KnownInput."
                    )
                Vu[row + i, controlled_idx.index(ui)] = 1.0
            row += len(idx)
        elif kind == "KNOWN_INPUT":
            continue
        elif kind == "INPUT_REG":
            trend = str(term.trend).upper()
            idx = np.asarray(term.target_idx, dtype=int)
            for ui in idx:
                if ui not in controlled_idx:
                    raise ValueError(
                        f"Input regulator on index {ui} requires a controlled input."
                    )
                ci = controlled_idx.index(ui)
                Vu[row, ci] = 1.0
                if trend in ("FIRST_DIFF", "FIRST"):
                    Vx[row, col_fd1[ui]] = -1.0
                elif trend == "SECOND_DIFF":
                    Vx[row, col_sd1[ui]] = -2.0
                    Vx[row, col_sd2[ui]] = 1.0
                else:
                    raise ValueError(f"Unsupported INPUT_REG trend: {term.trend}")
                row += 1
        elif kind == "INPUT_RANDOM_WALK":
            continue
        else:
            raise ValueError(f"Unsupported term type: {kind}")

    ocp = AcadosOcp()
    _configure_acados_codegen(ocp)
    ocp.model = model
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = N_horizon * dt
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    _configure_nlp_solver(
        ocp,
        nlp_solver_type=nlp_solver_type,
        nlp_solver_max_iter=nlp_solver_max_iter,
        qp_solver=qp_solver,
        N_horizon=N_horizon,
    )

    pin_u_idx: list[int] = []
    if known_inputs:
        pin_u_idx = [controlled_idx.index(ui) for ui in known_inputs]
        pin_u_idx = sorted(pin_u_idx)
        ocp.constraints.idxbu = np.asarray(pin_u_idx, dtype=int)
        ocp.constraints.lbu = np.zeros(len(pin_u_idx))
        ocp.constraints.ubu = np.zeros(len(pin_u_idx))

    use_conl = needs_conl(builder)
    if use_conl:
        arrival_slice, n_residual_0, W0_template = build_conl_cost(
            ocp, model, Vx, Vu, builder, n_residual, nx_base, P_arrival
        )
    else:
        arrival_slice, n_residual_0, W0_template = build_linear_ls_cost(
            ocp, Vx, Vu, builder, n_residual, nx, nx_base, P_arrival
        )

    mode = cost_mode_tag(builder)
    n_known = len(known_inputs)
    json_name = (
        f"mhe_{mode}_{nlp_solver_type.lower()}_nr{n_residual}_nx{nx}_nu{nu}"
        f"_nrw{n_rw}_nfd{n_fd}_nsd{n_sd}_nk{n_known}_N{N_horizon}.json"
    )
    solver = AcadosOcpSolver(ocp, json_file=mhe_json_path(json_name))
    solver._nu = nu
    solver._nu_ctrl = nu_ctrl
    solver._nx = nx
    solver._nx_base = nx_base
    solver._nw = nw
    solver._N = N_horizon
    solver._builder = builder
    solver._arrival_slice = arrival_slice
    solver._n_residual_0 = n_residual_0
    solver._rw_indices = rw_indices
    solver._fd_indices = fd_indices
    solver._sd_indices = sd_indices
    solver._load_state_col = col_rw
    solver._col_fd1 = col_fd1
    solver._col_sd1 = col_sd1
    solver._col_sd2 = col_sd2
    solver._controlled_idx = controlled_idx
    solver._controlled_map = {ui: i for i, ui in enumerate(controlled_idx)}
    solver._known_inputs = known_inputs
    solver._pin_u_idx = pin_u_idx
    solver._cost_mode = mode
    solver._arrival_cost = arrival_cost
    solver._W0_template = W0_template
    solver._arrival_W_slice = (
        slice(n_residual, n_residual + nx_base) if arrival_slice is not None else None
    )
    solver._dt = dt
    solver._system = mhe_system
    return solver


def run_solver(solver, y, u, post_steps=None):
    """Run sliding-window MHE over a measurement sequence.

    Each window covers time indices ``[k - N, …, k - 1]``. Returned
    ``u_hat[:, idx]`` and ``x_hat[:, idx]`` are the solution at the last
    stage of that window (time ``k - 1``), not ``k``. Align ground truth as
    ``u[:, N - 1 : N - 1 + n_est]`` (and the same for ``x``).

    Parameters
    ----------
    post_steps : list, optional
        Callables invoked once per window **after** the estimates for that
        window are written to ``u_hat`` / ``x_hat``. Each callable receives a
        :class:`WindowStep` context and may modify ``ctx.u_hat[:, ctx.idx]``
        or ``ctx.x_hat[:, ctx.idx]`` in place. The raw MHE output is always
        used for regulator-state seeding of the next window, so post-step
        corrections affect only the returned arrays and are never fed back into
        the solver. Steps are skipped on failed solves (the corresponding
        column stays ``NaN``). If a step exposes a ``reset()`` method it is
        called once before the loop starts.
    """
    n_steps = y.shape[1] if y.ndim > 1 else len(y)
    nu = solver._nu
    nx_base = solver._nx_base
    N = solver._N
    builder = solver._builder
    load_state_col = solver._load_state_col
    controlled_idx = solver._controlled_idx
    nw = solver._nw

    n_est = n_steps - N
    if n_est <= 0:
        raise ValueError(
            f"Measurement sequence has {n_steps} samples but horizon is N={N}; "
            "need at least N+1 samples."
        )

    u_hat = np.full((nu, n_est), np.nan)
    x_hat = np.full((nx_base, n_est), np.nan)
    # Keeps raw MHE output for regulator-state seeding so that post-step
    # corrections to u_hat are never fed back into the solver.
    _u_hat_raw = np.full((nu, n_est), np.nan)
    x_prior = np.zeros(solver._nx)
    controlled_map = getattr(solver, "_controlled_map", None)
    if controlled_map is None:
        controlled_map = {ui: i for i, ui in enumerate(controlled_idx)}
    unmeasured = unmeasured_regulator_indices(
        builder, solver._rw_indices, solver._fd_indices, solver._sd_indices
    )

    yrefs = _precompute_yrefs(builder, y, u, nx_base, nu, nw, n_steps)
    pin_vals_all = None
    if solver._pin_u_idx and u is not None:
        known_inputs = np.asarray(solver._known_inputs, dtype=int)
        pin_vals_all = u[known_inputs, :].T

    arrival_cost = getattr(solver, "_arrival_cost", None)
    if arrival_cost is not None:
        arrival_cost.reset()

    _post_steps = list(post_steps) if post_steps is not None else []
    for step in _post_steps:
        if callable(getattr(step, "reset", None)):
            step.reset()

    has_arrival = solver._arrival_slice is not None
    yref0 = np.zeros(solver._n_residual_0) if has_arrival else None
    arrival_slice = solver._arrival_slice
    n_residual_0 = solver._n_residual_0
    W0_template = solver._W0_template
    arrival_W_slice = solver._arrival_W_slice
    pin_u_idx = solver._pin_u_idx
    dynamic_arrival = (
        arrival_cost is not None
        and arrival_cost.is_dynamic
        and W0_template is not None
    )

    for idx, k in enumerate(range(N, n_steps)):
        t_start = k - N
        x_bar = x_prior[:nx_base]
        P_arr = None
        if arrival_cost is not None:
            x_bar, P_arr = arrival_cost.window_prior(t_start, y, u)

        u_seed = seed_reg_state_prior(
            x_prior,
            rw_col=load_state_col,
            fd_col=solver._col_fd1,
            sd1_col=solver._col_sd1,
            sd2_col=solver._col_sd2,
            rw_indices=solver._rw_indices,
            fd_indices=solver._fd_indices,
            sd_indices=solver._sd_indices,
            u_hat=_u_hat_raw,
            window_idx=idx,
            unmeasured=unmeasured,
            u=u,
            t_start=t_start,
        )
        solver.set(0, "x", x_prior)

        if idx > 0 and u_seed:
            u_guess = np.array(solver.get(0, "u")).ravel()
            for ui, val in u_seed.items():
                ci = controlled_map.get(ui)
                if ci is not None and ui in unmeasured:
                    u_guess[ci] = val
            solver.set(0, "u", u_guess)

        for j in range(N):
            t_idx = k - N + j
            yref = yrefs[t_idx]

            if pin_u_idx and pin_vals_all is not None:
                pin_vals = pin_vals_all[t_idx]
                solver.set(j, "lbu", pin_vals)
                solver.set(j, "ubu", pin_vals)

            if j == 0 and has_arrival:
                yref0[: yref.size] = yref
                if arrival_cost is not None:
                    yref0[arrival_slice] = x_bar
                else:
                    yref0[arrival_slice] = x_prior[:nx_base]
                solver.set(0, "yref", yref0)
                if dynamic_arrival:
                    W0 = W0_template.copy()
                    W0[arrival_W_slice, arrival_W_slice] = np.linalg.inv(P_arr)
                    solver.cost_set(0, "W", W0, api="new")
            else:
                solver.set(j, "yref", yref)

        if solver.solve() != 0:
            continue

        x_end = np.array(solver.get(N - 1, "x")).ravel()
        u_full = np.array(solver.get(N - 1, "u")).ravel()
        x_hat[:, idx] = x_end[:nx_base]
        for ui in controlled_idx:
            u_hat[ui, idx] = u_full[controlled_map[ui]]
        for ui, col in load_state_col.items():
            u_hat[ui, idx] = x_end[col]
        for ui, col in solver._col_sd1.items():
            if ui not in controlled_map:
                u_hat[ui, idx] = x_end[col]
        for ui, col in solver._col_fd1.items():
            if ui not in controlled_map and ui not in load_state_col:
                u_hat[ui, idx] = x_end[col]

        _u_hat_raw[:, idx] = u_hat[:, idx]

        if _post_steps:
            ctx = WindowStep(
                idx=idx,
                t_start=t_start,
                k=k,
                N=N,
                dt=solver._dt,
                nx_base=nx_base,
                nu=nu,
                x_hat=x_hat,
                u_hat=u_hat,
                x_full=x_end,
                y=y,
                u=u,
            )
            for step in _post_steps:
                step(ctx)

        if has_arrival:
            x_prior = np.array(solver.get(1, "x")).ravel()
        else:
            x_prior = np.array(solver.get(0, "x")).ravel()

        for j in range(N - 1):
            solver.set(j, "x", np.array(solver.get(j + 1, "x")).ravel())
            solver.set(j, "u", np.array(solver.get(j + 1, "u")).ravel())

    return u_hat, x_hat


run_sliding_mhe = run_solver
