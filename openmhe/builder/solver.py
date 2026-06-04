"""Build and run Acados sliding-window MHE solvers from :class:`~openmhe.ObjectiveBuilder`."""

import os

import casadi as ca
import numpy as np
import scipy.signal as signal
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

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


def build_mhe_solver(
    A,
    B,
    C,
    D,
    N_horizon,
    builder: ObjectiveBuilder,
    dt: float = 0.001,
    P_arrival: np.ndarray | None = None,
    already_discrete: bool = False,
    input_as_state: list[int] | None = None,
):
    """Build an Acados MHE solver from an objective.

    Uses ``LINEAR_LS`` when every term has :class:`~openmhe.L2Penalty`; otherwise
    ``CONVEX_OVER_NONLINEAR`` for L1, Huber, or dead-zone penalties.

    Unknown inputs can be modeled with :class:`~openmhe.InputRandomWalk` (preferred)
    or the legacy ``input_as_state`` keyword.

    Each column of ``B`` must be assigned exactly once via regulated terms
    (RW/FD/SD), :class:`~openmhe.KnownInput`, or :class:`~openmhe.InputTrackingTerm`.
    """
    validate_term_penalties(builder)
    ensure_acados_environment()

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float) if D is not None else np.zeros((C.shape[0], B.shape[1]))

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
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.nlp_solver_max_iter = 100

    pin_u_idx: list[int] = []
    if known_inputs:
        pin_u_idx = [controlled_idx.index(ui) for ui in known_inputs]
        pin_u_idx = sorted(pin_u_idx)
        ocp.constraints.idxbu = np.asarray(pin_u_idx, dtype=int)
        ocp.constraints.lbu = np.zeros(len(pin_u_idx))
        ocp.constraints.ubu = np.zeros(len(pin_u_idx))

    use_conl = needs_conl(builder)
    if use_conl:
        arrival_slice, n_residual_0 = build_conl_cost(
            ocp, model, Vx, Vu, builder, n_residual, nx_base, P_arrival
        )
    else:
        arrival_slice, n_residual_0 = build_linear_ls_cost(
            ocp, Vx, Vu, builder, n_residual, nx, nx_base, P_arrival
        )

    mode = cost_mode_tag(builder)
    n_known = len(known_inputs)
    json_name = (
        f"mhe_{mode}_nr{n_residual}_nx{nx}_nu{nu}"
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
    solver._known_inputs = known_inputs
    solver._pin_u_idx = pin_u_idx
    solver._cost_mode = mode
    return solver


def run_solver(solver, y, u):
    """Run sliding-window MHE over a measurement sequence.

    Each window covers time indices ``[k - N, …, k - 1]``. Returned
    ``u_hat[:, idx]`` and ``x_hat[:, idx]`` are the solution at the last
    stage of that window (time ``k - 1``), not ``k``. Align ground truth as
    ``u[:, N - 1 : N - 1 + n_est]`` (and the same for ``x``).
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
    u_hat = np.full((nu, n_est), np.nan)
    x_hat = np.full((nx_base, n_est), np.nan)
    x_prior = np.zeros(solver._nx)
    unmeasured = unmeasured_regulator_indices(
        builder, solver._rw_indices, solver._fd_indices, solver._sd_indices
    )

    for idx, k in enumerate(range(N, n_steps)):
        u_seed = seed_reg_state_prior(
            x_prior,
            rw_col=load_state_col,
            fd_col=solver._col_fd1,
            sd1_col=solver._col_sd1,
            sd2_col=solver._col_sd2,
            rw_indices=solver._rw_indices,
            fd_indices=solver._fd_indices,
            sd_indices=solver._sd_indices,
            u_hat=u_hat,
            window_idx=idx,
            unmeasured=unmeasured,
            u=u,
            t_start=k - N,
        )
        solver.set(0, "x", x_prior.copy())

        if idx > 0 and u_seed:
            u_guess = np.array(solver.get(0, "u")).ravel()
            for ui, val in u_seed.items():
                if ui in unmeasured and ui in controlled_idx:
                    u_guess[controlled_idx.index(ui)] = val
            solver.set(0, "u", u_guess)

        for j in range(N):
            t_idx = k - N + j
            y_meas = y[:, t_idx] if y.ndim > 1 else y[t_idx]
            yref = _build_yref_stack(
                builder, y_meas, u[:, t_idx], nx_base, nu, nw
            )

            if solver._pin_u_idx and u is not None:
                pin_vals = np.array(
                    [float(u[ui, t_idx]) for ui in solver._known_inputs]
                )
                solver.set(j, "lbu", pin_vals)
                solver.set(j, "ubu", pin_vals)

            if j == 0 and solver._arrival_slice is not None:
                yref0 = np.zeros(solver._n_residual_0)
                yref0[: len(yref)] = yref
                yref0[solver._arrival_slice] = x_prior[:nx_base]
                solver.set(0, "yref", yref0)
            else:
                solver.set(j, "yref", yref)

        if solver.solve() != 0:
            continue

        x_end = np.array(solver.get(N - 1, "x")).ravel()
        u_full = np.array(solver.get(N - 1, "u")).ravel()
        x_hat[:, idx] = x_end[:nx_base]
        for ui in controlled_idx:
            u_hat[ui, idx] = u_full[controlled_idx.index(ui)]
        for ui, col in load_state_col.items():
            u_hat[ui, idx] = x_end[col]
        for ui, col in solver._col_sd1.items():
            if ui not in controlled_idx:
                u_hat[ui, idx] = x_end[col]
        for ui, col in solver._col_fd1.items():
            if ui not in controlled_idx and ui not in load_state_col:
                u_hat[ui, idx] = x_end[col]

        if solver._arrival_slice is not None:
            x_prior = np.array(solver.get(1, "x")).ravel()
        else:
            x_prior = np.array(solver.get(0, "x")).ravel()

        for j in range(N - 1):
            solver.set(j, "x", np.array(solver.get(j + 1, "x")).ravel())
            solver.set(j, "u", np.array(solver.get(j + 1, "u")).ravel())

    return u_hat, x_hat


run_sliding_mhe = run_solver
