"""Symbolic LTI + LINEAR_LS MHE cost Hessian via CasADi."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import casadi as ca
import numpy as np

PlantStateNamer = Callable[[int], str]
InputNamer = Callable[[int], str]


def _quadratic_cost(residual: ca.SX, weight: np.ndarray) -> ca.SX:
    W = ca.DM(np.asarray(weight, dtype=float))
    return 0.5 * ca.dot(residual, ca.mtimes(W, residual))


def lti_ls_decision_hessian(
    solver,
    *,
    include_x0: bool = True,
    W0: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build the exact Gauss-Newton Hessian of a constant LTI LINEAR_LS MHE cost.

    OpenMHE stores process noise in the Acados input vector ``u = [u_ctrl; w]``.
    Physical inputs modeled with :class:`~openmhe.InputRandomWalk` live in the
    augmented state ``x``. This helper eliminates intermediate states through the
    LTI dynamics and differentiates the total window cost with respect to

    ``[u_0, …, u_{N-1}, x_0]`` (optionally omitting ``x_0``).

    Path-stage weights ``W`` and dynamics are fixed at build time. Pass ``W0`` to
    use the stage-0 weight matrix from a particular sliding window (e.g. after
    dynamic EKF arrival has updated ``P^{-1}``).

    Parameters
    ----------
    solver
        Output of :func:`~openmhe.build_mhe_solver` with ``LINEAR_LS`` cost.
    include_x0
        When ``True`` (default), include the arrival-cost block on ``x_0``.
    W0
        Optional ``ny0 × ny0`` stage-0 weight override. When omitted, uses the
        weight from solver codegen (initial arrival covariance only).

    Returns
    -------
    dict
        ``H`` — dense symmetric Hessian matrix;
        ``decision_labels`` — ``("u", …, "x0")`` layout description;
        ``nu_ctrl``, ``nw``, ``nx``, ``N`` — problem dimensions;
        ``slices`` — ``u_k``, ``w_k``, ``x0`` index slices into ``H`` when
        ``split_u_noise=True`` metadata is requested via ``layout``.
    """
    if not getattr(solver, "_linear_ls", False):
        raise ValueError(
            "lti_ls_decision_hessian requires LINEAR_LS (all L2) problems."
        )

    A = np.asarray(getattr(solver, "_A_aug"), dtype=float)
    B = np.asarray(getattr(solver, "_B_aug"), dtype=float)
    if A.size == 0 or B.size == 0:
        raise ValueError(
            "Solver is missing _A_aug/_B_aug; rebuild with a recent openmhe version."
        )

    ocp = solver.acados_ocp
    N = int(solver._N)
    nx = int(solver._nx)
    nu_err = int(B.shape[1])
    nu_ctrl = int(getattr(solver, "_nu_ctrl", nu_err))
    nw = int(getattr(solver, "_nw", nu_err - nu_ctrl))

    Vx = np.asarray(ocp.cost.Vx, dtype=float)
    Vu = np.asarray(ocp.cost.Vu, dtype=float)
    W = np.asarray(ocp.cost.W, dtype=float)
    Vx0 = np.asarray(ocp.cost.Vx_0, dtype=float)
    Vu0 = np.asarray(ocp.cost.Vu_0, dtype=float)
    if W0 is None:
        W0_arr = np.asarray(ocp.cost.W_0, dtype=float)
    else:
        W0_arr = np.asarray(W0, dtype=float)
        expected = (int(ocp.cost.W_0.shape[0]), int(ocp.cost.W_0.shape[1]))
        if W0_arr.shape != expected:
            raise ValueError(f"W0 must have shape {expected}, got {W0_arr.shape}.")

    U = [ca.SX.sym(f"u_{k}", nu_err) for k in range(N)]
    x0 = ca.SX.sym("x0", nx)

    x = x0
    J = ca.SX(0)
    for k in range(N):
        if k == 0:
            residual = ca.DM(Vx0) @ x + ca.DM(Vu0) @ U[k]
            J += _quadratic_cost(residual, W0_arr)
        else:
            residual = ca.DM(Vx) @ x + ca.DM(Vu) @ U[k]
            J += _quadratic_cost(residual, W)
        x = ca.DM(A) @ x + ca.DM(B) @ U[k]

    if include_x0:
        decisions = ca.vertcat(*U, x0)
    else:
        # Arrival on x0 is constant when x0 is fixed; keep only input blocks.
        decisions = ca.vertcat(*U)

    H_sym, _ = ca.hessian(J, decisions)
    z0 = np.zeros(int(decisions.shape[0]))
    H = np.array(
        ca.Function("H_fun", [decisions], [H_sym])(z0),
        dtype=float,
    ).reshape(decisions.shape[0], decisions.shape[0])
    H = 0.5 * (H + H.T)

    u_slices = [slice(k * nu_err, (k + 1) * nu_err) for k in range(N)]
    w_slices = [
        slice(s.start + nu_ctrl, s.start + nu_ctrl + nw) for s in u_slices
    ]
    u_ctrl_slices = [slice(s.start, s.start + nu_ctrl) for s in u_slices]
    x0_slice = slice(N * nu_err, N * nu_err + nx) if include_x0 else None

    return {
        "H": H,
        "N": N,
        "nx": nx,
        "nu_err": nu_err,
        "nu_ctrl": nu_ctrl,
        "nw": nw,
        "slices": {
            "u_k": u_slices,
            "u_ctrl_k": u_ctrl_slices,
            "w_k": w_slices,
            "x0": x0_slice,
        },
        "U_stack": ca.vertcat(*U),
        "W_stack": ca.vertcat(*[U[k][nu_ctrl : nu_ctrl + nw] for k in range(N)]),
        "W0": W0_arr.copy(),
    }


def decision_hessian_at_window(
    solver,
    y,
    u,
    window_idx: int,
    *,
    include_x0: bool = True,
) -> dict[str, Any]:
    """Run the Python MHE loop through ``window_idx``, then build the decision Hessian.

    Uses the stage-0 weight ``W0`` active for that window (including dynamic EKF /
    UKF arrival updates). Path-stage ``W`` and LTI dynamics remain at build-time values.

    Parameters
    ----------
    window_idx
        Zero-based estimate column index (same as ``u_hat[:, window_idx]`` after
        :func:`~openmhe.run_solver`). Window 0 is the first full horizon;
        window 500 covers measurement times ``[N+500-N, …, N+500-1]``.

    Returns
    -------
    dict
        Same fields as :func:`lti_ls_decision_hessian`, plus ``window_idx``,
        ``t_start``, ``k``, and ``W0``.
    """
    from openmhe.builder.solver import run_solver

    n_steps = y.shape[1] if y.ndim > 1 else len(y)
    N = int(solver._N)
    n_est = n_steps - N
    if window_idx < 0 or window_idx >= n_est:
        raise ValueError(
            f"window_idx must be in [0, {n_est - 1}] for {n_steps} samples and N={N}."
        )

    run_solver(solver, y, u, stop_at_idx=window_idx, require_success=True)
    W0 = np.asarray(solver.cost_get(0, "W"), dtype=float)
    out = lti_ls_decision_hessian(solver, include_x0=include_x0, W0=W0)
    out["window_idx"] = int(window_idx)
    out["t_start"] = int(N + window_idx - N)
    out["k"] = int(N + window_idx)
    return out


def _noise_channel_targets(G_proc: np.ndarray, nx_base: int) -> list[tuple[str, int]]:
    """Map each column of ``G_proc`` to ``('plant', row)`` or ``('rw', input_idx)``."""
    G = np.asarray(G_proc, dtype=float)
    targets: list[tuple[str, int]] = []
    for j in range(G.shape[1]):
        rows = np.flatnonzero(G[:, j])
        if rows.size != 1:
            raise ValueError(f"Expected one injection row for noise channel {j}.")
        row = int(rows[0])
        if row < nx_base:
            targets.append(("plant", row))
        else:
            targets.append(("rw", row - nx_base))
    return targets


def decision_variable_labels(
    solver,
    *,
    include_x0: bool = True,
    plant_state_name: PlantStateNamer | None = None,
    input_name: InputNamer | None = None,
) -> list[dict[str, Any]]:
    """Human-readable names for each row/column of ``lti_ls_decision_hessian``'s ``H``.

    Layout matches :func:`lti_ls_decision_hessian`:

    ``[u_0, u_1, …, u_{N-1}, x_0]`` with ``u_k = [u_ctrl_k; w_k]``.

    Returns one dict per decision variable with keys ``index``, ``name``,
    ``group`` (``process_noise`` | ``ocp_control`` | ``initial_state``),
    ``stage``, and ``kind`` (``plant_noise`` | ``rw_noise`` | ``control`` |
    ``plant_state`` | ``rw_state``).
    """
    if not getattr(solver, "_linear_ls", False):
        raise ValueError(
            "decision_variable_labels requires LINEAR_LS (all L2) problems."
        )

    N = int(solver._N)
    nx = int(solver._nx)
    nx_base = int(solver._nx_base)
    nu_err = int(solver._B_aug.shape[1])
    nu_ctrl = int(getattr(solver, "_nu_ctrl", nu_err))
    nw = int(getattr(solver, "_nw", nu_err - nu_ctrl))
    rw_indices: Sequence[int] = getattr(solver, "_rw_indices", ())
    G_proc = np.asarray(getattr(solver, "_G_proc"), dtype=float)
    arrival_idx = getattr(solver, "_arrival_state_idx", None)

    if plant_state_name is None:
        plant_state_name = lambda i: f"plant.x[{i}]"
    if input_name is None:
        input_name = lambda ui: f"input[{ui}]"

    noise_targets = _noise_channel_targets(G_proc, nx_base)
    labels: list[dict[str, Any]] = []
    idx = 0

    for stage in range(N):
        for ci in range(nu_ctrl):
            ui = int(solver._controlled_idx[ci])
            labels.append(
                {
                    "index": idx,
                    "name": f"u_ctrl[k={stage}, {input_name(ui)}]",
                    "group": "ocp_control",
                    "stage": stage,
                    "kind": "control",
                    "input_index": ui,
                    "control_index": ci,
                }
            )
            idx += 1

        for wj, (target_kind, target_row) in enumerate(noise_targets):
            if target_kind == "plant":
                inj = plant_state_name(target_row)
                kind = "plant_noise"
                name = f"w[k={stage}, {inj}]"
                extra = {"plant_index": target_row}
            else:
                ui = int(rw_indices[target_row])
                kind = "rw_noise"
                name = f"w[k={stage}, d({input_name(ui)})]"
                extra = {"input_index": ui, "rw_slot": target_row}
            labels.append(
                {
                    "index": idx,
                    "name": name,
                    "group": "process_noise",
                    "stage": stage,
                    "kind": kind,
                    "noise_channel": wj,
                    **extra,
                }
            )
            idx += 1

    if include_x0:
        arrival_set = (
            None if arrival_idx is None else set(int(i) for i in arrival_idx)
        )
        for i in range(nx):
            if i < nx_base:
                kind = "plant_state"
                name = f"x0[{plant_state_name(i)}]"
                extra = {
                    "plant_index": i,
                    "has_arrival": arrival_set is None or i in arrival_set,
                }
            else:
                ui = int(rw_indices[i - nx_base])
                kind = "rw_state"
                name = f"x0[{input_name(ui)}]"
                extra = {"input_index": ui, "has_arrival": False}
            labels.append(
                {
                    "index": idx,
                    "name": name,
                    "group": "initial_state",
                    "stage": 0,
                    "kind": kind,
                    **extra,
                }
            )
            idx += 1

    expected = N * nu_err + (nx if include_x0 else 0)
    if idx != expected:
        raise RuntimeError(f"Label count {idx} != expected Hessian dimension {expected}.")
    return labels


def labels_to_names(labels: Sequence[dict[str, Any]]) -> list[str]:
    """Return ordered ``name`` strings from :func:`decision_variable_labels`."""
    return [str(entry["name"]) for entry in labels]
