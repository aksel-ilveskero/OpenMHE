"""Acados LINEAR_LS vs CONVEX_OVER_NONLINEAR cost assembly for MHE."""

from __future__ import annotations

import casadi as ca
import numpy as np
from acados_template import AcadosOcp

from openmhe.builder.input_regs import cost_terms
from openmhe.mhe_strategies import ObjectiveBuilder
from openmhe.mhe_strategies.penalties import BasePenalty, L2Penalty, _quadratic_form


def _term_penalty(term) -> BasePenalty:
    """Return the outer penalty for a cost term (default L2 if omitted)."""
    if not hasattr(term, "penalty"):
        return L2Penalty()
    penalty = term.penalty
    if not isinstance(penalty, BasePenalty):
        raise TypeError(
            f"Term {type(term).__name__} must use a BasePenalty, got {type(penalty)}."
        )
    return penalty


def needs_conl(builder: ObjectiveBuilder) -> bool:
    """True if any term requires CONVEX_OVER_NONLINEAR (non-L2 outer penalty)."""
    return any(
        _term_penalty(term).requires_nonlinear_cost for term in cost_terms(builder)
    )


def validate_term_penalties(builder: ObjectiveBuilder) -> None:
    """Ensure every stage-cost term uses a :class:`~openmhe.BasePenalty` instance."""
    for term in cost_terms(builder):
        _term_penalty(term)


def stack_weights(builder: ObjectiveBuilder) -> np.ndarray:
    """Block-diagonal stack of per-term inverse-covariance matrices."""
    blocks = [term.weight.W for term in cost_terms(builder)]
    if not blocks:
        raise ValueError("ObjectiveBuilder must contain at least one term.")
    n = sum(b.shape[0] for b in blocks)
    W = np.zeros((n, n))
    i = 0
    for block in blocks:
        s = block.shape[0]
        W[i : i + s, i : i + s] = block
        i += s
    return W


def build_psi_expr(builder: ObjectiveBuilder, r: ca.SX) -> ca.SX:
    """Sum per-term outer penalties on residual slice ``r``."""
    expr = ca.SX(0)
    row = 0
    for term in cost_terms(builder):
        d = term.weight.dim
        expr += _term_penalty(term).psi_contribution(r[row : row + d], term.weight.W)
        row += d
    return expr


def cost_mode_tag(builder: ObjectiveBuilder) -> str:
    """Return ``'ls'`` or ``'conl'`` for Acados JSON naming and cost assembly."""
    return "conl" if needs_conl(builder) else "ls"


def build_linear_ls_cost(
    ocp: AcadosOcp,
    Vx: np.ndarray,
    Vu: np.ndarray,
    builder: ObjectiveBuilder,
    n_residual: int,
    nx: int,
    nx_base: int,
    P_arrival: np.ndarray | None,
) -> tuple[slice | None, int, np.ndarray | None]:
    """Configure LINEAR_LS path and terminal cost. Returns arrival_slice, n_residual_0, W_0."""
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    ocp.cost.W = stack_weights(builder)
    ocp.cost.yref = np.zeros(n_residual)

    if P_arrival is not None:
        ny0 = n_residual + nx_base
        Vx_arrival = np.zeros((nx_base, nx))
        Vx_arrival[:, :nx_base] = np.eye(nx_base)
        ocp.cost.cost_type_0 = "LINEAR_LS"
        ocp.cost.Vx_0 = np.vstack([Vx, Vx_arrival])
        ocp.cost.Vu_0 = np.vstack([Vu, np.zeros((nx_base, Vu.shape[1]))])
        W0 = np.zeros((ny0, ny0))
        W0[:n_residual, :n_residual] = ocp.cost.W
        W0[n_residual:, n_residual:] = np.linalg.inv(P_arrival)
        ocp.cost.W_0 = W0
        ocp.cost.yref_0 = np.zeros(ny0)
        arrival_slice = slice(n_residual, ny0)
        n_residual_0 = ny0
    else:
        ocp.cost.cost_type_0 = "LINEAR_LS"
        ocp.cost.Vx_0 = Vx
        ocp.cost.Vu_0 = Vu
        ocp.cost.W_0 = ocp.cost.W
        ocp.cost.yref_0 = np.zeros(n_residual)
        arrival_slice = None
        n_residual_0 = n_residual

    configure_negligible_terminal_cost(ocp, nx)
    W0 = np.array(ocp.cost.W_0, dtype=float).copy() if P_arrival is not None else None
    return arrival_slice, n_residual_0, W0


def build_conl_cost(
    ocp: AcadosOcp,
    model,
    Vx: np.ndarray,
    Vu: np.ndarray,
    builder: ObjectiveBuilder,
    n_residual: int,
    nx_base: int,
    P_arrival: np.ndarray | None,
) -> tuple[slice | None, int, np.ndarray | None]:
    """Configure CONVEX_OVER_NONLINEAR path and terminal cost."""
    y_expr = ca.DM(Vx) @ model.x + ca.DM(Vu) @ model.u
    model.cost_y_expr = y_expr

    r = ca.SX.sym("r_mhe", n_residual)
    model.cost_r_in_psi_expr = r
    model.cost_psi_expr = build_psi_expr(builder, r)

    ocp.cost.cost_type = "CONVEX_OVER_NONLINEAR"
    ocp.cost.yref = np.zeros(n_residual)

    if P_arrival is not None:
        ny0 = n_residual + nx_base
        r0 = ca.SX.sym("r0_mhe", ny0)
        model.cost_y_expr_0 = ca.vertcat(y_expr, model.x[:nx_base])
        model.cost_r_in_psi_expr_0 = r0
        W_arr = np.linalg.inv(P_arrival)
        psi_arr = _quadratic_form(r0[n_residual:], W_arr)
        model.cost_psi_expr_0 = build_psi_expr(builder, r0[:n_residual]) + psi_arr
        ocp.cost.cost_type_0 = "CONVEX_OVER_NONLINEAR"
        ocp.cost.yref_0 = np.zeros(ny0)
        arrival_slice = slice(n_residual, ny0)
        n_residual_0 = ny0
    else:
        model.cost_y_expr_0 = y_expr
        model.cost_r_in_psi_expr_0 = r
        model.cost_psi_expr_0 = model.cost_psi_expr
        ocp.cost.cost_type_0 = "CONVEX_OVER_NONLINEAR"
        ocp.cost.yref_0 = np.zeros(n_residual)
        arrival_slice = None
        n_residual_0 = n_residual

    configure_negligible_terminal_cost(ocp, model.x.shape[0])
    W0 = None
    if P_arrival is not None and ocp.cost.cost_type_0 == "LINEAR_LS":
        W0 = np.array(ocp.cost.W_0, dtype=float).copy()
    return arrival_slice, n_residual_0, W0


def configure_negligible_terminal_cost(ocp: AcadosOcp, nx: int) -> None:
    """Avoid pinning augmented states at the terminal node."""
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.Vx_e = np.zeros((1, nx))
    ocp.cost.W_e = np.eye(1) * 1e-12
    ocp.cost.yref_e = np.zeros(1)
