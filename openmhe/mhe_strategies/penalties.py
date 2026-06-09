"""MHE objective terms, noise weights, and cost penalties."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import casadi as ca
import numpy as np

_DEADZONE_MODES = ("symmetric", "one_sided_lower", "one_sided_upper")


def _as_diag_weights(W_block: np.ndarray) -> np.ndarray:
    """Extract diagonal weights; error if a full matrix has off-diagonal entries."""
    W = np.asarray(W_block, dtype=float)
    if W.ndim == 2:
        if W.shape[0] != W.shape[1]:
            raise ValueError("Weight block must be square.")
        off = np.max(np.abs(W - np.diag(np.diag(W))))
        if off > 1e-12:
            raise ValueError(
                "Nonlinear penalties require diagonal NoiseWeight blocks (no coupling)."
            )
        return np.diag(W)
    return W.ravel()


class NoiseWeight:
    """Format user noise specs into an Acados weight matrix ``W``."""

    def __init__(
        self,
        dim: int,
        std: Optional[Union[float, List[float], np.ndarray]] = None,
        cov: Optional[Union[float, List[float], np.ndarray]] = None,
        inv_cov: Optional[Union[List[float], np.ndarray]] = None,
    ):
        """Exactly one of ``std``, ``cov``, or ``inv_cov`` must be supplied."""
        self.dim = dim
        provided = [arg for arg in [std, cov, inv_cov] if arg is not None]
        if len(provided) != 1:
            raise ValueError(
                f"You must provide exactly one noise metric (std, cov, or inv_cov). "
                f"Received {len(provided)}."
            )
        raw_val = provided[0]
        if isinstance(raw_val, (int, float)):
            val = np.ones(dim) * float(raw_val)
        else:
            val = np.array(raw_val, dtype=float)
        self.W = self._convert_to_weight_matrix(val, std, cov, inv_cov)

    def _convert_to_weight_matrix(self, val, std, cov, inv_cov) -> np.ndarray:
        """Map the chosen noise metric to a diagonal or dense inverse-covariance ``W``."""
        epsilon = 1e-8
        max_weight = 1e8

        if val.ndim == 1:
            if len(val) != self.dim:
                raise ValueError(
                    f"Expected noise array of length {self.dim}, got {len(val)}."
                )
            if std is not None:
                w = np.zeros(len(val))
                pos = val > 0
                if np.any(pos):
                    w[pos] = 1.0 / (val[pos] ** 2)
                return np.diag(w)
            if cov is not None:
                w = np.zeros(len(val))
                pos = val > 0
                if np.any(pos):
                    w[pos] = 1.0 / val[pos]
                return np.diag(w)
            if inv_cov is not None:
                w = np.zeros(len(val))
                pos = val > 0
                if np.any(pos):
                    w[pos] = np.minimum(val[pos], max_weight)
                return np.diag(w)

        if val.ndim == 2:
            if val.shape != (self.dim, self.dim):
                raise ValueError(
                    f"Expected {self.dim}x{self.dim} matrix, got {val.shape}."
                )
            if std is not None:
                raise ValueError(
                    "Cannot process a 2D matrix for 'std'. Use 'cov' or 'inv_cov'."
                )
            if cov is not None:
                diag = np.diag(val)
                if np.any(diag <= 0):
                    if not np.allclose(val - np.diag(diag), 0):
                        raise ValueError(
                            "Strict kinematics (zero process cov entries) requires "
                            "a diagonal cov matrix."
                        )
                    w = np.zeros(self.dim)
                    pos = diag > 0
                    if np.any(pos):
                        w[pos] = 1.0 / diag[pos]
                    return np.diag(w)
                safe_cov = val + (np.eye(self.dim) * epsilon)
                return np.linalg.inv(safe_cov)
            if inv_cov is not None:
                return np.clip(val, -max_weight, max_weight)

        raise ValueError("Noise input must be a float, 1D list, or 2D array.")


class BasePenalty(ABC):
    """Outer penalty ``psi(r)`` on a linear residual slice ``r`` (CONL) or L2 via ``W``."""

    @property
    @abstractmethod
    def requires_nonlinear_cost(self) -> bool:
        """If True, :func:`openmhe.build_mhe_solver` uses CONVEX_OVER_NONLINEAR."""

    @abstractmethod
    def psi_contribution(self, r: ca.SX, W_block: np.ndarray) -> ca.SX:
        """Scalar CasADi cost for one term's residual block."""


def _quadratic_form(r: ca.SX, W_block: np.ndarray) -> ca.SX:
    """Scalar ``0.5 * r.T @ W @ r`` (CasADi; avoids deprecated/missing ``bilinear``)."""
    W = ca.DM(np.asarray(W_block, dtype=float))
    return 0.5 * ca.dot(r, ca.mtimes(W, r))


def _huber_scalar(ri: ca.SX, delta: float) -> ca.SX:
    """Scalar Huber loss for one residual component."""
    return ca.if_else(
        ca.fabs(ri) <= delta,
        0.5 * ri**2,
        delta * (ca.fabs(ri) - 0.5 * delta),
    )


class L2Penalty(BasePenalty):
    """Quadratic (least-squares) penalty; enables Acados ``LINEAR_LS`` when used on all terms."""

    @property
    def requires_nonlinear_cost(self) -> bool:
        """L2 enables pure ``LINEAR_LS`` when all terms use this penalty."""
        return False

    def psi_contribution(self, r: ca.SX, W_block: np.ndarray) -> ca.SX:
        """Return ``0.5 * r.T @ W @ r``."""
        return _quadratic_form(r, W_block)


class L1Penalty(BasePenalty):
    """L1 (absolute value) penalty on each residual component."""

    def __init__(self, epsilon: float = 0.0):
        """
        Parameters
        ----------
        epsilon : float
            If ``> 0``, use smoothed ``sqrt(r_i^2 + epsilon^2)`` instead of ``|r_i|``.
        """
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        self.epsilon = float(epsilon)

    @property
    def requires_nonlinear_cost(self) -> bool:
        """L1 always uses CONVEX_OVER_NONLINEAR."""
        return True

    def psi_contribution(self, r: ca.SX, W_block: np.ndarray) -> ca.SX:
        """Weighted sum of absolute (or smoothed) residual components."""
        w = _as_diag_weights(W_block)
        if self.epsilon > 0:
            eps2 = self.epsilon**2
            terms = [
                w[i] * ca.sqrt(r[i] ** 2 + eps2) for i in range(int(r.shape[0]))
            ]
        else:
            terms = [w[i] * ca.fabs(r[i]) for i in range(int(r.shape[0]))]
        return ca.sum1(ca.vertcat(*terms)) if terms else 0


class HuberPenalty(BasePenalty):
    """Element-wise Huber penalty (quadratic near zero, linear tails)."""

    def __init__(self, delta: float = 1.0, min_hess: float = 0.0):
        """Set Huber knee ``delta`` (quadratic inside, linear outside)."""
        if delta <= 0:
            raise ValueError("delta must be positive.")
        self.delta = float(delta)
        self.min_hess = float(min_hess)

    @property
    def requires_nonlinear_cost(self) -> bool:
        """Huber always uses CONVEX_OVER_NONLINEAR."""
        return True

    def psi_contribution(self, r: ca.SX, W_block: np.ndarray) -> ca.SX:
        """Element-wise Huber loss weighted by diagonal ``W``."""
        w = _as_diag_weights(W_block)
        terms = [w[i] * _huber_scalar(r[i], self.delta) for i in range(int(r.shape[0]))]
        return ca.sum1(ca.vertcat(*terms)) if terms else 0


class DeadzonePenalty(BasePenalty):
    """Dead-zone penalty: no cost inside the zone, quadratic outside.

    Parameters
    ----------
    zone : float or array
        Dead-zone width per residual component.
    mode : str
        ``"symmetric"`` — no cost for ``|r| <= zone``;
        ``"one_sided_lower"`` — no cost for ``r >= -zone``, quadratic below;
        ``"one_sided_upper"`` — no cost for ``r <= zone``, quadratic above.
    """

    def __init__(
        self,
        zone: Union[float, List[float], np.ndarray] = 0.0,
        mode: str = "symmetric",
    ):
        """Set dead-zone width ``zone`` and ``mode`` (symmetric or one-sided)."""
        mode = str(mode).lower()
        if mode not in _DEADZONE_MODES:
            raise ValueError(f"mode must be one of {_DEADZONE_MODES}, got {mode!r}.")
        self.mode = mode
        self.zone = zone

    @property
    def requires_nonlinear_cost(self) -> bool:
        """Dead-zone always uses CONVEX_OVER_NONLINEAR."""
        return True

    def _zone_vec(self, dim: int) -> np.ndarray:
        """Per-component dead-zone width as a length-``dim`` vector."""
        z = self.zone
        if isinstance(z, (int, float)):
            return np.ones(dim) * float(z)
        arr = np.asarray(z, dtype=float).ravel()
        if len(arr) != dim:
            raise ValueError(f"zone length {len(arr)} != residual dim {dim}.")
        return arr

    def psi_contribution(self, r: ca.SX, W_block: np.ndarray) -> ca.SX:
        """Quadratic penalty on residual magnitude outside the dead zone."""
        w = _as_diag_weights(W_block)
        dim = int(r.shape[0])
        zones = self._zone_vec(dim)
        terms = []
        for i in range(dim):
            ri, zi, wi = r[i], zones[i], w[i]
            if self.mode == "symmetric":
                excess = ca.fmax(0, ca.fabs(ri) - zi)
                terms.append(0.5 * wi * excess**2)
            elif self.mode == "one_sided_lower":
                terms.append(
                    ca.if_else(ri < -zi, 0.5 * wi * (ri + zi) ** 2, 0)
                )
            else:  # one_sided_upper
                terms.append(
                    ca.if_else(ri > zi, 0.5 * wi * (ri - zi) ** 2, 0)
                )
        return ca.sum1(ca.vertcat(*terms)) if terms else 0


class ObjectiveBuilder:
    """Ordered collection of MHE cost terms passed to :func:`openmhe.build_mhe_solver`."""

    def __init__(self):
        """Start with an empty list of cost terms."""
        self.terms = []
        self.arrival_cost = None

    def add(self, term):
        """Append a cost term or arrival-cost strategy.

        Pass a :class:`~openmhe.BaseArrivalCost` subclass
        (:class:`~openmhe.SteadyStateArrivalCost`, :class:`~openmhe.EKFArrivalCost`,
        :class:`~openmhe.UKFArrivalCost`) to attach the window arrival cost.
        At most one arrival cost is allowed per objective.
        """
        from openmhe.mhe_strategies.arrival_cost import BaseArrivalCost

        if isinstance(term, BaseArrivalCost):
            if self.arrival_cost is not None:
                raise ValueError("ObjectiveBuilder accepts at most one arrival cost.")
            self.arrival_cost = term
            return
        self.terms.append(term)

    def to_latex(self, **kwargs) -> str:
        """Render this objective as a LaTeX MHE problem.

        Accepts the same keyword arguments as
        :func:`openmhe.export.latex.objective_to_latex` (e.g. ``underbrace``,
        ``symbols``, and ``form="substituted"`` | ``"constrained"`` for the
        ``minimize ... subject to`` layout). When ``self.arrival_cost`` is set,
        the matching arrival term is included automatically.
        """
        from openmhe.export.latex import objective_to_latex

        return objective_to_latex(self, **kwargs)


class MeasurementTerm:
    """Weighted fit of ``y`` to ``C x + D u``."""

    def __init__(self, penalty: BasePenalty, weight: NoiseWeight):
        """Combine outer ``penalty`` with measurement inverse-covariance ``weight``."""
        if not isinstance(penalty, BasePenalty):
            raise TypeError("penalty must be a BasePenalty instance.")
        self.penalty = penalty
        self.weight = weight
        self.target_type = "MEASUREMENT"


class ProcessTerm:
    """Penalty on process noise ``w`` in ``x_{k+1} = A x_k + B u_k + w_k``."""

    def __init__(self, penalty: BasePenalty, weight: NoiseWeight):
        """Combine outer ``penalty`` with process-noise inverse-covariance ``weight``."""
        if not isinstance(penalty, BasePenalty):
            raise TypeError("penalty must be a BasePenalty instance.")
        self.penalty = penalty
        self.weight = weight
        self.target_type = "PROCESS"


class InputTrackingTerm:
    """Penalize deviation of selected inputs from a reference."""

    def __init__(
        self,
        target_idx: list,
        weight: NoiseWeight,
        reference: str | float = "measured",
        penalty: BasePenalty | None = None,
    ):
        """Track inputs in ``target_idx`` against ``reference`` (``"measured"``, ``"zero"``, or scalar)."""
        self.target_idx = list(target_idx)
        self.weight = weight
        self.reference = reference
        self.penalty = penalty if penalty is not None else L2Penalty()
        if not isinstance(self.penalty, BasePenalty):
            raise TypeError("penalty must be a BasePenalty instance.")
        self.target_type = "INPUT_TRACKING"


class KnownInput:
    """Pin selected inputs to values passed in :func:`~openmhe.run_solver` (``u``).

    No extra cost row — the solver fixes these control channels via equality
    bounds. Use instead of :class:`InputTrackingTerm` when the input is known
    exactly and should not appear in the objective.
    """

    def __init__(self, target_idx: list):
        """Declare ``target_idx`` as known inputs fixed in :func:`~openmhe.run_solver`."""
        self.target_idx = list(target_idx)
        self.target_type = "KNOWN_INPUT"


def weight_from_lambda_u(
    dim: int, lambda_u: Union[float, List[float], np.ndarray]
) -> NoiseWeight:
    """Build diagonal inverse-covariance weights from regulator strength ``lambda_u``.

    Larger ``lambda_u`` implies stronger penalty on the regulated residual.
    """
    return NoiseWeight(dim=dim, inv_cov=lambda_u)


def _lambda_u_per_index(
    target_idx: list, lambda_u: Union[float, List[float], np.ndarray]
) -> np.ndarray:
    """Broadcast scalar ``lambda_u`` to one value per ``target_idx`` entry."""
    n = len(target_idx)
    if isinstance(lambda_u, (int, float)):
        return np.ones(n) * float(lambda_u)
    arr = np.asarray(lambda_u, dtype=float).ravel()
    if len(arr) != n:
        raise ValueError(
            f"lambda_u length {len(arr)} must match target_idx length {n}."
        )
    return arr


class InputFirstDiffReg:
    """Penalize first difference ``u_k - u_{k-1}`` via one delay state per input."""

    def __init__(
        self,
        target_idx: list,
        lambda_u: Union[float, List[float], np.ndarray],
        penalty: BasePenalty | None = None,
    ):
        """Penalize first difference on ``target_idx`` with strength ``lambda_u``."""
        self.target_idx = list(target_idx)
        self.lambda_u = lambda_u
        self.penalty = penalty if penalty is not None else L2Penalty()
        if not isinstance(self.penalty, BasePenalty):
            raise TypeError("penalty must be a BasePenalty instance.")
        self.target_type = "INPUT_REG"
        self.trend = "FIRST_DIFF"
        self.weight = weight_from_lambda_u(len(self.target_idx), lambda_u)


class InputSecondDiffReg:
    """Penalize second difference ``u_k - 2 u_{k-1} + u_{k-2}`` (two delay states)."""

    def __init__(
        self,
        target_idx: list,
        lambda_u: Union[float, List[float], np.ndarray],
        penalty: BasePenalty | None = None,
    ):
        """Penalize second difference on ``target_idx`` with strength ``lambda_u``."""
        self.target_idx = list(target_idx)
        self.lambda_u = lambda_u
        self.penalty = penalty if penalty is not None else L2Penalty()
        if not isinstance(self.penalty, BasePenalty):
            raise TypeError("penalty must be a BasePenalty instance.")
        self.target_type = "INPUT_REG"
        self.trend = "SECOND_DIFF"
        self.weight = weight_from_lambda_u(len(self.target_idx), lambda_u)


class InputRandomWalk:
    """Model selected inputs as random-walk states: ``u_{k+1} = u_k + w_u``.

    Contributes process noise weighted by ``lambda_u`` (inverse covariance).
    No extra measurement residual row — use with :class:`ProcessTerm` on plant states.
    """

    def __init__(
        self,
        target_idx: list,
        lambda_u: Union[float, List[float], np.ndarray],
    ):
        """Augment state with random-walk dynamics for inputs in ``target_idx``."""
        self.target_idx = list(target_idx)
        self.lambda_u = lambda_u
        self.target_type = "INPUT_RANDOM_WALK"
        self.lambdas = _lambda_u_per_index(self.target_idx, lambda_u)


class InputRegTerm:
    """Deprecated: use :class:`InputFirstDiffReg` or :class:`InputSecondDiffReg`."""

    def __init__(
        self,
        target_idx: list,
        penalty: BasePenalty,
        trend: str,
        weight=None,
        lambda_u: Union[float, List[float], np.ndarray, None] = None,
        live_param_name: str | None = None,
    ):
        """Legacy regulator term; prefer :class:`InputFirstDiffReg` / :class:`InputSecondDiffReg`."""
        warnings.warn(
            "InputRegTerm is deprecated; use InputFirstDiffReg, InputSecondDiffReg, "
            "or InputRandomWalk.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(penalty, BasePenalty):
            raise TypeError("penalty must be a BasePenalty instance.")
        self.target_idx = target_idx
        self.penalty = penalty
        self.trend = trend
        self.target_type = "INPUT_REG"
        self.is_live = live_param_name is not None
        self.live_param_name = live_param_name
        if weight is not None:
            self.weight = weight
        elif lambda_u is not None:
            self.weight = weight_from_lambda_u(len(target_idx), lambda_u)
        else:
            raise ValueError(
                "InputRegTerm requires weight or lambda_u when live_param_name is not set."
            )
        self.static_weight = self.weight
