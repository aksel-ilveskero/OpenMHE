"""Arrival-cost strategies for sliding-window MHE."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import scipy.linalg as sla

from openmhe.frontend.system import SystemModel
from openmhe.mhe_strategies.penalties import ObjectiveBuilder


def _term_kind(term) -> str:
    return getattr(term, "target_type", getattr(term, "type", ""))


def invert_arrival_covariance(
    P: np.ndarray,
    *,
    tol: float = 1e-8,
    max_weight: float = 1e6,
) -> np.ndarray:
    """Arrival weight ``P^{-1}`` with null-space directions zeroed and capped.

    Tiny or zero eigenvalues (strict kinematics / unobservable directions) receive
    no arrival penalty.  Large weights are clipped to keep the QP well scaled.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        P = np.diag(P)
    if P.shape[0] != P.shape[1]:
        raise ValueError(f"P_arrival must be square, got {P.shape}.")
    if P.shape[0] == 0:
        return np.zeros((0, 0))

    evals, evecs = sla.eigh(P)
    lam_max = max(float(np.max(evals)), 1.0)
    thresh = tol * lam_max
    w_evals = np.zeros_like(evals)
    pos = evals > thresh
    w_evals[pos] = np.minimum(1.0 / evals[pos], max_weight)
    return evecs @ np.diag(w_evals) @ evecs.T


def arrival_covariance_submatrix(
    P: np.ndarray,
    arrival_state_idx: np.ndarray | None,
) -> np.ndarray:
    """Extract the plant arrival block used in stage-0 cost."""
    P = np.asarray(P, dtype=float)
    if arrival_state_idx is None:
        return P
    idx = np.asarray(arrival_state_idx, dtype=int)
    return P[np.ix_(idx, idx)]


def _weight_to_cov(W: np.ndarray, name: str) -> np.ndarray:
    """Invert a diagonal or dense inverse-covariance weight matrix."""
    W = np.asarray(W, dtype=float)
    if W.ndim == 1:
        W = np.diag(W)
    off = W - np.diag(np.diag(W))
    if np.any(np.abs(off) > 1e-12):
        raise ValueError(
            f"{name} weight must be diagonal for arrival-cost filters."
        )
    diag = np.diag(W)
    if np.any(diag <= 0):
        raise ValueError(f"{name} inverse covariance must be positive.")
    return np.diag(1.0 / diag)


def _plant_Q_from_process(proc, nx: int) -> np.ndarray:
    """Return ``nx x nx`` plant process covariance ``Q`` (supports sparse kinematics)."""
    G = getattr(proc, "_G_proc", None)
    if G is not None:
        from openmhe.builder.input_regs import plant_process_cov

        return plant_process_cov(G, proc.weight.W, nx)

    W_plant = np.asarray(proc.weight.W, dtype=float)
    if W_plant.shape != (nx, nx):
        W_plant = W_plant[:nx, :nx]

    diag = np.diag(W_plant)
    off = W_plant - np.diag(diag)
    if np.any(np.abs(off) > 1e-12):
        return _weight_to_cov(W_plant, "Process")
    if np.all(diag > 0):
        return _weight_to_cov(W_plant, "Process")
    if not np.any(diag > 0):
        raise ValueError(
            "ProcessTerm requires at least one nonzero plant process-noise weight."
        )

    active = np.flatnonzero(diag > 0)
    W_sparse = W_plant[np.ix_(active, active)]
    G_plant = np.zeros((nx, active.size))
    for j, idx in enumerate(active):
        G_plant[idx, j] = 1.0
    from openmhe.builder.input_regs import plant_process_cov

    return plant_process_cov(G_plant, W_sparse, nx)


def noise_covs_from_builder(
    builder: ObjectiveBuilder, nx: int, ny: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Q, R)`` from the single process and measurement terms."""
    proc_terms = [t for t in builder.terms if _term_kind(t) == "PROCESS"]
    meas_terms = [t for t in builder.terms if _term_kind(t) == "MEASUREMENT"]
    if len(proc_terms) != 1:
        raise ValueError("Objective must contain exactly one ProcessTerm.")
    if len(meas_terms) != 1:
        raise ValueError("Objective must contain exactly one MeasurementTerm.")

    Q = _plant_Q_from_process(proc_terms[0], nx)
    R = _weight_to_cov(meas_terms[0].weight.W, "Measurement")
    return Q, R


def steady_state_cov(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Steady-state Kalman covariance from the discrete algebraic Riccati equation."""
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    return sla.solve_discrete_are(A.T, C.T, Q, R)


class BaseArrivalCost(ABC):
    """Arrival cost at the first stage of each MHE window."""

    @property
    @abstractmethod
    def is_dynamic(self) -> bool:
        """Whether ``P`` (and possibly ``x_bar``) changes each window."""

    @abstractmethod
    def initial_covariance(
        self, system: SystemModel, builder: ObjectiveBuilder
    ) -> np.ndarray:
        """Covariance matrix ``P`` used when configuring the Acados stage-0 cost."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state before a new :func:`~openmhe.run_solver` pass."""

    @abstractmethod
    def window_prior(
        self,
        t_start: int,
        y: np.ndarray,
        u: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x_bar, P)`` for base states at time ``t_start`` (before ``y[:, t_start]``)."""

    def latex_parts(self, *, state: str, window_lo: str) -> tuple[str, str, str]:
        """Return ``(residual, weight, description)`` for LaTeX export."""
        return (
            f"{state}_{{{window_lo}}} - \\bar{{{state}}}_{{{window_lo}}}",
            "P^{-1}",
            "arrival cost",
        )


class SteadyStateArrivalCost(BaseArrivalCost):
    """Fixed DARE/steady-state Kalman covariance; zero mean reference."""

    def __init__(self, P: np.ndarray | None = None):
        self._P: np.ndarray | None = None if P is None else np.asarray(P, dtype=float)

    @property
    def is_dynamic(self) -> bool:
        return False

    def initial_covariance(
        self, system: SystemModel, builder: ObjectiveBuilder
    ) -> np.ndarray:
        if self._P is not None:
            return self._P.copy()
        Q, R = noise_covs_from_builder(builder, system.nx, system.ny)
        self._P = steady_state_cov(system.A, system.B, system.C, system.D, Q, R)
        return self._P.copy()

    def reset(self) -> None:
        return

    def window_prior(
        self,
        t_start: int,
        y: np.ndarray,
        u: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._P is None:
            raise RuntimeError(
                "SteadyStateArrivalCost.window_prior called before initial_covariance."
            )
        x_bar = np.zeros(self._P.shape[0])
        return x_bar, self._P.copy()

    def latex_parts(self, *, state: str, window_lo: str) -> tuple[str, str, str]:
        return (
            f"{state}_{{{window_lo}}}",
            "P^{-1}",
            "steady-state arrival cost",
        )


class _LinearFilterArrivalCost(BaseArrivalCost):
    """Shared filter bookkeeping for EKF and UKF arrival costs."""

    def __init__(
        self,
        system: SystemModel,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ):
        self.A = np.asarray(system.A, dtype=float)
        self.B = np.asarray(system.B, dtype=float)
        self.C = np.asarray(system.C, dtype=float)
        self.D = (
            np.asarray(system.D, dtype=float)
            if system.D is not None
            else np.zeros((system.ny, system.nu))
        )
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        nx = self.A.shape[0]
        self._x0 = np.zeros(nx) if x0 is None else np.asarray(x0, dtype=float).ravel()
        if P0 is None:
            self._P0 = np.diag(np.diag(self.Q))
        else:
            self._P0 = np.asarray(P0, dtype=float)
            if self._P0.ndim == 1:
                self._P0 = np.diag(self._P0)
        self._posterior_t = -1
        self._x = self._x0.copy()
        self._P = self._P0.copy()

    @property
    def is_dynamic(self) -> bool:
        return True

    def reset(self) -> None:
        self._posterior_t = -1
        self._x = self._x0.copy()
        self._P = self._P0.copy()

    def _advance_posterior_to(
        self, t_end: int, y: np.ndarray, u: np.ndarray
    ) -> None:
        while self._posterior_t < t_end:
            t = self._posterior_t + 1
            self._predict_inplace(u[:, t])
            self._update_inplace(y[:, t], u[:, t])
            self._posterior_t = t

    def window_prior(
        self,
        t_start: int,
        y: np.ndarray,
        u: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._advance_posterior_to(t_start - 1, y, u)
        return self._predict_copy(u[:, t_start])

    @abstractmethod
    def _predict_inplace(self, u_t: np.ndarray) -> None:
        """Advance filter mean/covariance to the prior at the current step."""

    @abstractmethod
    def _update_inplace(self, y_t: np.ndarray, u_t: np.ndarray) -> None:
        """Assimilate a measurement at the current step."""

    @abstractmethod
    def _predict_copy(self, u_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return prior at the current step without mutating filter state."""


class EKFArrivalCost(_LinearFilterArrivalCost):
    """Discrete extended Kalman filter arrival cost on the base LTI model."""

    def __init__(
        self,
        system: SystemModel,
        builder: ObjectiveBuilder | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ):
        if Q is None or R is None:
            if builder is None:
                raise ValueError(
                    "EKFArrivalCost requires builder or explicit Q and R."
                )
            Q_b, R_b = noise_covs_from_builder(builder, system.nx, system.ny)
            Q = Q if Q is not None else Q_b
            R = R if R is not None else R_b
        super().__init__(system, Q, R, x0=x0, P0=P0)
        self._I = np.eye(self.A.shape[0])

    def initial_covariance(
        self, system: SystemModel, builder: ObjectiveBuilder
    ) -> np.ndarray:
        return self._P0.copy()

    def latex_parts(self, *, state: str, window_lo: str) -> tuple[str, str, str]:
        lo_m1 = f"{window_lo}-1"
        return (
            f"{state}_{{{window_lo}}} - \\hat{{{state}}}_{{{window_lo}|{lo_m1}}}",
            f"P_{{{window_lo}|{lo_m1}}}^{{-1}}",
            "EKF arrival cost",
        )

    def _predict_inplace(self, u_t: np.ndarray) -> None:
        u_t = np.asarray(u_t, dtype=float).ravel()
        self._x = self.A @ self._x + self.B @ u_t
        self._P = self.A @ self._P @ self.A.T + self.Q

    def _update_inplace(self, y_t: np.ndarray, u_t: np.ndarray) -> None:
        y_t = np.asarray(y_t, dtype=float).ravel()
        u_t = np.asarray(u_t, dtype=float).ravel()
        y_pred = self.C @ self._x + self.D @ u_t
        innov = y_t - y_pred
        S = self.C @ self._P @ self.C.T + self.R
        K = self._P @ self.C.T @ np.linalg.inv(S)
        self._x = self._x + K @ innov
        IKC = self._I - K @ self.C
        self._P = IKC @ self._P @ IKC.T + K @ self.R @ K.T

    def _predict_copy(self, u_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u_t = np.asarray(u_t, dtype=float).ravel()
        x = self.A @ self._x + self.B @ u_t
        P = self.A @ self._P @ self.A.T + self.Q
        return x, P


class UKFArrivalCost(_LinearFilterArrivalCost):
    """Unscented Kalman filter arrival cost (LTI-compatible; optional nonlinear ``f``/``h``)."""

    def __init__(
        self,
        system: SystemModel,
        builder: ObjectiveBuilder | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        f: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        h: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ):
        if Q is None or R is None:
            if builder is None:
                raise ValueError(
                    "UKFArrivalCost requires builder or explicit Q and R."
                )
            Q_b, R_b = noise_covs_from_builder(builder, system.nx, system.ny)
            Q = Q if Q is not None else Q_b
            R = R if R is not None else R_b
        super().__init__(system, Q, R, x0=x0, P0=P0)
        self._f = f if f is not None else self._default_f
        self._h = h if h is not None else self._default_h
        self._nx = self.A.shape[0]
        self._ny = self.C.shape[0]
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        self._lambda = alpha**2 * (self._nx + kappa) - self._nx
        self._gamma = np.sqrt(self._nx + self._lambda)

    def initial_covariance(
        self, system: SystemModel, builder: ObjectiveBuilder
    ) -> np.ndarray:
        return self._P0.copy()

    def latex_parts(self, *, state: str, window_lo: str) -> tuple[str, str, str]:
        lo_m1 = f"{window_lo}-1"
        return (
            f"{state}_{{{window_lo}}} - \\hat{{{state}}}_{{{window_lo}|{lo_m1}}}",
            f"P_{{{window_lo}|{lo_m1}}}^{{-1}}",
            "UKF arrival cost",
        )

    def _default_f(self, x: np.ndarray, u_t: np.ndarray) -> np.ndarray:
        return self.A @ x + self.B @ u_t

    def _default_h(self, x: np.ndarray, u_t: np.ndarray) -> np.ndarray:
        return self.C @ x + self.D @ u_t

    def _sigma_weights(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self._nx
        lam = self._lambda
        Wm = np.full(2 * n + 1, 0.5 / (n + lam))
        Wc = Wm.copy()
        Wm[0] = lam / (n + lam)
        Wc[0] = lam / (n + lam) + (1.0 - self._alpha**2 + self._beta)
        return Wm, Wc, lam

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        n = self._nx
        lam = self._lambda
        try:
            sqrt_p = sla.cholesky((n + lam) * P, lower=True)
        except sla.LinAlgError:
            sqrt_p = sla.cholesky(
                (n + lam) * P + np.eye(n) * 1e-9, lower=True
            )
        chi = np.zeros((2 * n + 1, n))
        chi[0] = x
        for i in range(n):
            chi[i + 1] = x + sqrt_p[:, i]
            chi[i + 1 + n] = x - sqrt_p[:, i]
        return chi

    def _unscented_predict(
        self, x: np.ndarray, P: np.ndarray, u_t: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        chi = self._sigma_points(x, P)
        Wm, Wc, _ = self._sigma_weights()
        chi_pred = np.array([self._f(pt, u_t) for pt in chi])
        x_pred = Wm @ chi_pred
        P_pred = self.Q.copy()
        for i, pt in enumerate(chi_pred):
            dx = pt - x_pred
            P_pred += Wc[i] * np.outer(dx, dx)
        return x_pred, P_pred

    def _unscented_update(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        y_t: np.ndarray,
        u_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        chi = self._sigma_points(x_pred, P_pred)
        Wm, Wc, _ = self._sigma_weights()
        gamma_pred = np.array([self._h(pt, u_t) for pt in chi])
        y_mean = Wm @ gamma_pred
        P_yy = self.R.copy()
        P_xy = np.zeros((self._nx, self._ny))
        for i, pt in enumerate(chi):
            dy = gamma_pred[i] - y_mean
            P_yy += Wc[i] * np.outer(dy, dy)
            P_xy += Wc[i] * np.outer(pt - x_pred, dy)
        K = P_xy @ np.linalg.inv(P_yy)
        x = x_pred + K @ (y_t - y_mean)
        P = P_pred - K @ P_yy @ K.T
        return x, P

    def _predict_inplace(self, u_t: np.ndarray) -> None:
        self._x, self._P = self._unscented_predict(self._x, self._P, u_t)

    def _update_inplace(self, y_t: np.ndarray, u_t: np.ndarray) -> None:
        y_t = np.asarray(y_t, dtype=float).ravel()
        self._x, self._P = self._unscented_update(self._x, self._P, y_t, u_t)

    def _predict_copy(self, u_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._unscented_predict(self._x, self._P, u_t)
