"""Collect and validate input regulator terms for MHE solver construction."""

from __future__ import annotations

import numpy as np

from openmhe.mhe_strategies import ObjectiveBuilder


def term_kind(term) -> str:
    """Return the term type tag (``target_type`` or legacy ``type``)."""
    return getattr(term, "target_type", getattr(term, "type", ""))


def cost_terms(builder: ObjectiveBuilder) -> list:
    """Terms that contribute rows to the stage cost residual."""
    return [
        t
        for t in builder.terms
        if term_kind(t) not in ("INPUT_RANDOM_WALK", "KNOWN_INPUT", "UNKNOWN_INPUT")
    ]


def first_diff_terms(builder: ObjectiveBuilder) -> list:
    """All :class:`~openmhe.InputFirstDiffReg` (and legacy FIRST) terms."""
    out = []
    for term in builder.terms:
        if term_kind(term) == "INPUT_REG" and str(term.trend).upper() in (
            "FIRST_DIFF",
            "FIRST",
        ):
            out.append(term)
    return out


def second_diff_terms(builder: ObjectiveBuilder) -> list:
    """All :class:`~openmhe.InputSecondDiffReg` terms."""
    out = []
    for term in builder.terms:
        if term_kind(term) == "INPUT_REG" and str(term.trend).upper() == "SECOND_DIFF":
            out.append(term)
    return out


def random_walk_terms(builder: ObjectiveBuilder) -> list:
    """All :class:`~openmhe.InputRandomWalk` terms."""
    return [t for t in builder.terms if term_kind(t) == "INPUT_RANDOM_WALK"]


def known_input_terms(builder: ObjectiveBuilder) -> list:
    """All :class:`~openmhe.KnownInput` terms."""
    return [t for t in builder.terms if term_kind(t) == "KNOWN_INPUT"]


def collect_known_indices(builder: ObjectiveBuilder) -> list[int]:
    """Sorted unique input indices declared as :class:`~openmhe.KnownInput`."""
    out: list[int] = []
    for term in known_input_terms(builder):
        out.extend(int(i) for i in term.target_idx)
    if len(out) != len(set(out)):
        raise ValueError("Duplicate input index in KnownInput.")
    return out


def collect_tracked_indices(builder: ObjectiveBuilder) -> list[int]:
    """Sorted unique input indices in :class:`~openmhe.InputTrackingTerm` terms."""
    out: list[int] = []
    for term in builder.terms:
        if term_kind(term) == "INPUT_TRACKING":
            out.extend(int(i) for i in term.target_idx)
    if len(out) != len(set(out)):
        raise ValueError("Duplicate input index in InputTrackingTerm.")
    return out


def regulated_indices(
    rw_indices: list[int],
    fd_indices: list[int],
    sd_indices: list[int],
) -> set[int]:
    """Union of random-walk, first-diff, and second-diff input indices."""
    return set(rw_indices) | set(fd_indices) | set(sd_indices)


def collect_unknown_indices(builder: ObjectiveBuilder) -> list[int]:
    """Sorted unique input indices declared as :class:`~openmhe.UnknownInput`."""
    out: list[int] = []
    for term in builder.terms:
        if term_kind(term) == "UNKNOWN_INPUT":
            out.extend(int(i) for i in term.target_idx)
    if len(out) != len(set(out)):
        raise ValueError("Duplicate input index in UnknownInput.")
    return sorted(set(out))


def unknown_input_terms(builder: ObjectiveBuilder) -> list:
    """All :class:`~openmhe.UnknownInput` terms."""
    return [t for t in builder.terms if term_kind(t) == "UNKNOWN_INPUT"]


def validate_input_partition(
    builder: ObjectiveBuilder,
    nu: int,
    rw_indices: list[int],
    fd_indices: list[int],
    sd_indices: list[int],
) -> list[int]:
    """Ensure each column of ``B`` (input index ``0 .. nu-1``) has exactly one role.

    Roles are mutually exclusive: regulated (RW / FD / SD), :class:`KnownInput`,
    :class:`UnknownInput`, or :class:`InputTrackingTerm`.
    """
    known = set(collect_known_indices(builder))
    unknown = set(collect_unknown_indices(builder))
    tracked = set(collect_tracked_indices(builder))
    regulated = regulated_indices(rw_indices, fd_indices, sd_indices)

    if known & regulated:
        raise ValueError(
            f"Input index {sorted(known & regulated)} cannot be both KnownInput "
            "and regulated (RW/FD/SD)."
        )
    if known & unknown:
        raise ValueError(
            f"Input index {sorted(known & unknown)} cannot be both KnownInput "
            "and UnknownInput."
        )
    if unknown & regulated:
        raise ValueError(
            f"Input index {sorted(unknown & regulated)} cannot be both "
            "UnknownInput and regulated (RW/FD/SD)."
        )
    if known & tracked:
        raise ValueError(
            f"Input index {sorted(known & tracked)} cannot be both KnownInput "
            "and InputTrackingTerm."
        )
    if unknown & tracked:
        raise ValueError(
            f"Input index {sorted(unknown & tracked)} cannot be both UnknownInput "
            "and InputTrackingTerm."
        )
    if tracked & regulated:
        raise ValueError(
            f"Input index {sorted(tracked & regulated)} cannot be both "
            "InputTrackingTerm and regulated (RW/FD/SD)."
        )

    assigned = regulated | known | unknown | tracked
    expected = set(range(nu))
    missing = expected - assigned
    if missing:
        raise ValueError(
            f"Each input 0..{nu - 1} (B has {nu} columns) must be modeled exactly "
            f"once as regulated, KnownInput, UnknownInput, or InputTrackingTerm. "
            f"Unassigned: {sorted(missing)}."
        )
    extra = assigned - expected
    if extra:
        raise ValueError(
            f"Input index {sorted(extra)} is out of range; B has {nu} input(s)."
        )
    return sorted(known)


def collect_fd_indices(builder: ObjectiveBuilder) -> list[int]:
    """Unique input indices with first-difference regulation."""
    out: list[int] = []
    for term in first_diff_terms(builder):
        out.extend(int(i) for i in term.target_idx)
    return list(dict.fromkeys(out))


def collect_sd_indices(builder: ObjectiveBuilder) -> list[int]:
    """Unique input indices with second-difference regulation."""
    out: list[int] = []
    for term in second_diff_terms(builder):
        out.extend(int(i) for i in term.target_idx)
    return list(dict.fromkeys(out))


def collect_rw_indices(
    builder: ObjectiveBuilder, legacy_input_as_state: list[int] | None
) -> tuple[list[int], np.ndarray]:
    """Merge InputRandomWalk terms and legacy ``input_as_state`` kwarg."""
    indices: list[int] = []
    lambdas: list[float] = []
    for term in random_walk_terms(builder):
        for ui, lam in zip(term.target_idx, term.lambdas):
            indices.append(int(ui))
            lambdas.append(float(lam))
    for ui in legacy_input_as_state or []:
        ui = int(ui)
        if ui not in indices:
            indices.append(ui)
            lambdas.append(1.0)
    unique_idx: list[int] = []
    unique_lam: list[float] = []
    for ui, lam in zip(indices, lambdas):
        if ui not in unique_idx:
            unique_idx.append(ui)
            unique_lam.append(lam)
    return unique_idx, np.asarray(unique_lam, dtype=float)


def validate_input_models(
    rw_indices: list[int],
    fd_indices: list[int],
    sd_indices: list[int],
) -> None:
    """Reject duplicate indices and incompatible regulator combinations."""
    for name, group in (
        ("random walk", rw_indices),
        ("first difference", fd_indices),
        ("second difference", sd_indices),
    ):
        if len(group) != len(set(group)):
            raise ValueError(f"Duplicate input index in {name} regulator.")

    rw_set, fd_set, sd_set = set(rw_indices), set(fd_indices), set(sd_indices)
    overlap = (rw_set & fd_set) | (rw_set & sd_set) | (fd_set & sd_set)
    if overlap:
        raise ValueError(
            f"Input index {sorted(overlap)} cannot use multiple regulator models."
        )


def unmeasured_regulator_indices(
    builder: ObjectiveBuilder,
    rw_indices: list[int],
    fd_indices: list[int],
    sd_indices: list[int],
) -> set[int]:
    """Inputs with RW/FD/SD that are not pinned by measured tracking or KnownInput."""
    pinned = set(collect_known_indices(builder))
    for term in builder.terms:
        if term_kind(term) == "INPUT_TRACKING" and term.reference == "measured":
            pinned.update(int(i) for i in term.target_idx)
    return regulated_indices(rw_indices, fd_indices, sd_indices) - pinned


def unmeasured_input_indices(
    builder: ObjectiveBuilder,
    rw_indices: list[int],
    fd_indices: list[int],
    sd_indices: list[int],
) -> set[int]:
    """Input indices with no measured ``u`` reference (RW/FD/SD + UnknownInput)."""
    return unmeasured_regulator_indices(
        builder, rw_indices, fd_indices, sd_indices
    ) | set(collect_unknown_indices(builder))


def seed_reg_state_prior(
    x_prior: np.ndarray,
    *,
    rw_col: dict[int, int],
    fd_col: dict[int, int],
    sd1_col: dict[int, int],
    sd2_col: dict[int, int],
    rw_indices: list[int],
    fd_indices: list[int],
    sd_indices: list[int],
    unknown_indices: list[int],
    u_hat: np.ndarray,
    window_idx: int,
    unmeasured: set[int],
    u: np.ndarray | None = None,
    t_start: int | None = None,
) -> dict[int, float]:
    """Initialize RW / FD / SD states at the start of a sliding window.

    Prior-window MHE estimates ``u_hat[:, window_idx - lag]`` are used for
    unmeasured inputs. On the first window only, measured ``u`` at
    ``t_start - lag`` fills missing lags when estimates do not exist yet.

    Returns
    -------
    dict
        Per-input values used as ``u_{k-1}`` seed (for optional control init).
    """
    u_at_lag1: dict[int, float] = {}

    def _u_value(ui: int, lag: int) -> float:
        """Prior estimate, measured ``u``, or zero for lag ``lag`` of input ``ui``."""
        if window_idx >= lag and not np.isnan(u_hat[ui, window_idx - lag]):
            return float(u_hat[ui, window_idx - lag])
        if window_idx == 0 and u is not None and t_start is not None:
            t_lag = t_start - lag
            if t_lag >= 0:
                return float(u[ui, t_lag])
        if ui not in unmeasured and u is not None and t_start is not None:
            t_lag = t_start - lag
            if t_lag >= 0:
                return float(u[ui, t_lag])
        return 0.0

    for ui in rw_indices:
        val = _u_value(ui, 1)
        x_prior[rw_col[ui]] = val
        u_at_lag1[ui] = val

    for ui in fd_indices:
        val = _u_value(ui, 1)
        x_prior[fd_col[ui]] = val
        u_at_lag1[ui] = val

    for ui in sd_indices:
        x_prior[sd1_col[ui]] = _u_value(ui, 1)
        x_prior[sd2_col[ui]] = _u_value(ui, 2)
        u_at_lag1[ui] = x_prior[sd1_col[ui]]

    for ui in unknown_indices:
        u_at_lag1[ui] = _u_value(ui, 1)

    return u_at_lag1


def merge_process_weight(
    process_term,
    nx_base: int,
    rw_indices: list[int],
    rw_lambdas: np.ndarray,
) -> np.ndarray:
    """Return full ``nw x nw`` process weight with plant + random-walk blocks."""
    W = np.asarray(process_term.weight.W, dtype=float)
    n_rw = len(rw_indices)
    nw = nx_base + n_rw
    if W.shape == (nx_base, nx_base):
        W_full = np.zeros((nw, nw))
        W_full[:nx_base, :nx_base] = W
        for j in range(n_rw):
            W_full[nx_base + j, nx_base + j] = rw_lambdas[j]
        return W_full
    if W.shape == (nw, nw):
        for j in range(n_rw):
            W[nx_base + j, nx_base + j] = rw_lambdas[j]
        return W
    raise ValueError(
        f"ProcessTerm weight must be ({nx_base}, {nx_base}) or ({nw}, {nw}); "
        f"got {W.shape} with {n_rw} random-walk input(s)."
    )


def sparse_process_noise_config(
    W_full: np.ndarray,
    nx_base: int,
    n_rw: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Shrink process-noise channels with zero diagonal weight (strict kinematics).

    Returns ``(nw, W_sparse, G)`` where ``G`` has shape ``(nx_base + n_rw, nw)``
    and maps the reduced noise vector ``w`` into the full plant + RW injection
  sites ``[x_base; x_rw]``.
    """
    nw_full = nx_base + n_rw
    W_full = np.asarray(W_full, dtype=float)
    if W_full.shape != (nw_full, nw_full):
        raise ValueError(
            f"Expected process weight ({nw_full}, {nw_full}), got {W_full.shape}."
        )
    off_diag = W_full - np.diag(np.diag(W_full))
    if np.any(np.abs(off_diag) > 0):
        raise ValueError(
            "Strict kinematics (zero process cov entries) requires a diagonal "
            "process weight matrix."
        )
    diag = np.diag(W_full)
    active = np.flatnonzero(diag > 0)
    nw = int(active.size)
    if nw == 0:
        raise ValueError(
            "ProcessTerm requires at least one nonzero process-noise weight."
        )
    if nw == nw_full:
        return nw, W_full, np.eye(nw_full)

    W_sparse = W_full[np.ix_(active, active)]
    G = np.zeros((nw_full, nw))
    for j, idx in enumerate(active):
        G[idx, j] = 1.0
    return nw, W_sparse, G


def plant_arrival_state_indices(
    G_proc: np.ndarray,
    nx_base: int,
) -> np.ndarray | None:
    """Plant state indices that receive arrival cost under strict kinematics.

    Returns ``None`` when every plant state is included (dense process noise).
    """
    Gp = np.asarray(G_proc[:nx_base, :], dtype=float)
    if Gp.shape[1] == nx_base and np.allclose(Gp, np.eye(nx_base)):
        return None
    idx = np.unique(np.nonzero(Gp)[0]).astype(int)
    if idx.size == nx_base:
        return None
    return idx


def plant_process_cov(
    G: np.ndarray,
    W_sparse: np.ndarray,
    nx_base: int,
) -> np.ndarray:
    """Map sparse process weights to ``nx_base x nx_base`` plant covariance ``Q``."""
    Gp = np.asarray(G[:nx_base, :], dtype=float)
    W = np.asarray(W_sparse, dtype=float)
    diag_w = np.diag(W) if W.ndim == 2 else W
    if np.any(diag_w <= 0):
        raise ValueError("Plant process covariance requires positive sparse weights.")
    cov_w = 1.0 / diag_w
    return Gp @ np.diag(cov_w) @ Gp.T


def sparse_process_cov(G: np.ndarray, W_sparse: np.ndarray) -> np.ndarray:
    """Map sparse inverse-covariance weights to full state process covariance ``G Q_w G.T``."""
    G = np.asarray(G, dtype=float)
    W = np.asarray(W_sparse, dtype=float)
    diag_w = np.diag(W) if W.ndim == 2 else W
    if np.any(diag_w <= 0):
        raise ValueError("sparse_process_cov requires positive diagonal weights.")
    Q_w = np.diag(1.0 / diag_w)
    return G @ Q_w @ G.T
