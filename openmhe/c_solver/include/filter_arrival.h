/**
 * LTI Kalman filters and arrival-weight utilities for the C MHE driver.
 *
 * Production use (``run_loop.c``):
 *   - ``OPENMHE_FILTER_EKF`` / ``OPENMHE_FILTER_UKF``: incremental arrival cost.
 *   - ``openmhe_invert_arrival_covariance``: stage-0 weight block from filter ``P``.
 *
 * Matrix layout: all filter dynamics/noise matrices (``A``, ``B``, ``C``, ``D``,
 * ``Q``, ``R``, ``P``) are dense row-major.  Acados stage-0 ``W`` passed from
 * Python is column-major (Fortran); see ``scatter_arrival_W`` in ``run_loop.c``.
 */
#ifndef OPENMHE_FILTER_ARRIVAL_H_
#define OPENMHE_FILTER_ARRIVAL_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OPENMHE_FILTER_NONE = 0,
    OPENMHE_FILTER_UKF = 1,
    OPENMHE_FILTER_EKF = 2,
} openmhe_filter_kind_t;

/**
 * Immutable LTI model and noise matrices for one filter instance.
 *
 * Dimensions ``nx``, ``ny``, ``nu`` refer to the *plant* (``nx_base``), not the
 * augmented MHE state.  UKF tuning fields are ignored when ``kind == EKF``.
 */
typedef struct {
    openmhe_filter_kind_t kind;
    int nx;
    int ny;
    int nu;
    double alpha;
    double beta;
    double kappa;
    const double *A;
    const double *B;
    const double *C;
    const double *D;
    const double *Q;
    const double *R;
} openmhe_filter_config_t;

/** Scratch for UKF sigma points (caller allocates, length ``(2*nx+1)*nx``). */
typedef struct {
    double *chi;
    double *chi_pred;
    double *gamma_pred;
} openmhe_ukf_workspace_t;

/**
 * Mutable filter state.  ``posterior_t`` is the time index of the last
 * assimilated measurement: -1 before any ``openmhe_filter_assimilate`` call.
 *
 * For sliding-window MHE at window ``idx`` (covering measurements
 * ``y[idx] … y[idx+N-1]``):
 *   1. ``openmhe_filter_prior(u[idx])``  → prior at ``idx`` (posterior still at ``idx-1``).
 *   2. MHE solve uses that prior in the stage-0 arrival cost.
 *   3. ``openmhe_filter_assimilate(y[idx], u[idx])`` advances posterior to ``idx``.
 *
 * This matches Python ``EKFArrivalCost.window_prior(idx, …)`` semantics.
 */
typedef struct {
    openmhe_filter_config_t cfg;
    int posterior_t;
    double lambda;
    double *x;
    double *P;
    openmhe_ukf_workspace_t *ws;
} openmhe_filter_state_t;

/**
 * Bind caller-owned storage.  ``x`` has length ``nx``; ``P`` is ``nx×nx`` row-major.
 *
 * Initial mean is zero; initial covariance is ``diag(Q)`` (same as Python
 * ``EKFArrivalCost`` when ``P0`` is not supplied).  ``ws`` may be NULL for EKF.
 */
void openmhe_filter_init(
    openmhe_filter_state_t *state,
    const openmhe_filter_config_t *cfg,
    double *x,
    double *P,
    openmhe_ukf_workspace_t *ws);

/**
 * One filter time step: predict with ``u_t``, then assimilate ``y_t``.
 *
 * Called once per MHE window after the NLP solve (including failed solves) so
 * the next window's prior reflects ``y[idx]``.  Mutates ``state->x`` and ``state->P``.
 */
void openmhe_filter_assimilate(
    openmhe_filter_state_t *state, const double *y_t, const double *u_t);

/**
 * Prior at the current step without advancing ``posterior_t``.
 *
 * For EKF this copies the posterior and applies one predict step into ``x_bar``,
 * ``P_prior``.  Used to populate stage-0 ``yref`` and ``W`` before the MHE solve.
 */
void openmhe_filter_prior(
    const openmhe_filter_state_t *state,
    const double *u_t,
    double *x_bar,
    double *P_prior);

/**
 * Plain Cholesky inverse for SPD ``P`` (e.g. innovation covariance ``S``).
 *
 * Used inside the EKF update where ``P`` is guaranteed SPD after adding ``R``.
 * Do *not* use for arrival weights — see ``openmhe_invert_arrival_covariance``.
 */
int openmhe_symmetric_inv(const double *P, int n, double *P_inv);

/** Relative eigenvalue cutoff; mirrors ``invert_arrival_covariance(..., tol=…)``. */
#define OPENMHE_ARRIVAL_INV_TOL 1e-8
/** Clip large diagonal weights to keep the stage-0 QP well scaled. */
#define OPENMHE_ARRIVAL_MAX_WEIGHT 1e6

/**
 * Arrival weight ``P^{-1}`` with null-space eigenvalues zeroed and weights capped.
 *
 * Implements the same logic as ``openmhe.invert_arrival_covariance`` in Python:
 * eigenvalues below ``tol * max(λ, 1)`` get zero weight (strict kinematics /
 * unobservable directions); remaining weights are ``min(1/λ, max_weight)``.
 *
 * ``P`` and ``P_inv`` are ``n×n`` row-major symmetric.
 */
void openmhe_invert_arrival_covariance(
    const double *P,
    int n,
    double *P_inv,
    double tol,
    double max_weight);

/**
 * Build the stage-0 arrival weight block from the filter prior covariance.
 *
 * Extracts the ``n_arrival × n_arrival`` plant submatrix of ``P_full`` (either
 * the leading block when ``arrival_state_idx == NULL``, or rows/cols indexed by
 * ``arrival_state_idx`` for sparse process noise), then writes ``W_block =
 * invert_arrival_covariance(P_sub)`` in row-major form for ``scatter_arrival_W``.
 */
void openmhe_arrival_weight_block(
    const double *P_full,
    int nx,
    const int *arrival_state_idx,
    int n_arrival,
    double *W_block);

#ifdef __cplusplus
}
#endif

#endif /* OPENMHE_FILTER_ARRIVAL_H_ */
