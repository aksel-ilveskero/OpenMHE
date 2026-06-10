/**
 * LTI EKF/UKF, arrival-weight inversion, and Cholesky utilities for the C MHE driver.
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

/** LTI noise / dynamics matrices (row-major). */
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

typedef struct {
    openmhe_filter_config_t cfg;
    int posterior_t;
    double lambda;
    double *x;
    double *P;
    openmhe_ukf_workspace_t *ws;
} openmhe_filter_state_t;

/** Bind stack-allocated ``x`` (length nx) and ``P`` (nx*nx). */
void openmhe_filter_init(
    openmhe_filter_state_t *state,
    const openmhe_filter_config_t *cfg,
    double *x,
    double *P,
    openmhe_ukf_workspace_t *ws);

/**
 * One UKF predict+update at time ``t`` using ``y_t``, ``u_t``.
 * Call once per sliding-window step when advancing the arrival filter.
 */
void openmhe_filter_assimilate(
    openmhe_filter_state_t *state, const double *y_t, const double *u_t);

/**
 * UKF prior for ``u_t`` into ``x_bar``, ``P_prior`` (does not mutate posterior).
 */
void openmhe_filter_prior(
    const openmhe_filter_state_t *state,
    const double *u_t,
    double *x_bar,
    double *P_prior);

/**
 * Invert symmetric positive-definite ``P`` (n×n row-major) into ``P_inv``.
 * Returns 0 on success.
 */
int openmhe_symmetric_inv(const double *P, int n, double *P_inv);

/** Defaults matching ``openmhe.invert_arrival_covariance`` in Python. */
#define OPENMHE_ARRIVAL_INV_TOL 1e-8
#define OPENMHE_ARRIVAL_MAX_WEIGHT 1e6

/**
 * Arrival weight ``P^{-1}`` with null-space eigenvalues zeroed and weights capped.
 * ``P`` and ``P_inv`` are ``n×n`` row-major symmetric.
 */
void openmhe_invert_arrival_covariance(
    const double *P,
    int n,
    double *P_inv,
    double tol,
    double max_weight);

/**
 * Extract the plant arrival block from ``P_full`` and write ``n_arrival×n_arrival``
 * inverted weights into ``W_block`` (row-major).
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
