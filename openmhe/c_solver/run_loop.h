#ifndef OPENMHE_RUN_LOOP_H_
#define OPENMHE_RUN_LOOP_H_

#include "acados_solver_openmhe_mhe.h"
#include "filter_arrival.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OPENMHE_MAX_U_EXTRACT 16
#define OPENMHE_MAX_PIN 8
#define OPENMHE_MAX_NU_PHY 16

/** One row of ``u_hat`` filled from the NLP solution at a stage. */
typedef struct {
    int out_ui;
    int from_u; /* 1: u_full[src_idx]; 0: x_end[src_idx] */
    int src_idx;
} openmhe_u_extract_t;

/** Runtime layout (must match the generated Acados problem). */
typedef struct {
    int n_steps;
    int N;
    int nx;       /* augmented state dimension (must match OPENMHE_MHE_NX) */
    int nx_base;
    int nu;
    int nu_ocp;   /* Acados control dimension (must match OPENMHE_MHE_NU) */
    int nu_ctrl;
    int ny_stage;
    int ny0;
    int n_arrival; /* plant arrival residual rows (<= nx_base) */
    int has_arrival;
    int arrival_off;     /* row offset of arrival block inside stage-0 ``yref`` */
    int dynamic_arrival; /* 1 when stage-0 ``W`` changes each window (EKF path) */
    int n_pin;
    int n_u_extract;
    int n_rw;
    int n_fd;
    int n_sd;
    int n_unmeasured;
    /** Enable LTI vector-only solves after window 0 (see ``lti_fast.h``). */
    int lti_linear_ls_fast;
    /** 1 when the problem was built with all-L2 ``LINEAR_LS`` penalties. */
    int linear_ls;
} openmhe_run_config_t;

/**
 * Plant-side EKF matrices and stage-0 weight template for in-C arrival cost.
 *
 * Passed from ``c_runner.run_c_solver`` when ``solver._filter_kind == "ekf"``.
 * ``NULL`` disables the live filter; legacy precomputed ``x_bar_pre`` /
 * ``W0_stage_pre`` buffers are unused in that case.
 *
 * ``y_meas`` (separate argument to ``openmhe_mhe_run_sliding``) must be
 * ``(n_steps, ny)`` row-major raw measurements for ``openmhe_filter_assimilate``.
 */
typedef struct {
    openmhe_filter_kind_t kind;
    int nx_base;
    int ny;
    int nu;
    int n_arrival;
    int arrival_w_off;
    const double *A;
    const double *B;
    const double *C;
    const double *D;
    const double *Q;
    const double *R;
    /** Fixed stage-0 weight from codegen; arrival block overwritten each window. */
    const double *W0_template;
} openmhe_arrival_filter_setup_t;

/** Allocate and set up the Acados NLP solver inside ``capsule``. */
int openmhe_mhe_init_solver(openmhe_mhe_solver_capsule *capsule);

/** Tear down the Acados NLP solver; safe if init was not called or failed. */
int openmhe_mhe_free_solver(openmhe_mhe_solver_capsule *capsule);

/**
 * Sliding-window MHE driver (C counterpart of ``run_solver``).
 *
 * Requires a prior successful call to ``openmhe_mhe_init_solver``.
 *
 * ``yrefs``: ``(n_steps, ny_stage)`` row-major; the driver advances a pointer
 * by ``ny_stage`` each window (no per-window copy of the full table).
 * ``x_bar_pre``: ``(n_est, nx_base)`` row-major arrival mean, or NULL.
 * ``W0_stage_pre``: ``(n_est, ny0, ny0)`` Fortran stage-0 weights for dynamic
 * arrival, or NULL (legacy; unused when ``filter_setup`` is set).
 * ``y_meas``: ``(n_steps, ny)`` row-major raw measurements for the in-C filter,
 * or NULL when ``filter_setup`` is NULL.
 * ``filter_setup``: live EKF arrival configuration, or NULL.
 */
int openmhe_mhe_run_sliding(
    openmhe_mhe_solver_capsule *capsule,
    const openmhe_run_config_t *cfg,
    const double *yrefs,
    const double *x_bar_pre,
    const double *W0_stage_pre,
    const double *pin_vals,
    const int *controlled_idx,
    const int *u_extract,
    const int *rw_idx,
    const int *rw_col,
    const int *fd_idx,
    const int *fd_col,
    const int *sd_idx,
    const int *sd1_col,
    const int *sd2_col,
    const int *unmeasured_ui,
    const int *arrival_state_idx,
    const double *u_meas,
    const double *y_meas,
    const openmhe_arrival_filter_setup_t *filter_setup,
    double *u_hat,
    double *x_hat,
    double *u_hat_raw);

#ifdef __cplusplus
}
#endif

#endif /* OPENMHE_RUN_LOOP_H_ */
