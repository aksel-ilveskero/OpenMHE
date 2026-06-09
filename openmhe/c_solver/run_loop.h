#ifndef OPENMHE_RUN_LOOP_H_
#define OPENMHE_RUN_LOOP_H_

#include "acados_solver_openmhe_mhe.h"

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
    int has_arrival;
    int arrival_off;
    int dynamic_arrival;
    int n_pin;
    int n_u_extract;
    int n_rw;
    int n_fd;
    int n_sd;
    int n_unmeasured;
    int lti_linear_ls_fast;
    int linear_ls;
} openmhe_run_config_t;

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
 * arrival, or NULL.
 * ``pin_vals``: ``(n_steps, n_pin)`` row-major input pins, or NULL.
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
    const double *u_meas,
    double *u_hat,
    double *x_hat,
    double *u_hat_raw);

#ifdef __cplusplus
}
#endif

#endif /* OPENMHE_RUN_LOOP_H_ */
