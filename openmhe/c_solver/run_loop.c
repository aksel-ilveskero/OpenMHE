/**
 * Sliding-window MHE driver (C counterpart of ``openmhe.run_solver``).
 *
 * Per window: seed regulator states → set ``yref`` / pins → (optional) EKF
 * arrival prior → Acados SQP-RTI solve → warm-start shift → filter assimilate.
 *
 * Dynamic ``EKFArrivalCost`` is evaluated in-process via ``filter_arrival.c``
 * (no Python pre-pass).  See ``openmhe/c_solver/README.md``.
 */
#define _POSIX_C_SOURCE 199309L

#include <string.h>

#include "acados_c/ocp_nlp_interface.h"
#include "acados/ocp_nlp/ocp_nlp_common.h"
#include "acados/ocp_nlp/ocp_nlp_sqp_rti.h"

#include "filter_arrival.h"
#include "lti_fast.h"
#include "profile.h"
#include "run_loop.h"

#define NX OPENMHE_MHE_NX
#define NU_OCP OPENMHE_MHE_NU
#define NY0 OPENMHE_MHE_NY0
#define N_HORIZON OPENMHE_MHE_N

#define ISNAN(x) ((x) != (x))

int openmhe_mhe_init_solver(openmhe_mhe_solver_capsule *capsule)
{
    if (capsule == NULL) {
        return -1;
    }
    return openmhe_mhe_acados_create(capsule);
}

int openmhe_mhe_free_solver(openmhe_mhe_solver_capsule *capsule)
{
    if (capsule == NULL) {
        return 0;
    }
    if (openmhe_mhe_acados_get_nlp_solver(capsule) == NULL) {
        return 0;
    }
    return openmhe_mhe_acados_free(capsule);
}

static void set_yref_stage(
    ocp_nlp_config *config,
    ocp_nlp_dims *dims,
    ocp_nlp_in *in,
    int stage,
    const double *yref)
{
    ocp_nlp_cost_model_set(config, dims, in, stage, "yref", (void *)yref);
}

static void get_x(
    ocp_nlp_config *config,
    ocp_nlp_dims *dims,
    ocp_nlp_out *out,
    int stage,
    double *x)
{
    ocp_nlp_out_get(config, dims, out, stage, "x", x);
}

static void get_u(
    ocp_nlp_config *config,
    ocp_nlp_dims *dims,
    ocp_nlp_out *out,
    int stage,
    double *u)
{
    ocp_nlp_out_get(config, dims, out, stage, "u", u);
}

static void set_x(
    ocp_nlp_config *config,
    ocp_nlp_dims *dims,
    ocp_nlp_out *out,
    ocp_nlp_in *in,
    int stage,
    const double *x)
{
    ocp_nlp_out_set(config, dims, out, in, stage, "x", (void *)x);
}

static void set_u(
    ocp_nlp_config *config,
    ocp_nlp_dims *dims,
    ocp_nlp_out *out,
    ocp_nlp_in *in,
    int stage,
    const double *u)
{
    ocp_nlp_out_set(config, dims, out, in, stage, "u", (void *)u);
}

/**
 * Shift x[1..N] -> x[0..N-1], u[1..N-1] -> u[0..N-2], and pi[1..N-1] -> pi[0..N-2]
 * via one dense memmove per field (ocp_nlp_{get,set}_all layout).
 *
 * OpenMHE keeps the same nx and nu at every shooting stage, so the get_all
 * packing is evenly strided and a single memmove per field is correct.
 */
static void warm_start_shift_bulk(
    ocp_nlp_solver *solver,
    ocp_nlp_in *in,
    ocp_nlp_out *out,
    int N,
    int nx,
    int nu)
{
    double x_traj[(N_HORIZON + 1) * NX];
    double u_traj[N_HORIZON * NU_OCP];
    double pi_traj[N_HORIZON * NX];

    ocp_nlp_get_all(solver, in, out, "x", x_traj);
    memmove(x_traj, x_traj + nx, (size_t)N * (size_t)nx * sizeof(double));
    ocp_nlp_set_all(solver, in, out, "x", x_traj);

    if (N <= 1) {
        return;
    }
    ocp_nlp_get_all(solver, in, out, "u", u_traj);
    memmove(u_traj, u_traj + nu, (size_t)(N - 1) * (size_t)nu * sizeof(double));
    ocp_nlp_set_all(solver, in, out, "u", u_traj);

    ocp_nlp_get_all(solver, in, out, "pi", pi_traj);
    memmove(pi_traj, pi_traj + nx, (size_t)(N - 1) * (size_t)nx * sizeof(double));
    ocp_nlp_set_all(solver, in, out, "pi", pi_traj);
}

static int pin_changed(
    const double *pin,
    const double *cached,
    int n_pin,
    int have_cache)
{
    if (!have_cache) {
        return 1;
    }
    return memcmp(pin, cached, (size_t)n_pin * sizeof(double)) != 0;
}

static void set_pin_if_changed(
    ocp_nlp_config *nlp_config,
    ocp_nlp_dims *nlp_dims,
    ocp_nlp_in *nlp_in,
    ocp_nlp_out *nlp_out,
    int stage,
    const double *pin,
    double *pin_cache,
    int n_pin,
    int *pin_cached)
{
    if (!pin_changed(pin, pin_cache, n_pin, *pin_cached)) {
        return;
    }
    ocp_nlp_constraints_model_set(
        nlp_config, nlp_dims, nlp_in, nlp_out, stage, "lbu", (void *)pin);
    ocp_nlp_constraints_model_set(
        nlp_config, nlp_dims, nlp_in, nlp_out, stage, "ubu", (void *)pin);
    memcpy(pin_cache, pin, (size_t)n_pin * sizeof(double));
    *pin_cached = 1;
}

static double u_value_at_lag(
    int ui,
    int lag,
    int window_idx,
    int t_start,
    int nu,
    const double *u_hat,
    const double *u_meas,
    const int *is_unmeasured)
{
    if (window_idx >= lag) {
        double v = u_hat[(size_t)(window_idx - lag) * (size_t)nu + (size_t)ui];
        if (!ISNAN(v)) {
            return v;
        }
    }
    if (window_idx == 0 && u_meas != NULL && t_start - lag >= 0) {
        return u_meas[(size_t)(t_start - lag) * (size_t)nu + (size_t)ui];
    }
    if (!is_unmeasured[ui] && u_meas != NULL && t_start - lag >= 0) {
        return u_meas[(size_t)(t_start - lag) * (size_t)nu + (size_t)ui];
    }
    return 0.0;
}

static void seed_reg_state_prior(
    double *x_prior,
    int window_idx,
    int t_start,
    int nu,
    int n_rw,
    const int *rw_idx,
    const int *rw_col,
    int n_fd,
    const int *fd_idx,
    const int *fd_col,
    int n_sd,
    const int *sd_idx,
    const int *sd1_col,
    const int *sd2_col,
    const int *is_unmeasured,
    const double *u_hat,
    const double *u_meas)
{
    for (int i = 0; i < n_rw; ++i) {
        int ui = rw_idx[i];
        x_prior[rw_col[i]] = u_value_at_lag(
            ui, 1, window_idx, t_start, nu, u_hat, u_meas, is_unmeasured);
    }
    for (int i = 0; i < n_fd; ++i) {
        int ui = fd_idx[i];
        x_prior[fd_col[i]] = u_value_at_lag(
            ui, 1, window_idx, t_start, nu, u_hat, u_meas, is_unmeasured);
    }
    for (int i = 0; i < n_sd; ++i) {
        int ui = sd_idx[i];
        x_prior[sd1_col[i]] = u_value_at_lag(
            ui, 1, window_idx, t_start, nu, u_hat, u_meas, is_unmeasured);
        x_prior[sd2_col[i]] = u_value_at_lag(
            ui, 2, window_idx, t_start, nu, u_hat, u_meas, is_unmeasured);
    }
}

static void extract_u_hat(
    const openmhe_u_extract_t *specs,
    int n_specs,
    const double *x_end,
    const double *u_full,
    double *u_hat_col)
{
    for (int i = 0; i < n_specs; ++i) {
        const openmhe_u_extract_t *sp = &specs[i];
        u_hat_col[sp->out_ui] =
            sp->from_u ? u_full[sp->src_idx] : x_end[sp->src_idx];
    }
}

static void set_stage0_yref(
    ocp_nlp_config *nlp_config,
    ocp_nlp_dims *nlp_dims,
    ocp_nlp_in *nlp_in,
    int ny0,
    int ny_stage,
    int arrival_off,
    int n_arrival,
    const int *arrival_state_idx,
    const double *yref_j,
    const double *x_bar,
    double *yref0)
{
    /*
     * Stage 0 carries the usual measurement / regulator residuals (first
     * ny_stage rows) plus an arrival block: ||x_0 - x_bar||^2_{W_arrival}.
     * ``x_bar`` comes from the EKF prior when dynamic arrival is active, or
     * from precomputed tables / warm-start otherwise.
     */
    memset(yref0, 0, (size_t)ny0 * sizeof(double));
    memcpy(yref0, yref_j, (size_t)ny_stage * sizeof(double));
    if (x_bar != NULL && n_arrival > 0) {
        if (arrival_state_idx == NULL) {
            memcpy(&yref0[arrival_off], x_bar, (size_t)n_arrival * sizeof(double));
        } else {
            for (int i = 0; i < n_arrival; ++i) {
                yref0[arrival_off + i] = x_bar[arrival_state_idx[i]];
            }
        }
    }
    set_yref_stage(nlp_config, nlp_dims, nlp_in, 0, yref0);
}

/**
 * Insert row-major ``W_block`` into column-major ``W0_buf`` at the arrival rows.
 *
 * Acados expects ``W`` in Fortran (column-major) layout.  ``W_block`` is the
 * ``n_arrival × n_arrival`` output of ``openmhe_arrival_weight_block`` (C order).
 */
static void scatter_arrival_W(
    double *W0_buf,
    int ny0,
    int arrival_w_off,
    int n_arrival,
    const double *W_block)
{
    for (int j = 0; j < n_arrival; ++j) {
        for (int i = 0; i < n_arrival; ++i) {
            W0_buf[(arrival_w_off + i) + (arrival_w_off + j) * ny0] =
                W_block[i * n_arrival + j];
        }
    }
}

/** Factor the condensed QP Hessian / coupling once (after a full RTI prep step). */
static void openmhe_condense_qp_lhs(
    ocp_nlp_config *config,
    ocp_nlp_dims *dims,
    ocp_nlp_opts *nlp_opts,
    ocp_nlp_memory *nlp_mem,
    ocp_nlp_workspace *nlp_work)
{
    ocp_qp_xcond_solver_config *qp_solver = config->qp_solver;
    qp_solver->condense_lhs(
        qp_solver, dims->qp_solver, nlp_mem->qp_in, nlp_mem->qp_out,
        nlp_opts->qp_solver_opts, nlp_mem->qp_solver_mem, nlp_work->qp_work);
}

static void openmhe_profile_add_acados(
    openmhe_profile_t *prof, ocp_nlp_solver *nlp_solver)
{
    ocp_nlp_config *config = nlp_solver->config;
    ocp_nlp_sqp_rti_memory *rti_mem = nlp_solver->mem;
    ocp_nlp_memory *nlp_mem = rti_mem->nlp_mem;
    ocp_nlp_timings *timings = nlp_mem->nlp_timings;

    prof->acados_lin += timings->time_lin;
    prof->acados_qp += timings->time_qp_sol;
    prof->acados_reg += timings->time_reg;
    prof->acados_glob += timings->time_glob;
    (void)config;
}

/**
 * Solve one sliding-window NLP.
 *
 * Fast path (``openmhe_mhe_solve_lti_fast``) is attempted from window 1 onward
 * when ``use_fast`` is set and either the condensed LHS is valid or dynamic
 * arrival requires a stage-0 Hessian refresh.  On failure, ``use_fast`` is
 * cleared and subsequent windows use the full Acados solve.
 */
static int openmhe_mhe_solve_window(
    openmhe_mhe_solver_capsule *capsule,
    ocp_nlp_solver *nlp_solver,
    const openmhe_run_config_t *cfg,
    int window_idx,
    int *lhs_valid,
    int *use_fast)
{
    if (*use_fast && window_idx > 0
        && (*lhs_valid || cfg->dynamic_arrival)) {
        const int stage0_hess = cfg->dynamic_arrival;
        const int status = openmhe_mhe_solve_lti_fast(
            capsule, stage0_hess, lhs_valid);
        if (status == 0) {
            return 0;
        }
        *use_fast = 0;
    }

    const int status = openmhe_mhe_acados_solve(capsule);
    if (status == 0 && *use_fast) {
        ocp_nlp_sqp_rti_memory *rti_mem = nlp_solver->mem;
        ocp_nlp_sqp_rti_opts *rti_opts = openmhe_mhe_acados_get_nlp_opts(capsule);
        ocp_nlp_sqp_rti_workspace *rti_work = nlp_solver->work;
        openmhe_condense_qp_lhs(
            nlp_solver->config, nlp_solver->dims, rti_opts->nlp_opts,
            rti_mem->nlp_mem, rti_work->nlp_work);
        *lhs_valid = 1;
    }
    (void)window_idx;
    return status;
}

int openmhe_mhe_run_sliding(
    openmhe_mhe_solver_capsule *capsule,
    const openmhe_run_config_t *cfg,
    const double *yrefs,
    const double *x_bar_pre,
    const double *W0_stage_pre,
    const double *pin_vals,
    const int *controlled_idx,
    const int *u_extract_raw,
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
    double *u_hat_raw)
{
    const int n_steps = cfg->n_steps;
    const int N = cfg->N;
    const int nx = cfg->nx;
    const int nx_base = cfg->nx_base;
    const int nu_ocp = cfg->nu_ocp;
    const int nu = cfg->nu;
    const int nu_ctrl = cfg->nu_ctrl;
    const int ny_stage = cfg->ny_stage;
    const int out_stage =
        (cfg->output_stage >= 0 && cfg->output_stage < N) ? cfg->output_stage : (N - 1);
    const int ny0 = cfg->ny0;
    const int n_arrival = cfg->n_arrival;
    const int n_pin = cfg->n_pin;
    const int n_est = n_steps - N;

    if (n_est <= 0 || capsule == NULL) {
        return -1;
    }
    if (nx != OPENMHE_MHE_NX || nu_ocp != OPENMHE_MHE_NU || ny0 > OPENMHE_MHE_NY0
        || N != N_HORIZON || n_pin > OPENMHE_MAX_PIN) {
        return -3;
    }
    if (openmhe_mhe_acados_get_nlp_solver(capsule) == NULL) {
        return -4;
    }

    ocp_nlp_config *nlp_config = openmhe_mhe_acados_get_nlp_config(capsule);
    ocp_nlp_dims *nlp_dims = openmhe_mhe_acados_get_nlp_dims(capsule);
    ocp_nlp_in *nlp_in = openmhe_mhe_acados_get_nlp_in(capsule);
    ocp_nlp_out *nlp_out = openmhe_mhe_acados_get_nlp_out(capsule);
    ocp_nlp_solver *nlp_solver = openmhe_mhe_acados_get_nlp_solver(capsule);

    openmhe_u_extract_t u_specs[OPENMHE_MAX_U_EXTRACT];
    if (cfg->n_u_extract > OPENMHE_MAX_U_EXTRACT) {
        return -2;
    }
    for (int i = 0; i < cfg->n_u_extract; ++i) {
        u_specs[i].out_ui = u_extract_raw[3 * i];
        u_specs[i].from_u = u_extract_raw[3 * i + 1];
        u_specs[i].src_idx = u_extract_raw[3 * i + 2];
    }

    if (nu > OPENMHE_MAX_NU_PHY) {
        return -2;
    }
    int is_unmeasured[OPENMHE_MAX_NU_PHY];
    memset(is_unmeasured, 0, sizeof(is_unmeasured));
    for (int j = 0; j < cfg->n_unmeasured; ++j) {
        is_unmeasured[unmeasured_ui[j]] = 1;
    }

    double x_prior[NX];
    double x_end[NX];
    double u_full[NU_OCP];
    double yref0[NY0];
    double pin_cache[N_HORIZON][OPENMHE_MAX_PIN];
    int pin_cached[N_HORIZON];

    const double *yref_win = yrefs;
    const double *pin_win = pin_vals;

    memset(x_prior, 0, (size_t)NX * sizeof(double));
    memset(pin_cached, 0, sizeof(pin_cached));

    openmhe_profile_t prof = {0};
    OPENMHE_PROF_DECL;

    int status = 0;
    int lhs_valid = 0; /* condensed QP LHS reusable across windows */
    int use_fast = 0;
    if (cfg->lti_linear_ls_fast && cfg->linear_ls
        && openmhe_mhe_is_linear_ls_plan(capsule)) {
        use_fast = 1; /* first window always runs full solve + condense_lhs */
    }

    const int filter_live = filter_setup != NULL
        && (filter_setup->kind == OPENMHE_FILTER_EKF
            || filter_setup->kind == OPENMHE_FILTER_UKF)
        && cfg->dynamic_arrival;

    const int ukf_filter = filter_live
        && filter_setup->kind == OPENMHE_FILTER_UKF;
    const int ny_f = filter_setup != NULL ? filter_setup->ny : 1;
    const int n_ukf_chi = ukf_filter ? (2 * nx_base + 1) * nx_base : 1;
    const int n_ukf_gamma = ukf_filter ? (2 * nx_base + 1) * ny_f : 1;

    /*
     * Incremental EKF/UKF for dynamic arrival cost (replaces Python pre-pass).
     * Stack buffers scale with plant nx_base (not augmented nx).
     */
    openmhe_filter_state_t filter_st;
    openmhe_filter_config_t filter_cfg;
    double filter_x[nx_base];
    double filter_P[nx_base * nx_base];
    double ukf_chi[n_ukf_chi];
    double ukf_chi_pred[n_ukf_chi];
    double ukf_gamma[n_ukf_gamma];
    openmhe_ukf_workspace_t ukf_ws;

    if (filter_live) {
        filter_cfg.kind = filter_setup->kind;
        filter_cfg.nx = filter_setup->nx_base;
        filter_cfg.ny = filter_setup->ny;
        filter_cfg.nu = filter_setup->nu;
        filter_cfg.alpha = filter_setup->alpha;
        filter_cfg.beta = filter_setup->beta;
        filter_cfg.kappa = filter_setup->kappa;
        filter_cfg.A = filter_setup->A;
        filter_cfg.B = filter_setup->B;
        filter_cfg.C = filter_setup->C;
        filter_cfg.D = filter_setup->D;
        filter_cfg.Q = filter_setup->Q;
        filter_cfg.R = filter_setup->R;
        if (ukf_filter) {
            ukf_ws.chi = ukf_chi;
            ukf_ws.chi_pred = ukf_chi_pred;
            ukf_ws.gamma_pred = ukf_gamma;
            openmhe_filter_init(
                &filter_st, &filter_cfg, filter_x, filter_P, &ukf_ws);
        } else {
            openmhe_filter_init(
                &filter_st, &filter_cfg, filter_x, filter_P, NULL);
        }
    }

    for (int idx = 0; idx < n_est; ++idx) {
        const int t_start = idx;
        const double *x_bar_win = NULL;
        double x_bar_live[nx_base];
        double P_prior[nx_base * nx_base];
        double W0_buf[NY0 * NY0];

        if (filter_live && u_meas != NULL) {
            /* Prior at t=idx before y[idx] enters the filter (matches Python). */
            const double *u_t = u_meas + (size_t)idx * (size_t)filter_setup->nu;
            openmhe_filter_prior(&filter_st, u_t, x_bar_live, P_prior);
            x_bar_win = x_bar_live;
        } else if (x_bar_pre != NULL) {
            x_bar_win = x_bar_pre + (size_t)idx * (size_t)nx_base;
        }

        OPENMHE_PROF_NOW(&_omhe_ta);

        seed_reg_state_prior(
            x_prior, idx, t_start, nu, cfg->n_rw, rw_idx, rw_col, cfg->n_fd,
            fd_idx, fd_col, cfg->n_sd, sd_idx, sd1_col, sd2_col, is_unmeasured,
            u_hat_raw, u_meas);

        set_x(nlp_config, nlp_dims, nlp_out, nlp_in, 0, x_prior);

        if (idx > 0 && nu_ctrl > 0) {
            get_u(nlp_config, nlp_dims, nlp_out, 0, u_full);
            for (int c = 0; c < nu_ctrl; ++c) {
                int ui = controlled_idx[c];
                if (is_unmeasured[ui]) {
                    u_full[c] = u_value_at_lag(
                        ui, 1, idx, t_start, nu, u_hat_raw, u_meas,
                        is_unmeasured);
                }
            }
            set_u(nlp_config, nlp_dims, nlp_out, nlp_in, 0, u_full);
        }

        for (int j = 0; j < N; ++j) {
            const double *yref_j = yref_win + (size_t)j * (size_t)ny_stage;

            if (n_pin > 0 && pin_win != NULL) {
                const double *pin = pin_win + (size_t)j * (size_t)n_pin;
                set_pin_if_changed(
                    nlp_config, nlp_dims, nlp_in, nlp_out, j, pin, pin_cache[j],
                    n_pin, &pin_cached[j]);
            }

            if (j == 0 && cfg->has_arrival) {
                set_stage0_yref(
                    nlp_config, nlp_dims, nlp_in, ny0, ny_stage, cfg->arrival_off,
                    n_arrival, arrival_state_idx, yref_j, x_bar_win, yref0);
                if (cfg->dynamic_arrival) {
                    if (filter_live && filter_setup->W0_template != NULL) {
                        /*
                         * Copy static stage-0 weights, then patch only the
                         * arrival diagonal block from invert_arrival(P_prior).
                         */
                        memcpy(
                            W0_buf, filter_setup->W0_template,
                            (size_t)ny0 * (size_t)ny0 * sizeof(double));
                        if (n_arrival > 0) {
                            double W_block[n_arrival * n_arrival];
                            openmhe_arrival_weight_block(
                                P_prior, nx_base, arrival_state_idx, n_arrival,
                                W_block);
                            scatter_arrival_W(
                                W0_buf, ny0, cfg->arrival_off, n_arrival, W_block);
                        }
                        ocp_nlp_cost_model_set(
                            nlp_config, nlp_dims, nlp_in, 0, "W", (void *)W0_buf);
                    } else if (W0_stage_pre != NULL) {
                        const double *W_win = W0_stage_pre
                            + (size_t)idx * (size_t)ny0 * (size_t)ny0;
                        ocp_nlp_cost_model_set(
                            nlp_config, nlp_dims, nlp_in, 0, "W", (void *)W_win);
                    }
                }
            } else {
                set_yref_stage(nlp_config, nlp_dims, nlp_in, j, yref_j);
            }
        }

        OPENMHE_PROF_NOW(&_omhe_tb);
        openmhe_profile_add(&prof.setup, &_omhe_ta, &_omhe_tb);

        OPENMHE_PROF_NOW(&_omhe_ta);
        status = openmhe_mhe_solve_window(
            capsule, nlp_solver, cfg, idx, &lhs_valid, &use_fast);
        openmhe_profile_add_acados(&prof, nlp_solver);
        OPENMHE_PROF_NOW(&_omhe_tb);
        openmhe_profile_add(&prof.solve, &_omhe_ta, &_omhe_tb);
        if (status != 0) {
            /*
             * Advance the filter even when the NLP fails so the next window's
             * prior stays aligned with Python ``window_prior(idx+1, …)``.
             */
            if (filter_live && y_meas != NULL && u_meas != NULL) {
                const double *y_t =
                    y_meas + (size_t)idx * (size_t)filter_setup->ny;
                const double *u_t =
                    u_meas + (size_t)idx * (size_t)filter_setup->nu;
                openmhe_filter_assimilate(&filter_st, y_t, u_t);
            }
            yref_win += ny_stage;
            if (pin_win != NULL) {
                pin_win += n_pin;
            }
            continue;
        }

        get_x(nlp_config, nlp_dims, nlp_out, out_stage, x_end);
        get_u(nlp_config, nlp_dims, nlp_out, out_stage, u_full);

        memcpy(
            &x_hat[(size_t)idx * (size_t)nx_base], x_end,
            (size_t)nx_base * sizeof(double));

        double *u_hat_col = &u_hat[(size_t)idx * (size_t)nu];
        double *u_raw_col = &u_hat_raw[(size_t)idx * (size_t)nu];
        extract_u_hat(u_specs, cfg->n_u_extract, x_end, u_full, u_hat_col);
        memcpy(u_raw_col, u_hat_col, (size_t)nu * sizeof(double));

        if (cfg->has_arrival) {
            get_x(nlp_config, nlp_dims, nlp_out, 1, x_prior);
        } else {
            get_x(nlp_config, nlp_dims, nlp_out, 0, x_prior);
        }

        OPENMHE_PROF_NOW(&_omhe_ta);
        warm_start_shift_bulk(nlp_solver, nlp_in, nlp_out, N, nx, nu_ocp);
        OPENMHE_PROF_NOW(&_omhe_tb);
        openmhe_profile_add(&prof.shift, &_omhe_ta, &_omhe_tb);

        if (filter_live && y_meas != NULL && u_meas != NULL) {
            /* Assimilate y[idx] after this window (posterior ← idx). */
            const double *y_t = y_meas + (size_t)idx * (size_t)filter_setup->ny;
            const double *u_t = u_meas + (size_t)idx * (size_t)filter_setup->nu;
            openmhe_filter_assimilate(&filter_st, y_t, u_t);
        }

        yref_win += ny_stage;
        if (pin_win != NULL) {
            pin_win += n_pin;
        }
    }

    openmhe_profile_report(&prof, n_est, "profile");

    return 0;
}
