#define _POSIX_C_SOURCE 199309L

#include <string.h>

#include "acados_c/ocp_nlp_interface.h"
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
    int nx_base,
    const double *yref_j,
    const double *x_bar,
    double *yref0)
{
    memset(yref0, 0, (size_t)ny0 * sizeof(double));
    memcpy(yref0, yref_j, (size_t)ny_stage * sizeof(double));
    if (x_bar != NULL) {
        memcpy(&yref0[arrival_off], x_bar, (size_t)nx_base * sizeof(double));
    }
    set_yref_stage(nlp_config, nlp_dims, nlp_in, 0, yref0);
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
    const double *u_meas,
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
    const int ny0 = cfg->ny0;
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

    for (int idx = 0; idx < n_est; ++idx) {
        const int t_start = idx;
        const double *x_bar_win = NULL;
        if (x_bar_pre != NULL) {
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
                    nx_base, yref_j, x_bar_win, yref0);
                if (cfg->dynamic_arrival && W0_stage_pre != NULL) {
                    const double *W_win = W0_stage_pre
                        + (size_t)idx * (size_t)ny0 * (size_t)ny0;
                    ocp_nlp_cost_model_set(
                        nlp_config, nlp_dims, nlp_in, 0, "W", (void *)W_win);
                }
            } else {
                set_yref_stage(nlp_config, nlp_dims, nlp_in, j, yref_j);
            }
        }

        OPENMHE_PROF_NOW(&_omhe_tb);
        openmhe_profile_add(&prof.setup, &_omhe_ta, &_omhe_tb);

        OPENMHE_PROF_NOW(&_omhe_ta);
        status = openmhe_mhe_acados_solve(capsule);
        OPENMHE_PROF_NOW(&_omhe_tb);
        openmhe_profile_add(&prof.solve, &_omhe_ta, &_omhe_tb);
        if (status != 0) {
            yref_win += ny_stage;
            if (pin_win != NULL) {
                pin_win += n_pin;
            }
            continue;
        }

        get_x(nlp_config, nlp_dims, nlp_out, N - 1, x_end);
        get_u(nlp_config, nlp_dims, nlp_out, N - 1, u_full);

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

        yref_win += ny_stage;
        if (pin_win != NULL) {
            pin_win += n_pin;
        }
    }

    openmhe_profile_report(&prof, n_est, "profile");

    return 0;
}
