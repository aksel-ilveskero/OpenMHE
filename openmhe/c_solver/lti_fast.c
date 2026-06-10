/**
 * Implementation of the LTI + LINEAR_LS fast SQP-RTI path.
 *
 * Algorithm (per window, after window 0 has built ``lhs_valid``):
 *   1. ``update_qp_matrices`` on dynamics / cost / constraints (no Hessians).
 *   2. Levenberg-Marquardt regularisation term into the QP.
 *   3. ``ocp_nlp_approximate_qp_vectors_sqp`` (rhs only).
 *   4. ``ocp_nlp_solve_qp_and_correct_dual`` with ``precondensed_lhs=true``.
 *   5. RTI globalization (full step for LTI).
 *
 * Window 0 and any failed fast step fall back to a full
 * ``openmhe_mhe_acados_solve`` in ``run_loop.c``, which also condenses the LHS.
 */
#include "blasfeo_d_aux.h"
#include "acados/ocp_nlp/ocp_nlp_common.h"
#include "acados/ocp_nlp/ocp_nlp_sqp_rti.h"
#include "acados/utils/timing.h"

#include "lti_fast.h"

/** Check if the plan is a linear LS plan. */
int openmhe_mhe_is_linear_ls_plan(openmhe_mhe_solver_capsule *capsule)
{
    if (capsule == NULL) {
        return 0;
    }
    ocp_nlp_plan_t *plan = openmhe_mhe_acados_get_nlp_plan(capsule);
    if (plan == NULL) {
        return 0;
    }
    const int N = plan->N;
    if (plan->nlp_cost[0] != LINEAR_LS) {
        return 0;
    }
    for (int i = 1; i < N; ++i) {
        if (plan->nlp_cost[i] != LINEAR_LS) {
            return 0;
        }
    }
    if (plan->nlp_cost[N] != LINEAR_LS) {
        return 0;
    }
    return 1;
}

/** Assemble quadratic program vectors without recomputing constant Jacobians / Hessians. */
static void openmhe_approximate_qp_vectors_only(
    ocp_nlp_config *config,
    ocp_nlp_dims *dims,
    ocp_nlp_in *nlp_in,
    ocp_nlp_out *nlp_out,
    ocp_nlp_opts *nlp_opts,
    ocp_nlp_memory *nlp_mem,
    ocp_nlp_workspace *nlp_work,
    int stage0_full_hess)
{
    const int N = dims->N;
    int *nv = dims->nv;
    int *nx = dims->nx;
    int compute_hess_off = 0;

    ocp_nlp_initialize_submodules(
        config, dims, nlp_in, nlp_out, nlp_opts, nlp_mem, nlp_work);

    for (int i = 0; i <= N; ++i) {
        if (i < N) {
            config->dynamics[i]->update_qp_matrices(
                config->dynamics[i], dims->dynamics[i], nlp_in->dynamics[i],
                nlp_opts->dynamics[i], nlp_mem->dynamics[i],
                nlp_work->dynamics[i]);
        }

        const int hess_stage = (i == 0 && stage0_full_hess);
        if (!hess_stage) {
            /*
             * Skip cost Hessians on stages 1…N (LTI fast path).  Stage 0 still
             * gets a full Hessian when ``stage0_full_hess`` is set — required
             * each window for dynamic EKF arrival because ``W0`` changes.
             */
            config->cost[i]->opts_set(
                config->cost[i], nlp_opts->cost[i], "compute_hess", &compute_hess_off);
        }
        config->cost[i]->update_qp_matrices(
            config->cost[i], dims->cost[i], nlp_in->cost[i], nlp_opts->cost[i],
            nlp_mem->cost[i], nlp_work->cost[i]);

        config->constraints[i]->update_qp_matrices(
            config->constraints[i], dims->constraints[i], nlp_in->constraints[i],
            nlp_opts->constraints[i], nlp_mem->constraints[i],
            nlp_work->constraints[i]);
    }

    ocp_nlp_add_levenberg_marquardt_term(
        config, dims, nlp_in, nlp_out, nlp_opts, nlp_mem, nlp_work, 1.0, 0,
        nlp_mem->qp_in);

    for (int i = 0; i <= N; ++i) {
        struct blasfeo_dvec *cost_grad =
            config->cost[i]->memory_get_grad_ptr(nlp_mem->cost[i]);
        blasfeo_dveccp(nv[i], cost_grad, 0, nlp_mem->cost_grad + i, 0);
        if (i < N) {
            struct blasfeo_dvec *dyn_fun =
                config->dynamics[i]->memory_get_fun_ptr(nlp_mem->dynamics[i]);
            blasfeo_dveccp(nx[i + 1], dyn_fun, 0, nlp_mem->dyn_fun + i, 0);
        }
    }

    ocp_nlp_approximate_qp_vectors_sqp(
        config, dims, nlp_in, nlp_out, nlp_opts, nlp_mem, nlp_work);
}

int openmhe_mhe_solve_lti_fast(
    openmhe_mhe_solver_capsule *capsule,
    int stage0_full_hess,
    int *lhs_valid)
{
    if (capsule == NULL || lhs_valid == NULL) {
        return -1;
    }

    ocp_nlp_config *config = openmhe_mhe_acados_get_nlp_config(capsule);
    ocp_nlp_dims *dims = openmhe_mhe_acados_get_nlp_dims(capsule);
    ocp_nlp_in *nlp_in = openmhe_mhe_acados_get_nlp_in(capsule);
    ocp_nlp_out *nlp_out = openmhe_mhe_acados_get_nlp_out(capsule);
    ocp_nlp_solver *nlp_solver = openmhe_mhe_acados_get_nlp_solver(capsule);
    void *nlp_opts_wrap = openmhe_mhe_acados_get_nlp_opts(capsule);

    ocp_nlp_sqp_rti_opts *rti_opts = nlp_opts_wrap;
    ocp_nlp_opts *nlp_opts = rti_opts->nlp_opts;
    ocp_nlp_sqp_rti_memory *rti_mem = nlp_solver->mem;
    ocp_nlp_sqp_rti_workspace *rti_work = nlp_solver->work;
    ocp_nlp_memory *nlp_mem = rti_mem->nlp_mem;
    ocp_nlp_workspace *nlp_work = rti_work->nlp_work;
    ocp_nlp_timings *timings = nlp_mem->nlp_timings;

    acados_timer timer1;
    ocp_nlp_timings_reset(timings);

    acados_tic(&timer1);
    openmhe_approximate_qp_vectors_only(
        config, dims, nlp_in, nlp_out, nlp_opts, nlp_mem, nlp_work,
        stage0_full_hess);
    timings->time_lin += acados_toc(&timer1);

    acados_tic(&timer1);
    config->regularize->regularize(
        config->regularize, dims->regularize, nlp_opts->regularize,
        nlp_mem->regularize_mem);
    timings->time_reg += acados_toc(&timer1);

    const bool precondensed_lhs = (*lhs_valid) && !stage0_full_hess;
    /*
     * When dynamic arrival refreshed stage-0 ``W``, ``lhs_valid`` is cleared
     * inside this function so the next full solve rebuilds the condensed LHS.
     */
    const int qp_status = ocp_nlp_solve_qp_and_correct_dual(
        config, dims, nlp_opts, nlp_mem, nlp_work, precondensed_lhs, NULL, NULL,
        NULL, NULL, NULL);

    if ((qp_status != ACADOS_SUCCESS) && (qp_status != ACADOS_MAXITER)) {
        *lhs_valid = 0;
        nlp_mem->status = ACADOS_QP_FAILURE;
        return nlp_mem->status;
    }

    double step_size;
    acados_tic(&timer1);
    const int globalization_status = config->globalization->find_acceptable_iterate(
        config, dims, nlp_in, nlp_out, nlp_mem, rti_mem, nlp_work, nlp_opts,
        &step_size);
    timings->time_glob += acados_toc(&timer1);

    if (globalization_status != ACADOS_SUCCESS) {
        nlp_mem->status = globalization_status;
        return nlp_mem->status;
    }

    *lhs_valid = !stage0_full_hess;

    nlp_mem->status = ACADOS_SUCCESS;
    rti_mem->is_first_call = false;
    return nlp_mem->status;
}
