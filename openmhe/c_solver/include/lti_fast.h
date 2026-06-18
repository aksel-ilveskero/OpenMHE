/**
 * LTI + LINEAR_LS fast solve for sliding-window MHE.
 *
 * For linear time-invariant plants with Gauss-Newton L2 costs, the condensed
 * QP left-hand side (Hessian + dynamics coupling) is constant across windows.
 * After the first full SQP-RTI preparation step, later windows can refresh only
 * the QP vectors (gradients, bounds, references) and reuse precondensed factors.
 *
 * See ``openmhe/c_solver/README.md`` for requirements, fallback behaviour, and
 * interaction with dynamic EKF arrival costs.
 */
#ifndef OPENMHE_LTI_FAST_H_
#define OPENMHE_LTI_FAST_H_

#include "acados_solver_openmhe_mhe.h"
#include "run_loop.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Return 1 when every NLP stage uses ``LINEAR_LS`` in the generated Acados plan.
 *
 * Nonlinear (``CONVEX_OVER_NONLINEAR``) costs invalidate the constant-Hessian
 * assumption and disable the fast path at runtime.
 */
int openmhe_mhe_is_linear_ls_plan(openmhe_mhe_solver_capsule *capsule);

/**
 * One SQP-RTI iteration with vector-only QP assembly.
 *
 * Skips dynamics Jacobians and stage Hessians (except stage 0 when
 * ``stage0_full_hess`` is set).  Still runs ``update_qp_matrices`` on dynamics,
 * costs, and constraints so that ``KnownInput`` pins and changing references
 * propagate into the QP right-hand side.
 *
 * Parameters
 * ----------
 * capsule : generated Acados capsule for the current problem.
 * stage0_full_hess : non-zero when stage-0 ``W`` changes (dynamic arrival);
 *     forces a full Hessian at stage 0 and invalidates the condensed LHS.
 * lhs_valid : in/out flag.  Set to 0 on QP failure or when the LHS must be
 *     rebuilt; set to 1 after a successful fast step when the LHS remains valid.
 *
 * Returns Acados status (0 on success).
 */
int openmhe_mhe_solve_lti_fast(
    openmhe_mhe_solver_capsule *capsule,
    int stage0_full_hess,
    int *lhs_valid);

#ifdef __cplusplus
}
#endif

#endif /* OPENMHE_LTI_FAST_H_ */
