#ifndef OPENMHE_LTI_FAST_H_
#define OPENMHE_LTI_FAST_H_

#include "acados_solver_openmhe_mhe.h"
#include "run_loop.h"

#ifdef __cplusplus
extern "C" {
#endif

/** True when the generated Acados plan uses LINEAR_LS at every stage. */
int openmhe_mhe_is_linear_ls_plan(openmhe_mhe_solver_capsule *capsule);

/**
 * Vector-only SQP-RTI step for LTI + LINEAR_LS: skip dynamics Jacobians and
 * stage Hessians (except stage 0 when ``stage0_full_hess``).
 *
 * ``lhs_valid`` is set to 0 when the condensed LHS must be rebuilt (stage-0
 * Hessian change); otherwise left unchanged.  Returns Acados status.
 */
int openmhe_mhe_solve_lti_fast(
    openmhe_mhe_solver_capsule *capsule,
    int stage0_full_hess,
    int *lhs_valid);

#ifdef __cplusplus
}
#endif

#endif /* OPENMHE_LTI_FAST_H_ */
