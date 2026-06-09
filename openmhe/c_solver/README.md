# C sliding-window MHE driver

Builds `libopenmhe_mhe_run.so` with `-O3 -march=native -ffast-math` (override via `OPT_CFLAGS`). The Makefile auto-detects the BLASFEO target from `$(ACADOS_DIR)/include/blasfeo/include/blasfeo_target.h` and passes `-D$(BLASFEO_TARGET_DEFINE)` so compile-time struct layouts match `libblasfeo` (override with `BLASFEO_TARGET_DEFINE=TARGET_...`).

## Prerequisites

1. Run `openmhe.build_mhe_solver(...)` so `c_generated_code/acados_solver_openmhe_mhe.h` exists.
2. Set `ACADOS_DIR` or install Acados where `../../../acados` resolves from this directory.

```bash
make -C openmhe/c_solver CODEGEN_DIR=/path/to/c_generated_code ACADOS_DIR=/path/to/acados
```

Rebuild after every codegen change (stack buffers in `run_loop.c` use `OPENMHE_MHE_*` from the generated header). `run_c_solver` rebuilds automatically when `acados_solver_openmhe_mhe.h` is newer than `run_loop.o`, or when called with `rebuild=True`.

If you see a **dimension mismatch** (`status -3` / stale `libopenmhe_mhe_run.so`):

1. Re-run `build_mhe_solver` with the new horizon or model.
2. Call `run_c_solver(..., rebuild=True)` or `make -C openmhe/c_solver clean all`.
3. In Jupyter, **restart the kernel** after changing problem size — Python may keep an old `.so` mapped in memory until the process exits.

## Profiling

Per-window setup / solve / warm-start timings (stderr), plus Acados
`time_lin` / `time_qp_sol` / `time_reg` / `time_glob` averages:

```bash
make -C openmhe/c_solver OPENMHE_PROFILE=1 ...
```

### LTI + LINEAR_LS fast path

When the solver is built with all-L2 penalties (`LINEAR_LS`), `run_c_solver(...,
lti_linear_ls_fast=True)` (default) reuses condensed QP factors after the first
window: dynamics Jacobians and stage Hessians are skipped, only QP vectors are
refreshed. Disable with `lti_linear_ls_fast=False`. Requires
`build_mhe_solver(..., lti_linear_ls_fast=True)` (default for `LINEAR_LS`) and
`nlp_solver_type='SQP_RTI'`; otherwise a warning is emitted and the full solve is used.
Constant `A`, `B`, `Vx`, `Vu`, `W` are exported to `c_generated_code/openmhe_mhe_extra.{h,c}`.

## Unit tests (filters, linear algebra)

```bash
make -C openmhe/c_solver test
./openmhe/c_solver/test/test_filter
./openmhe/c_solver/test/test_inv
```

## Stack usage

The hot loop allocates workspace on the stack from generated dimensions (`NY0`, `NX`, `N`, …). Warm-start shifting uses `ocp_nlp_get_all` / `memmove` / `ocp_nlp_set_all` on dense buffers sized `(N+1)*NX` and `N*NU`. Large horizons or arrival costs (e.g. `NY0` in the hundreds) can require increasing the thread stack limit (`ulimit -s unlimited`).

Known-input bounds (`lbu`/`ubu`) are applied only when the pinned values change at that stage.

All shooting stages share the same `nx` and `nu` (standard for our MHE models), which is what the bulk warm-start `memmove` relies on.

## Python entry point

`openmhe.run_c_solver(solver, y, u, post_steps=...)` builds the library if needed and runs the C sliding-window loop. Each call allocates an Acados capsule, runs `openmhe_mhe_init_solver` → `openmhe_mhe_run_sliding` → `openmhe_mhe_free_solver`. Signature and return values match `run_solver` (`u_hat`, `x_hat` with shape `(nu, n_est)` and `(nx_base, n_est)`).

Dynamic arrival (`EKFArrivalCost`, etc.) is precomputed in Python (`x_bar`, `W0` per window); the C driver applies stage-0 `yref` / `W` and runs Acados solves. Stage cost weights `W` are fixed at codegen time — change `lambda_u` or noise covariances, then call `build_mhe_solver` again.

Use `run_solver()` for `UKFArrivalCost` until in-C UKF support is restored.

Optional: `post_steps` run in Python after the C loop (same `WindowStep` semantics as `run_solver`).
