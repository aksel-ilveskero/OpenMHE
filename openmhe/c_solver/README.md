# C sliding-window MHE driver

Builds `libopenmhe_mhe_run.so` with `-O3 -march=native -ffast-math` (override via `OPT_CFLAGS`). The Makefile auto-detects the BLASFEO target from `$(ACADOS_DIR)/include/blasfeo/include/blasfeo_target.h` and passes `-D$(BLASFEO_TARGET_DEFINE)` so compile-time struct layouts match `libblasfeo` (override with `BLASFEO_TARGET_DEFINE=TARGET_...`).

## Prerequisites

1. Run `openmhe.build_mhe_solver(...)` so `c_generated_code/acados_solver_openmhe_mhe.h` exists.
2. Set `ACADOS_DIR` or install Acados where `../../../acados` resolves from this directory.

```bash
make -C openmhe/c_solver CODEGEN_DIR=/path/to/c_generated_code ACADOS_DIR=/path/to/acados
```

Rebuild after every codegen change (stack buffers in `src/run_loop.c` use `OPENMHE_MHE_*` from the generated header). `run_c_solver` rebuilds automatically when `acados_solver_openmhe_mhe.h` is newer than `build/run_loop.o`, or when called with `rebuild=True`.

If you see a **dimension mismatch** (`status -3` / stale `libopenmhe_mhe_run.so`):

1. Re-run `build_mhe_solver` with the new horizon or model.
2. Call `run_c_solver(..., rebuild=True)` or `make -C openmhe/c_solver clean all`.
3. In Jupyter, **restart the kernel** after changing problem size — Python may keep an old `.so` mapped in memory until the process exits.

## Source layout

| Path | Role |
|------|------|
| `src/run_loop.c` | Sliding-window driver: yref/pin setup, warm-start shift, per-window solve dispatch |
| `include/run_loop.h` | Public API for the sliding-window driver |
| `src/lti_fast.c` / `include/lti_fast.h` | LTI + `LINEAR_LS` vector-only SQP-RTI step |
| `include/profile.h` | Optional per-window timing (`OPENMHE_PROFILE=1`) |
| `src/filter_arrival.c` / `include/filter_arrival.h` | EKF/UKF arrival filter, null-space-aware `P^{-1}`, Cholesky utilities |
| `build/` | Object files, `libopenmhe_mhe_run.so`, and unit-test binaries (generated) |

## Profiling

Per-window setup / solve / warm-start timings (stderr), plus Acados
`time_lin` / `time_qp_sol` / `time_reg` / `time_glob` averages:

```bash
make -C openmhe/c_solver OPENMHE_PROFILE=1 ...
```

## LTI + LINEAR_LS fast path

For **linear** plants with **all-L2** penalties, the Gauss-Newton QP Hessian and
dynamics coupling are constant across sliding windows.  After one full SQP-RTI
preparation step, later windows can skip Jacobian / Hessian assembly and reuse
**precondensed QP factors**, updating only the right-hand side (references,
gradients, bound changes from `KnownInput` pins).

### Requirements

| Requirement | Reason |
|-------------|--------|
| All terms use `L2Penalty` (`LINEAR_LS` mode) | Constant stage Hessians |
| LTI dynamics (fixed `A`, `B` at codegen) | Constant dynamics Jacobians |
| `nlp_solver_type='SQP_RTI'` | Fast path calls RTI vector assembly + `precondensed_lhs` QP solve |
| `lti_linear_ls_fast=True` (default for LS builds) | Opt-in flag on solver and `run_c_solver` |

Non-RTI solvers (`SQP`) or `CONVEX_OVER_NONLINEAR` costs disable the fast path.
Python emits a `UserWarning` and the C driver uses a full `openmhe_mhe_acados_solve`
every window.

### Python API

```python
solver = mhe.build_mhe_solver(
    model, N_horizon, builder, dt=dt,
    nlp_solver_type="SQP_RTI",      # required for fast path
    lti_linear_ls_fast=True,        # default when cost is LINEAR_LS
)
u_hat, x_hat = mhe.run_c_solver(
    solver, y, u,
    lti_linear_ls_fast=True,        # default: solver._lti_linear_ls_fast
    rebuild=False,
)
```

Disable explicitly with `lti_linear_ls_fast=False` on `build_mhe_solver` or
`run_c_solver`.

### Per-window behaviour

```
Window 0:  full openmhe_mhe_acados_solve  →  condense_qp_lhs  →  lhs_valid = 1
Window k>0 (fast):  openmhe_mhe_solve_lti_fast  →  vectors only  →  solve_qp(precondensed_lhs)
```

`openmhe_mhe_solve_lti_fast` (`lti_fast.c`):

1. `update_qp_matrices` on dynamics, cost, constraints (still needed for `KnownInput` pins and changing `yref`).
2. Skip cost Hessians except stage 0 when dynamic arrival updates `W0`.
3. Levenberg–Marquardt term, `ocp_nlp_approximate_qp_vectors_sqp`, QP solve with reused LHS.
4. RTI globalization.

On QP failure or a bad status, `use_fast` is cleared and remaining windows fall back to the full solve.

### Dynamic EKF/UKF arrival

When `EKFArrivalCost` or `UKFArrivalCost` (LTI plant only) supplies a
time-varying stage-0 weight `W0`, the C driver runs an incremental filter each
window (`filter_prior` → invert arrival block → set stage-0 `yref`/`W` → solve
→ `filter_assimilate`).  Null-space directions in `P` (strict kinematics /
sparse process noise) are zeroed in the weight, matching Python
`invert_arrival_covariance`.  Each window refreshes the stage-0 Hessian
(`stage0_full_hess=1`).  The condensed LHS is invalidated for that window but
rebuilt on the next full solve.  Fast steps between windows still skip stages
`1…N` Hessians.

Custom nonlinear `f`/`h` on `UKFArrivalCost` are not supported in C; use
`run_solver()` for those cases.

### Codegen export (`openmhe_mhe_extra.{h,c}`)

`build_mhe_solver` writes constant augmented matrices to `c_generated_code/`:

- `OPENMHE_MHE_A`, `OPENMHE_MHE_B` — discrete augmented dynamics
- `OPENMHE_MHE_Vx`, `OPENMHE_MHE_Vu` — measurement / residual Jacobians
- `OPENMHE_MHE_W` — stage cost weight

These support validation tooling and a future standalone RHS assembler; the
current fast path still goes through Acados NLP submodules.

### Tests

```bash
pytest tests/test_lti_fast.py -q
pytest tests/test_c_runner_parity.py -q   # EKF/UKF C vs Python
pytest tests/test_input_partition.py -q   # KnownInput, UnknownInput, RW partition
```

Compares fast vs full C solve and vs Python `run_solver` on tiny LTI problems,
including `KnownInput`, EKF arrival, and non-RTI fallback.

## Unit tests (filters, linear algebra)

```bash
make -C openmhe/c_solver test
./openmhe/c_solver/build/test_filter
./openmhe/c_solver/build/test_inv
./openmhe/c_solver/build/test_arrival_inv
```

## Stack usage

The hot loop allocates workspace on the stack from generated dimensions (`NY0`, `NX`, `N`, …). Warm-start shifting uses `ocp_nlp_get_all` / `memmove` / `ocp_nlp_set_all` on dense buffers sized `(N+1)*NX` and `N*NU`. Large horizons or arrival costs (e.g. `NY0` in the hundreds) can require increasing the thread stack limit (`ulimit -s unlimited`).

Known-input bounds (`lbu`/`ubu`) are applied only when the pinned values change at that stage.

All shooting stages share the same `nx` and `nu` (standard for our MHE models), which is what the bulk warm-start `memmove` relies on.

## Python entry point

`openmhe.run_c_solver(solver, y, u, post_steps=...)` builds the library if needed and runs the C sliding-window loop. Each call allocates an Acados capsule, runs `openmhe_mhe_init_solver` → `openmhe_mhe_run_sliding` → `openmhe_mhe_free_solver`. Signature and return values match `run_solver` (`u_hat`, `x_hat` with shape `(nu, n_est)` and `(nx_base, n_est)`).

Dynamic arrival (`EKFArrivalCost`, `UKFArrivalCost`) is computed in the C sliding
loop via an incremental filter; stage cost weights `W` (except the stage-0
arrival block) are fixed at codegen time — change `lambda_u` or noise
covariances, then call `build_mhe_solver` again.

Optional: `post_steps` run in Python after the C loop (same `WindowStep` semantics as `run_solver`).
