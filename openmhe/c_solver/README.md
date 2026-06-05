# C sliding-window MHE driver

Builds `libopenmhe_mhe_run.so` with `-O3 -march=native -ffast-math` (override via `OPT_CFLAGS`).

## Prerequisites

1. Run `openmhe.build_mhe_solver(...)` so `c_generated_code/acados_solver_openmhe_mhe.h` exists.
2. Set `ACADOS_DIR` or install Acados where `../../../acados` resolves from this directory.

```bash
make -C openmhe/c_solver CODEGEN_DIR=/path/to/c_generated_code ACADOS_DIR=/path/to/acados
```

Rebuild after every codegen change (stack buffers in `run_loop.c` use `OPENMHE_MHE_*` from the generated header). `run_c_solver` rebuilds automatically when `acados_solver_openmhe_mhe.h` is newer than `run_loop.o`; if you still see a dimension mismatch error, run `make -C openmhe/c_solver clean all`.

## Profiling

Per-window setup / solve / warm-start timings (stderr):

```bash
make -C openmhe/c_solver OPENMHE_PROFILE=1 ...
```

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

`openmhe.run_c_solver(solver, y, u)` builds the library if needed and runs the C sliding-window loop. Dynamic arrival (`EKFArrivalCost`, etc.) is precomputed in Python (`x_bar`, `W0` per window); the C driver only applies `yref` / `W` updates and Acados solves. Use `run_solver()` for UKF arrival until in-C support is restored.
