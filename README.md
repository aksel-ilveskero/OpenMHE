# OpenMHE

Moving horizon estimation (MHE) on [Acados](https://docs.acados.org/) with composable Gauss-Newton cost terms: measurement fit, process noise, known inputs, input tracking, and input regularization (random walk, first/second difference).

## Features

- **Composable objectives** — stack `MeasurementTerm`, `ProcessTerm`, `KnownInput`, `InputTrackingTerm`, and regulators in an `ObjectiveBuilder`
- **Robust penalties** — L2 (fast `LINEAR_LS`), L1, Huber, and dead-zone ( `CONVEX_OVER_NONLINEAR` )
- **Unknown inputs** — model loads or biases as random-walk augmented states (`InputRandomWalk`)
- **Sliding windows** — `run_solver` (Python) or `run_c_solver` (compiled C loop) with optional arrival cost and regulator state seeding
- **LTI fast solve** — for all-L2 linear problems, `run_c_solver` reuses condensed QP factors after the first window (~25% faster RTI on shaft-line demos; requires `SQP_RTI`)
- **LaTeX export** — paper-ready objective in substituted or constrained (`minimize … subject to`) form

## Requirements

- Python 3.10+
- [Acados](https://docs.acados.org/installation/) built locally (C libraries + Python `acados_template`)
- Environment (see [Acados getting started](https://docs.acados.org/getting_started/)):
  - `ACADOS_SOURCE_DIR` — path to your Acados source tree
  - `LD_LIBRARY_PATH` — include `$ACADOS_SOURCE_DIR/lib`

`acados_template` is installed from your Acados tree (editable install). It is not pinned in this package’s dependencies.

## Install

```bash
git clone https://github.com/your-org/OpenMHE.git
cd OpenMHE
pip install -e ".[opentorsion,demo]"
```

Optional extras:

| Extra | Packages | Use |
|-------|----------|-----|
| `opentorsion` | `opentorsion` | `SystemModel.from_opentorsion`, shaft demo |
| `demo` | `matplotlib` | Plotting in the example script |

Before building a solver, you can silence repeated template warnings:

```python
import openmhe as mhe
mhe.ensure_acados_environment()
```

## Quickstart

```python
import numpy as np
import openmhe as mhe

mhe.ensure_acados_environment()

model = mhe.SystemModel.from_matrices(A, B, C, D, is_discrete=False, dt=0.001)
ny, nx = model.ny, model.nx

obj = mhe.ObjectiveBuilder()
obj.add(mhe.MeasurementTerm(mhe.L2Penalty(), mhe.NoiseWeight(dim=ny, cov=0.01)))
obj.add(mhe.ProcessTerm(mhe.L2Penalty(), mhe.NoiseWeight(dim=nx, cov=1e-4)))
obj.add(mhe.KnownInput([0]))                      # motor: fixed from u in run_solver
obj.add(mhe.InputRandomWalk([1], lambda_u=1.0))   # load: estimated
obj.add(mhe.SteadyStateArrivalCost())

solver = mhe.build_mhe_solver(
    model,
    N_horizon=50,
    builder=obj,
    dt=0.001,
    already_discrete=True,
)

u_hat, x_hat = mhe.run_solver(solver, y, u)
```

`y` and `u` must be `(channels, samples)`. Transpose simulation arrays if they are stored as `(samples, channels)`.

### C solver (`run_c_solver`)

For production-style sliding windows, use the compiled driver in `openmhe/c_solver/`:

```python
solver = mhe.build_mhe_solver(
    model, N_horizon, builder, dt=dt,
    nlp_solver_type="SQP_RTI",   # required for the fast path below
)
u_hat, x_hat = mhe.run_c_solver(
    solver, y, u,
    lti_linear_ls_fast=True,     # default for all-L2 builds; disable to benchmark
    rebuild=False,
)
```

Same outputs as `run_solver`. EKF arrival is precomputed in Python; the C loop applies Acados solves. Rebuild after horizon or dimension changes (`rebuild=True` or `make -C openmhe/c_solver clean all`).

**LTI fast path** (default when every term uses `L2Penalty`):

| Window | What happens |
|--------|----------------|
| 0 | Full SQP-RTI solve; condensed QP left-hand side is factorised |
| 1… | Vector-only update + QP solve with reused factors |

Requires `nlp_solver_type='SQP_RTI'`. With `SQP` or robust (`CONL`) penalties, OpenMHE warns and uses a full solve every window. Dynamic `EKFArrivalCost` refreshes the stage-0 Hessian each window but still skips Hessians on later stages.

Details, profiling, and C source map: [openmhe/c_solver/README.md](openmhe/c_solver/README.md).

### Arrival cost

The stage-0 term `(x - x̄)ᵀ P⁻¹ (x - x̄)` ties each window to information outside the horizon. Attach a strategy with ``obj.add(...)``:

```python
obj.add(mhe.EKFArrivalCost(model, builder=obj))
solver = mhe.build_mhe_solver(model, 50, obj, dt=0.001)
```

| Class | Behavior |
|-------|----------|
| `SteadyStateArrivalCost` | Fixed DARE covariance; `x̄ = 0` |
| `EKFArrivalCost` | Discrete Kalman filter on base states; time-varying `P` and `x̄` |
| `UKFArrivalCost` | Unscented filter (LTI-compatible; optional nonlinear `f`/`h`) |

Alternatively pass a fixed matrix with `P_arrival=`. Dynamic arrival costs (`EKF`, `UKF`) require all-L2 penalties (`LINEAR_LS` mode). `run_solver` updates stage-0 weights each window when `P` changes.

### Input partition rule

Every column of `B` (each input index `0 … nu-1`) must be assigned **exactly once**:

| Role | API | Notes |
|------|-----|--------|
| Known | `KnownInput([i])` | Pinned to `u` passed to `run_solver` (equality bounds) |
| Soft known | `InputTrackingTerm([i], …)` | Penalize deviation from reference (`"measured"`, `"zero"`, or scalar) |
| Unknown / regulated | `InputRandomWalk`, `InputFirstDiffReg`, `InputSecondDiffReg` | Augmented states and/or smoothness priors |

Typical shaft line: `KnownInput([0])` or strong `InputTrackingTerm` on motor; `InputRandomWalk([1])` on load.

**NoiseWeight:** `cov` is variance; stored weight is `1/cov` (smaller `cov` → stronger penalty). `1e-8` is very strong tracking, not “no penalty”.

### Output alignment

`run_solver` returns estimates at the **last stage** of each window (time index `k-1` for a window ending at `k`). Compare to ground truth with:

```python
u[:, N - 1 : N - 1 + u_hat.shape[1]]
```

### Window post-steps

Inject arbitrary post-processing after each window solve via `post_steps=`:

```python
u_hat, x_hat = mhe.run_solver(solver, y, u, post_steps=[my_observer])
```

Each callable receives a `mhe.WindowStep` context and may overwrite
`ctx.u_hat[:, ctx.idx]` or `ctx.x_hat[:, ctx.idx]` in place.
The raw MHE output is always used for regulator-state seeding of the next
window, so post-step corrections appear only in the returned arrays and are
never fed back into the solver.

**`WindowStep` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `idx` | `int` | Column index into `x_hat` / `u_hat` for this window |
| `t_start` | `int` | Time index of the first stage (`k - N`) |
| `k` | `int` | Time index one past the last stage |
| `N` | `int` | Horizon length |
| `dt` | `float` | Sample period (seconds) |
| `nx_base` | `int` | Number of plant states (excludes augmented inputs) |
| `nu` | `int` | Total number of physical inputs |
| `x_hat` | `ndarray` | Full `(nx_base, n_est)` state array — **mutable** |
| `u_hat` | `ndarray` | Full `(nu, n_est)` input array — **mutable** |
| `x_full` | `ndarray` | Augmented state at the last horizon stage |
| `y` | `ndarray` | Raw measurement sequence `(ny, n_steps)` |
| `u` | `ndarray` | Raw known-input sequence `(nu, n_steps)` |

If a step object exposes a `reset()` method it is called once before the
window loop starts. Steps are skipped on failed solves (the column stays
`NaN`).

## Examples

| Example | Command |
|---------|---------|
| Four-disk shaft (intro) | `python examples/opentorsion_4_disk/ot_4_disk.py` |
| 22-disk test bench | `python examples/opentorsion_22_disk/ot_22_disk.py` |
| ICE tutorial (notebook) | `examples/opentorsion_ic_engine/ice_estimation.ipynb` |

Install extras and set `ACADOS_SOURCE_DIR` first. Overview: [examples/README.md](examples/README.md).

## Tests

```bash
pip install pytest
pytest tests/ -q
```

## Project layout

| Path | Purpose |
|------|---------|
| `openmhe/` | Installable package |
| `openmhe/mhe_strategies/` | Cost terms, penalties, weights |
| `openmhe/builder/` | Acados OCP assembly and sliding-window driver |
| `openmhe/export/` | LaTeX rendering |
| `openmhe/frontend/` | `SystemModel`, Acados runtime helpers |
| `openmhe/c_solver/` | Compiled sliding-window driver (`run_c_solver`, LTI fast path in `lti_fast.c`) |
| `examples/` | OpenTorsion shaft-line and ICE tutorials |
| `tests/` | Partition validation and LaTeX export tests |

### Generated artifacts

Acados JSON and generated C code live under a configurable data root:

1. `OPENMHE_DATA_DIR` if set
2. Repository root when developing in-tree (`pyproject.toml` found above the package)
3. Otherwise `./.openmhe/` in the current working directory

Subdirectories: `mhe_json/`, `c_generated_code/`.

```python
import openmhe
print(openmhe.get_data_root())
print(openmhe.get_mhe_json_dir())
```

Add `mhe_json/`, `c_generated_code/`, and local `*.png` to `.gitignore` before publishing if you do not want them in the repo.

## Objective terms

| Term | Role |
|------|------|
| `MeasurementTerm` | Weighted LS: measurements vs `C x + D u` |
| `ProcessTerm` | Penalty on process noise `w` in `x_{k+1} = A x_k + B u_k + w_k` |
| `KnownInput` | Known input fixed from `u` in `run_solver` (no cost row) |
| `InputTrackingTerm` | Soft tracking (`reference="measured"`, `"zero"`, or scalar) |
| `InputRandomWalk` | Unknown input as extra state; inferred from `y` and model |
| `InputFirstDiffReg` | Penalize `u_k - u_{k-1}` (`lambda_u` on regulator residual) |
| `InputSecondDiffReg` | Penalize `u_k - 2 u_{k-1} + u_{k-2}` on controlled inputs |
| `InputRandomWalk` | Unknown input as augmented state; `lambda_u` weights increment noise `w_u` in `ProcessTerm` |
| `InputRegTerm` | Deprecated; use `InputFirstDiffReg` / `InputSecondDiffReg` |
| `input_as_state` kwarg | Legacy random-walk indices (prefer `InputRandomWalk`) |

### Penalties and cost mode

| Penalty | Cost on residual `r` | Acados mode |
|---------|------------------------|-------------|
| `L2Penalty` | `0.5 rᵀ W r` | `LINEAR_LS` when **all** terms use L2 |
| `L1Penalty` | `Σ wᵢ \|rᵢ\|` (optional smoothed `ε`) | `CONVEX_OVER_NONLINEAR` |
| `HuberPenalty` | Element-wise Huber with `delta` | `CONVEX_OVER_NONLINEAR` |
| `DeadzonePenalty` | Zero inside zone, quadratic outside | `CONVEX_OVER_NONLINEAR` |

Mixed L2 + robust terms use **CONVEX_OVER_NONLINEAR**. Solver JSON files are prefixed `mhe_ls_` or `mhe_conl_`. Nonlinear penalties require **diagonal** `NoiseWeight` blocks.

## Export to LaTeX

```python
obj.add(mhe.EKFArrivalCost(model, builder=obj))
print(obj.to_latex(underbrace=True))
print(obj.to_latex(form="constrained"))
```

When an arrival cost has been added with ``obj.add(...)``, LaTeX export includes the
matching term (steady-state, EKF, or UKF notation).

`form="constrained"` writes the cost in terms of noise/defect variables (`v_k`, `w_k`, `δ_k`, `q^{(i)}_{u,k}`) and lists dynamics, measurement, and difference constraints under `subject to`.

| Argument | Effect |
|----------|--------|
| `underbrace` | Label each term (default `False`) |
| `multiline` | `aligned` block, one term per line |
| `environment` | Outer math environment (`"equation"`, `"align"`, or `""`) |
| `standalone` | Wrap in a minimal compilable document |
| `define_penalties` | Prepend Huber/dead-zone definitions |
| `symbols` | `LatexSymbols` overrides |
| `form` | `"substituted"` (default) or `"constrained"` |

## Publishing to GitHub

1. Update `Repository` in `pyproject.toml` to your fork URL.
2. Ensure Acados is **not** vendored; document the install steps above.
3. Run `pytest` and the demo once after cloning.
4. Optionally add a root `LICENSE` file (MIT, per `pyproject.toml`).

## License

MIT
