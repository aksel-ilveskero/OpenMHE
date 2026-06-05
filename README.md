# OpenMHE

Moving horizon estimation (MHE) on [Acados](https://docs.acados.org/) with composable Gauss-Newton cost terms: measurement fit, process noise, known inputs, input tracking, and input regularization (random walk, first/second difference).

## Features

- **Composable objectives** — stack `MeasurementTerm`, `ProcessTerm`, `KnownInput`, `InputTrackingTerm`, and regulators in an `ObjectiveBuilder`
- **Robust penalties** — L2 (fast `LINEAR_LS`), L1, Huber, and dead-zone ( `CONVEX_OVER_NONLINEAR` )
- **Unknown inputs** — model loads or biases as random-walk augmented states (`InputRandomWalk`)
- **Sliding windows** — `run_solver` with optional arrival cost and regulator state seeding from prior windows
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

## Run the demo

From the repository root:

```bash
pip install -e ".[opentorsion,demo]"
python examples/opentorsion_demo/opentorsion_demo.py
```

Writes `mhe_results.png` and `load_torque_filtfilt.png` in the current directory. See [examples/opentorsion_demo/README.md](examples/opentorsion_demo/README.md).

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
| `examples/opentorsion_demo/` | OpenTorsion shaft-line MHE demo |
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
| `InputFirstDiffReg` | Penalize `u_k - u_{k-1}` (`lambda_u` = inverse penalty strength) |
| `InputSecondDiffReg` | Penalize `u_k - 2 u_{k-1} + u_{k-2}` on controlled inputs |
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
