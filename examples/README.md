# Examples

Runnable demos for sliding-window MHE with [OpenTorsion](https://github.com/tamasmak/opentorsion) shaft-line models and [Acados](https://docs.acados.org/).

## Prerequisites

From the repository root:

```bash
pip install -e ".[opentorsion,demo]"
```

Set Acados environment variables (see [main README](../README.md#requirements)):

```bash
export ACADOS_SOURCE_DIR=/path/to/acados
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
```

Before the first solver build:

```python
import openmhe as mhe
mhe.ensure_acados_environment()
```

For long horizons or the ICE notebook, increase the stack limit in the shell that launches Jupyter or Python:

```bash
ulimit -s unlimited
```

## Examples

| Directory | What it demonstrates |
|-----------|----------------------|
| [`opentorsion_4_disk/`](opentorsion_4_disk/) | Minimal four-disk shaft: known motor torque, random-walk load, EKF arrival cost, **C solver** + optional PI post-step |
| [`opentorsion_22_disk/`](opentorsion_22_disk/) | Larger 22-disk test-bench model on recorded data; torque + velocity sensors; load on input 1 |
| [`opentorsion_ic_engine/`](opentorsion_ic_engine/) | Six-cylinder ICE tutorial (Jupyter): piston torques from pressure tables, propeller load estimation |

### Quick commands

```bash
# Four-disk demo (matplotlib window)
python examples/opentorsion_4_disk/ot_4_disk.py

# 22-disk test bench (feather dataset, prints RMSE)
python examples/opentorsion_22_disk/ot_22_disk.py

# ICE tutorial (open ice_estimation.ipynb in Jupyter / VS Code)
jupyter notebook examples/opentorsion_ic_engine/ice_estimation.ipynb
```

## Python vs C sliding window

| API | When to use |
|-----|-------------|
| `mhe.run_solver(solver, y, u)` | Pure Python Acados loop; supports `UKFArrivalCost` and per-window `cost_set` updates |
| `mhe.run_c_solver(solver, y, u)` | Fast C driver (`openmhe/c_solver`); EKF arrival precomputed in Python; same return layout as `run_solver` |

Both expect `y` and `u` as `(channels, samples)`. Simulation arrays are usually `(samples, channels)` — transpose before calling.

After changing horizon, objective terms, or noise weights, call `build_mhe_solver` again. If you change only dimensions, pass `run_c_solver(..., rebuild=True)` or run `make -C openmhe/c_solver clean all`.

## Generated artifacts

Acados JSON and generated C land under `mhe_json/` and `c_generated_code/` at the [data root](../README.md#generated-artifacts) (repo root when developing in-tree). These are local build outputs, not part of the examples themselves.
