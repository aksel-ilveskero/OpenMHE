# ICE state-estimation tutorial

Jupyter walkthrough for moving-horizon estimation on a **six-cylinder internal combustion engine** model built with OpenTorsion, plus a simple propeller load.

## Contents

| Path | Role |
|------|------|
| `ice_estimation.ipynb` | Full tutorial: model, simulation, MHE, plots |
| `ice_data.py` | Loads `ICE_data/*.csv` (path anchored to this folder) |
| `ICE_data/pressure_data.csv` | Normalized cylinder pressure vs crank angle |
| `ICE_data/peak_data.csv` | Peak pressure scale vs engine speed |

## Setup

```bash
pip install -e ".[opentorsion,demo]"
pip install jupyter ipykernel   # if needed
export ACADOS_SOURCE_DIR=/path/to/acados
ulimit -s unlimited              # recommended before running MHE in the notebook
```

Register the venv kernel once:

```bash
python -m ipykernel install --user --name=openmhe --display-name "OpenMHE (.venv)"
```

Open `ice_estimation.ipynb` from this directory (or the repo root) and run **Kernel → Restart**, then execute cells top to bottom.

The setup cell adds this folder to `sys.path` so `from ice_data import load_pressure_tables` resolves `ICE_data/` via `ice_data.py` (not relative to a fragile `/ICE_data` path).

## MHE problem (summary)

- **States** — minimal-coordinate shaft torques and speeds (21 plant states)
- **Known inputs** — six piston torques (`KnownInput` on channels 0–5)
- **Unknown input** — propeller load (`InputRandomWalk` on channel 6 with `lambda_u`)
- **Measurements** — two angular speeds (pulley and flywheel)
- **Solver** — `run_c_solver` with `EKFArrivalCost` (incremental filter in C; matches `run_solver`)

Zero the load column in `u` passed to the estimator (`u_mhe[:, 6] = 0`); keep the simulated load only for plotting ground truth.

## Notes

- **Load observability** is weak (speed-only sensors, load at the end of the shaft). The notebook discusses tuning and limitations.
- **`lambda_u`** on `InputRandomWalk` penalizes load *increments* in the augmented-state model; it competes with plant process noise `Q_w` — see tutorial text for tuning context.
- If the Jupyter kernel dies during the MHE cell, raise the stack limit (`ulimit -s unlimited`) and restart the kernel after changing horizon or objective (Acados codegen + C driver rebuild).
