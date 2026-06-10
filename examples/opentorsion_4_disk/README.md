# Four-disk OpenTorsion demo

Sliding-window MHE on a minimal four-disk shaft line:

- **Input 0** — known motor torque (`KnownInput`, pinned from `u`)
- **Input 1** — unknown load torque (`InputRandomWalk`; or `UnknownInput` for an unpenalized estimate)
- **Measurements** — shaft torque and disk velocity (from `TestBench`)
- **Arrival cost** — `EKFArrivalCost` (incremental filter in `run_c_solver`)
- **Solver** — `run_c_solver` (compiled C loop)
- **Post-step** — optional `PILoadObserver` refines the load estimate after each window

## Run

From the repository root:

```bash
pip install -e ".[opentorsion,demo]"
export ACADOS_SOURCE_DIR=/path/to/acados
python examples/opentorsion_4_disk/ot_4_disk.py
```

The script simulates 10 s of data, builds the Acados problem, runs MHE, prints load RMSE (raw MHE vs MHE+PI), and opens comparison plots.

## Files

| File | Role |
|------|------|
| `ot_4_disk.py` | Simulation, MHE setup, PI observer, plotting |
| `test_bench.py` | OpenTorsion assembly, discrete simulation, noise |

## Tuning knobs (top of `ot_4_disk.py`)

| Symbol | Default | Meaning |
|--------|---------|---------|
| `W_COV` | `0.001` | Process noise variance (plant `w`) |
| `V_COV` | `0.05` | Measurement noise variance |
| `LOAD_LAMBDA` | `5` | Random-walk weight on estimated load (`InputRandomWalk`; omit term if using `UnknownInput`) |
| `n_window` | `15` | MHE horizon |
