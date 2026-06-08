# 22-disk test-bench demo

MHE on a larger OpenTorsion assembly using **recorded test-bench data** (`data/testbench_evaluation_dataset.feather`).

- **Input 0** — motor torque (measured or estimated depending on sensor config)
- **Input 1** — propeller / load torque (`InputRandomWalk`)
- **Sensors** — configurable subset of motor input, shaft torque, and velocities (default: `torque1` + `velocity2`)
- **Arrival cost** — `EKFArrivalCost`
- **Solver** — `run_c_solver`

## Run

From the repository root:

```bash
pip install -e ".[opentorsion,demo]"
pip install pandas pyarrow   # feather I/O (not in openmhe extras)
export ACADOS_SOURCE_DIR=/path/to/acados
python examples/opentorsion_22_disk/ot_22_disk.py
```

The script loads the feather file, loops over `sensor_configs` (edit the list in `__main__`), prints MHE timing and RMSE for shaft torque and load, and shows plots.

## Data layout

`load_feather()` returns motor and propeller torque traces plus four measurement channels. The main block reorders columns to match the OpenTorsion state-space model before building `y`.

Pass only **known** inputs in `u` (motor row). Leave the load row zero so the EKF arrival filter does not see oracle load values.

## Files

| Path | Role |
|------|------|
| `ot_22_disk.py` | Model, MHE, evaluation plots |
| `data/testbench_evaluation_dataset.feather` | Recorded inputs and measurements |
