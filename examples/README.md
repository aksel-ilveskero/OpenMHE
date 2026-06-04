# Examples

## OpenTorsion shaft-line demo

[`opentorsion_demo/`](opentorsion_demo/) — sliding-window MHE on a four-disk shaft: known motor torque, unknown load torque estimated with `InputRandomWalk`.

From the repository root:

```bash
pip install -e ".[opentorsion,demo]"
export ACADOS_SOURCE_DIR=/path/to/acados   # if not already set
python examples/opentorsion_demo/opentorsion_demo.py
```

Outputs (current working directory):

- `mhe_results.png` — load torque and selected states
- `load_torque_filtfilt.png` — optional low-pass post-processing of the load estimate
