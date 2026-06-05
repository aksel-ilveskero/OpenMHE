# Examples

**OpenTorsion demo** — sliding-window MHE on a four-disk shaft line: known drive torque, unknown load torque estimated as a random-walk state.

From the repository root (with Acados built and `ACADOS_SOURCE_DIR` set):

```bash
pip install -e ".[opentorsion,demo]"
python examples/opentorsion_demo/opentorsion_demo.py
```

Writes `mhe_results.png` in the current working directory.
