"""OpenTorsion shaft-line demo: estimate unknown load torque with sliding-window MHE."""

import time

import numpy as np
import scipy.linalg as sla
import scipy.signal as signal
import matplotlib.pyplot as plt

from test_bench import TestBench

import openmhe as mhe

W_COV = 0.001
V_COV = 0.2
# Inverse penalty strength on load random walk (larger => smoother load estimate).
LOAD_RW_LAMBDA = 10.0

# Small cov => STRONG tracking (weight ~ 1/cov), not "no penalty".
MOTOR_TRACK_COV = 1e-8
USE_MOTOR_TRACKING = False


def simulate(test_bench: TestBench, dt: float, t_end: float):
    """Generate motor/load inputs and simulate the OpenTorsion plant with noise."""
    t = np.arange(0, t_end, dt)

    u = np.zeros((test_bench.B.shape[1], t.shape[0]))
    u[0, :] = 1.0
    u[1, :] = 4*np.sin(2 * np.pi * t) + np.sin((4 * np.pi * t) + np.pi/2) + np.random.normal(0, 0.01, t.shape[0])

    w_cov = W_COV * np.eye(test_bench.A.shape[0])
    v_cov = V_COV * np.eye(test_bench.C.shape[0])
    x, y = test_bench.simulate(dt, u, w_cov, v_cov)

    return t, x, y, u


def steady_state_arrival_cov(A, B, C, D, dt, Q, R):
    """Discrete algebraic Riccati matrix for steady-state arrival cost."""
    A_d, B_d, C_d, D_d, _ = signal.cont2discrete((A, B, C, D), dt=dt)
    return sla.solve_discrete_are(A_d.T, C_d.T, Q, R)


def lowpass_filtfilt(x: np.ndarray, dt: float, cutoff_hz: float = 15.0, order: int = 4) -> np.ndarray:
    """Zero-phase low-pass (quick noise cleanup on load torque)."""
    fs = 1.0 / dt
    wn = min(cutoff_hz / (0.5 * fs), 0.99)
    b, a = signal.butter(order, wn, btype="low")
    return signal.filtfilt(b, a, np.asarray(x, dtype=float))


def main():
    """Build MHE, run sliding windows, plot estimates, and print LaTeX objective."""
    test_bench = TestBench()
    dt = 0.001
    t_end = 10.0
    t, x_ref, y, u = simulate(test_bench, dt, t_end)

    mhe_system = mhe.SystemModel.from_matrices(
        test_bench.A,
        test_bench.B,
        test_bench.C,
        test_bench.D,
        is_discrete=False,
        dt=dt,
    )

    n_window = 50
    ny, nx, nu = mhe_system.ny, mhe_system.nx, mhe_system.nu

    Q = W_COV * np.eye(nx)
    R = V_COV * np.eye(ny)
    P_arrival = steady_state_arrival_cov(
        test_bench.A, test_bench.B, test_bench.C, test_bench.D, dt, Q, R
    )

    mhe_objective = mhe.ObjectiveBuilder()
    meas_cov, proc_cov = V_COV, W_COV
    P_use = P_arrival

    mhe_objective.add(
        mhe.MeasurementTerm(
            penalty=mhe.L2Penalty(),
            weight=mhe.NoiseWeight(dim=ny, cov=meas_cov),
        )
    )
    mhe_objective.add(
        mhe.ProcessTerm(
            penalty=mhe.L2Penalty(),
            weight=mhe.NoiseWeight(dim=nx, cov=proc_cov),
        )
    )
    mhe_objective.add(
        mhe.InputRandomWalk(target_idx=[1], lambda_u=LOAD_RW_LAMBDA),
    )

    if USE_MOTOR_TRACKING:
        mhe_objective.add(
            mhe.InputTrackingTerm(
                [0],
                mhe.NoiseWeight(dim=1, cov=MOTOR_TRACK_COV),
                reference="measured",
            ),
        )
    else:
        mhe_objective.add(mhe.KnownInput([0]))

    print("Compiling ACADOS solver...")
    solver = mhe.build_mhe_solver(
        mhe_system.A,
        mhe_system.B,
        mhe_system.C,
        mhe_system.D,
        n_window,
        mhe_objective,
        dt=dt,
        P_arrival=P_use,
        already_discrete=True,
    )

    print("Running MHE...")
    t_mhe_start = time.perf_counter()
    u_hat, x_hat = mhe.run_solver(solver, y, u)
    t_mhe_elapsed = time.perf_counter() - t_mhe_start
    print(f"MHE solver time: {t_mhe_elapsed:.3f} s")



    # run_solver reports u/x at the last stage of each window (time index k-1).
    t_est = t[n_window - 1 : n_window - 1 + u_hat.shape[1]]
    u_load_true = u[1, n_window - 1 : n_window - 1 + u_hat.shape[1]]
    u_load_est = u_hat[1, :]
    load_rmse = np.sqrt(np.nanmean((u_load_est - u_load_true) ** 2))

    state_indices = [0, 2, 4]
    zoom = (5.0, 5.2)
    zoom_mask = (t_est >= zoom[0]) & (t_est <= zoom[1])

    series = [("Load torque", u_load_true, u_load_est)]
    for si in state_indices:
        series.append(
            (
                f"State x[{si}]",
                x_ref[si, n_window - 1 : n_window - 1 + x_hat.shape[1]],
                x_hat[si, :],
            )
        )

    n_panels = len(series)
    fig, axes = plt.subplots(
        n_panels, 2, figsize=(14, 2.2 * n_panels), sharex="col", width_ratios=[1.4, 1]
    )
    if n_panels == 1:
        axes = np.array([axes])

    for row, (ylabel, true, est) in enumerate(series):
        full_rmse = np.sqrt(np.nanmean((est - true) ** 2))
        zoom_rmse = np.sqrt(np.nanmean((est[zoom_mask] - true[zoom_mask]) ** 2))

        for col, (t_lo, t_hi, rmse, subtitle) in enumerate(
            [
                (t_est[0], t_est[-1], full_rmse, "Full window"),
                (zoom[0], zoom[1], zoom_rmse, f"Zoom {zoom[0]:g}–{zoom[1]:g} s"),
            ]
        ):
            ax = axes[row, col]
            ax.plot(t_est, true, "k-", label="True")
            ax.plot(t_est, est, "b--", linewidth=1.5, label="MHE", alpha=0.7)
            ax.set_xlim(t_lo, t_hi)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{subtitle}  |  RMSE = {rmse:.4f}", fontsize=9, loc="left")
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("MHE: load torque and selected states", y=1.002)
    fig.tight_layout()
    fig.savefig("mhe_results.png", dpi=150, bbox_inches="tight")
    print(f"Saved mhe_results.png  |  load RMSE = {load_rmse:.4f}")

    valid = np.isfinite(u_load_est)
    u_load_filt = lowpass_filtfilt(u_load_est[valid], dt, cutoff_hz=5.0, order=4)
    t_filt = t_est[valid]
    filt_rmse = np.sqrt(np.nanmean((u_load_filt - u_load_true[valid]) ** 2))

    fig_filt, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t_est, u_load_true, "k-", label="True load", linewidth=1.2)
    ax.plot(t_est, u_load_est, color="C0", alpha=0.45, linewidth=1, label="MHE estimate")
    ax.plot(t_filt, u_load_filt, "r-", linewidth=1.5, label="MHE + filtfilt")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Load torque")
    ax.set_title(f"Load torque (filtfilt low-pass, fc=15 Hz)  |  RMSE = {filt_rmse:.4f}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig_filt.tight_layout()
    fig_filt.savefig("load_torque_filtfilt.png", dpi=150, bbox_inches="tight")
    print(f"Saved load_torque_filtfilt.png  |  filtered load RMSE = {filt_rmse:.4f}")

    objective_latex = mhe_objective.to_latex(underbrace=False, form="constrained")
    print(objective_latex)

if __name__ == "__main__":
    main()
