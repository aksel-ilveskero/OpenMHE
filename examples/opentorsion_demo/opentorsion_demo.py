"""OpenTorsion shaft-line demo: estimate unknown load torque with sliding-window MHE."""

import time

import numpy as np
import matplotlib.pyplot as plt

from test_bench import TestBench

import openmhe as mhe

W_COV = 0.001
V_COV = 0.05
# Inverse penalty strength on load random walk (larger => smoother load estimate).
LOAD_LAMBDA = 5


class PILoadObserver:
    """PI (disturbance) observer that refines the load-torque estimate.

    Classic proportional-integral observer on the disk-4 rotational balance.
    An internal velocity estimate ``omega4_hat`` is propagated by the model and
    pulled toward the MHE velocity by a proportional gain; an integral term on
    the same innovation reconstructs the unknown load torque::

        e          = omega4_mhe - omega4_hat                 (innovation)
        omega4_hat += dt/J4 * (T_shaft3 + T_load_hat) + kp*e (model + correction)
        T_load_hat += ki * e                                 (disturbance update)

    Because the velocity estimate is *propagated*, not overwritten, the loop
    low-pass filters the MHE state estimates instead of differentiating them.

    State layout (OpenTorsion 4-disk minimal representation, 7 states):
        x[0] = T_shaft1   x[1] = T_shaft2   x[2] = T_shaft3
        x[3] = omega_1    x[4] = omega_2    x[5] = omega_3   x[6] = omega_4

    Shaft-3 torque: x[2]  (directly)
    Disk-4 velocity: x[6]
    Disk-4 balance: J4 * d(omega4)/dt = T_shaft3 + T_load
    (The OpenTorsion convention has T_load acting in the same direction as
    positive rotation; opposing loads are represented as negative values.)

    Tuning
    ------
    The error dynamics are governed by
        M = [[1 - kp, dt/J4], [-ki, 1]]
    Place both poles at ``r`` (well-damped) via::
        kp = 2 * (1 - r)
        ki = (1 - r)**2 * J4 / dt
    Smaller ``r`` (further from 1) converges faster but filters less.
    """

    def __init__(
        self,
        J4: float,
        load_input_idx: int,
        kp: float,
        ki: float,
    ):
        """
        Parameters
        ----------
        J4 : float
            Inertia of disk 4.
        load_input_idx : int
            Column index of the load torque in ``u_hat`` (typically ``1``).
        kp : float
            Proportional (velocity-error) observer gain.
        ki : float
            Integral (load-estimate) observer gain.
        """
        self.J4 = J4
        self.load_input_idx = load_input_idx
        self.kp = kp
        self.ki = ki
        self._omega4: float = 0.0
        self._T_load_hat: float = 0.0
        self._initialized: bool = False
        self.u_load_mhe: list[float] = []

    def reset(self) -> None:
        """Clear filter state before a new run_solver pass."""
        self._omega4 = 0.0
        self._T_load_hat = 0.0
        self._initialized = False
        self.u_load_mhe = []

    def __call__(self, ctx: mhe.WindowStep) -> None:
        """Refine the load estimate for the current window.

        Reads shaft-3 torque (x[2]) and disk-4 velocity (x[6]) from
        ``ctx.x_hat[:, ctx.idx]`` and overwrites
        ``ctx.u_hat[load_input_idx, ctx.idx]`` with the PI-corrected load estimate.
        """
        x = ctx.x_hat[:, ctx.idx]
        dt = ctx.dt

        # Shaft-3 torque and disk-4 velocity are direct states
        T_shaft3 = x[2]
        omega4_mhe = x[6]

        self.u_load_mhe.append(float(ctx.u_hat[self.load_input_idx, ctx.idx]))

        if not self._initialized:
            self._omega4 = omega4_mhe
            self._T_load_hat = float(ctx.u_hat[self.load_input_idx, ctx.idx])
            self._initialized = True

        # Innovation: how far the filtered velocity is from the MHE velocity
        e = omega4_mhe - self._omega4

        # Propagate the velocity model (OpenTorsion convention, +T_load) with a
        # proportional correction toward the measurement.
        self._omega4 += dt * (T_shaft3 + self._T_load_hat) / self.J4 + self.kp * e

        # Integral action reconstructs the unknown load torque.
        self._T_load_hat += self.ki * e

        ctx.u_hat[self.load_input_idx, ctx.idx] = self._T_load_hat


def simulate(test_bench: TestBench, dt: float, t_end: float):
    """Generate motor/load inputs and simulate the OpenTorsion plant with noise."""
    t = np.arange(0, t_end, dt)

    u = np.zeros((test_bench.B.shape[1], t.shape[0]))
    u[0, :] = 1.0
    u[1, :] = 4*np.sin(2 * np.pi * t) + np.sin((4 * np.pi * t) + np.pi/2) + np.random.normal(0, 0.5, t.shape[0])

    w_cov = W_COV * np.eye(test_bench.A.shape[0])
    v_cov = V_COV * np.eye(test_bench.C.shape[0])
    x, y = test_bench.simulate(dt, u, w_cov, v_cov)

    return t, x, y, u


def main():
    """Build MHE, run sliding windows, plot estimates, and print LaTeX objective."""
    np.random.seed(0)  # reproducible noise so tuning comparisons are meaningful
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

    mhe_objective = mhe.ObjectiveBuilder()
    meas_cov, proc_cov = V_COV, W_COV

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
        mhe.InputRandomWalk(target_idx=[1], lambda_u=LOAD_LAMBDA),
    )

    mhe_objective.add(mhe.KnownInput([0]))
    mhe_objective.add(mhe.UKFArrivalCost(mhe_system, builder=mhe_objective))

    # PI observer: reads shaft-3 torque (x[2]) and disk-4 velocity (x[6])
    # directly from the MHE state to reconstruct the load on disk 4.
    # Gains placed for a double error-pole at r=0.9 (well-damped, light
    # filtering): kp = 2*(1-r) = 0.2, ki = (1-r)^2 * J4/dt = 1.0.
    r_pole = 0.975
    J4 = test_bench.inertias[3]
    pi_observer = PILoadObserver(
        J4=J4,
        load_input_idx=1,
        kp=2.0 * (1.0 - r_pole),
        ki=(1.0 - r_pole) ** 2 * J4 / dt,
    )

    print("Compiling ACADOS solver...")
    solver = mhe.build_mhe_solver(
        mhe_system,
        n_window,
        mhe_objective,
        dt=dt,
        already_discrete=True,
    )

    print("Running MHE...")
    t_mhe_start = time.perf_counter()
    u_hat, x_hat = mhe.run_solver(solver, y, u, post_steps=[pi_observer])
    t_mhe_elapsed = time.perf_counter() - t_mhe_start
    print(f"MHE solver time: {t_mhe_elapsed:.3f} s")

    # run_solver reports u/x at the last stage of each window (time index k-1).
    t_est = t[n_window - 1 : n_window - 1 + u_hat.shape[1]]
    u_load_true = u[1, n_window - 1 : n_window - 1 + u_hat.shape[1]]
    u_load_mhe = np.asarray(pi_observer.u_load_mhe)
    u_load_pi = u_hat[1, :]
    load_rmse_mhe = np.sqrt(np.nanmean((u_load_mhe - u_load_true) ** 2))
    load_rmse_pi = np.sqrt(np.nanmean((u_load_pi - u_load_true) ** 2))
    print(f"Load RMSE (MHE): {load_rmse_mhe:.4f}")
    print(f"Load RMSE (MHE + PI): {load_rmse_pi:.4f}")

    state_indices = [0, 2, 4]
    zoom = (5.0, 5.2)
    zoom_mask = (t_est >= zoom[0]) & (t_est <= zoom[1])

    state_series = [
        (
            f"State x[{si}]",
            x_ref[si, n_window - 1 : n_window - 1 + x_hat.shape[1]],
            x_hat[si, :],
        )
        for si in state_indices
    ]
    n_panels = 1 + len(state_series)
    fig, axes = plt.subplots(
        n_panels, 2, figsize=(14, 2.2 * n_panels), sharex="col", width_ratios=[1.4, 1]
    )
    if n_panels == 1:
        axes = np.array([axes])

    for col, (t_lo, t_hi, subtitle) in enumerate(
        [
            (t_est[0], t_est[-1], "Full window"),
            (zoom[0], zoom[1], f"Zoom {zoom[0]:g}–{zoom[1]:g} s"),
        ]
    ):
        ax = axes[0, col]
        ax.plot(t_est, u_load_true, "k-", label="True")
        ax.plot(t_est, u_load_mhe, "C0--", linewidth=1.5, label="MHE", alpha=0.8)
        ax.plot(t_est, u_load_pi, "b-", linewidth=1.2, label="MHE + PI", alpha=0.8)
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylabel("Load torque")
        if col == 0:
            rmse_mhe = np.sqrt(np.nanmean((u_load_mhe - u_load_true) ** 2))
            rmse_pi = np.sqrt(np.nanmean((u_load_pi - u_load_true) ** 2))
        else:
            rmse_mhe = np.sqrt(
                np.nanmean((u_load_mhe[zoom_mask] - u_load_true[zoom_mask]) ** 2)
            )
            rmse_pi = np.sqrt(
                np.nanmean((u_load_pi[zoom_mask] - u_load_true[zoom_mask]) ** 2)
            )
        ax.set_title(
            f"{subtitle}  |  RMSE MHE = {rmse_mhe:.4f}, MHE+PI = {rmse_pi:.4f}",
            fontsize=9,
            loc="left",
        )
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(loc="upper right", fontsize=8)

    for row, (ylabel, true, est) in enumerate(state_series, start=1):
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

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("MHE + PI observer: load torque and selected states", y=1.002)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
