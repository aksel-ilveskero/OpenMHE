import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv
import opentorsion as ot
import pandas as pd
import openmhe as mhe


class TestBench:
    def __init__(self, measured_states):
        self.measured_states = measured_states
        self.assembly = self.create_opentorsion_system()
        self.A, self.B, self.C, self.D = self.assembly.state_space(
            minimal_representation=True
        )
        self.B = self.B[:, [0, -1]]
        self.C = self.C[self.measured_states, :]
        self.D = self.D[self.measured_states, :][:, [0, -1]]
    
    def create_opentorsion_system(self):

        # PARAMETERS
        I1 = 7.94e-4
        k1 = 1.90e5
        c1 = 8.08
        d1 = 0.0030

        I2 = 3.79e-6
        k2 = 6.95e3
        c2 = 0.29

        I3 = 3.00e-6
        k3 = 90.00
        c3 = 0.24

        I4 = 2.00e-6
        k4 = 90.00
        c4 = 0.24

        I5 = 7.81e-3
        k5 = 90.00
        c5 = 0.24

        I6 = 2.00e-6
        k6 = 90.00
        c6 = 0.24

        I7 = 3.29e-6
        k7 = 94.06
        c7 = 0.00

        I8 = 5.01e-5
        k8 = 4.19e4  
        c8 = 1.78

        I9 = 6.50e-6
        k9 = 5.40e3     
        c9 = 0.23

        I10 = 5.65e-5
        k10 = 4.19e4
        c10 = 1.78

        I11 = 4.27e-6
        k11 = 1.22e3
        c11 = 0.52

        I12 = 3.25e-4
        k12 = 4.33e4
        c12 = 1.84
        d12 = 0.0042

        I13 = 1.20e-4
        k13 = 3.10e4
        c13 = 1.32

        I14 = 1.15e-5
        k14 = 1.14e3
        c14 = 0.05

        I15 = 1.32e-4
        k15 = 3.10e4
        c15 = 1.32

        I16 = 4.27e-6
        k16 = 1.22e4
        c16 = 0.52

        I17 = 2.69e-4
        k17 = 4.43e4
        c17 = 1.88
        d17 = 0.0042

        I18 = 1.80e-4
        k18 = 1.38e5
        c18 = 5.86

        I19 = 2.00e-5
        k19 = 2.00e4
        c19 = 0.85

        I20 = 2.00e-4
        k20 = 1.38e5
        c20 = 5.86

        I21 = 4.27e-6
        k21 = 1.22e4
        c21 = 0.52

        I22 = 4.95e-2
        d22 = 0.24

        shafts = []
        shafts.append(ot.Shaft(0, 1, k=k1, c=c1))
        shafts.append(ot.Shaft(1, 2, k=k2, c=c2))
        shafts.append(ot.Shaft(2, 3, k=k3, c=c3))
        shafts.append(ot.Shaft(3, 4, k=k4, c=c4))
        shafts.append(ot.Shaft(4, 5, k=k5, c=c5))
        shafts.append(ot.Shaft(5, 6, k=k6, c=c6))
        shafts.append(ot.Shaft(6, 7, k=k7, c=c7))
        shafts.append(ot.Shaft(7, 8, k=k8, c=c8))
        shafts.append(ot.Shaft(8, 9, k=k9, c=c9))
        shafts.append(ot.Shaft(9, 10, k=k10, c=c10))
        shafts.append(ot.Shaft(10, 11, k=k11, c=c11))

        shafts.append(ot.Shaft(12, 13, k=k12, c=c12))
        shafts.append(ot.Shaft(13, 14, k=k13, c=c13))
        shafts.append(ot.Shaft(14, 15, k=k14, c=c14))
        shafts.append(ot.Shaft(15, 16, k=k15, c=c15))
        shafts.append(ot.Shaft(16, 17, k=k16, c=c16))

        shafts.append(ot.Shaft(18, 19, k=k17, c=c17))
        shafts.append(ot.Shaft(19, 20, k=k18, c=c18))
        shafts.append(ot.Shaft(20, 21, k=k19, c=c19))
        shafts.append(ot.Shaft(21, 22, k=k20, c=c20))
        shafts.append(ot.Shaft(22, 23, k=k21, c=c21))

        """ Disk elements """
        disks = []
        disks.append(ot.Disk(0, I=I1, c=d1))
        disks.append(ot.Disk(1, I=I2))
        disks.append(ot.Disk(2, I=I3))
        disks.append(ot.Disk(3, I=I4))
        disks.append(ot.Disk(4, I=I5))
        disks.append(ot.Disk(5, I=I6))
        disks.append(ot.Disk(6, I=I7))
        disks.append(ot.Disk(7, I=I8))
        disks.append(ot.Disk(8, I=I9))
        disks.append(ot.Disk(9, I=I10))
        disks.append(ot.Disk(10, I=I11))
        disks.append(ot.Disk(11, I=I12/2, c=d12))
        disks.append(ot.Disk(12, I=I12/2, c=d12))
        disks.append(ot.Disk(13, I=I13))
        disks.append(ot.Disk(14, I=I14))
        disks.append(ot.Disk(15, I=I15))
        disks.append(ot.Disk(16, I=I16))
        disks.append(ot.Disk(17, I=I17/2, c=d17))
        disks.append(ot.Disk(18, I=I17/2, c=d17))
        disks.append(ot.Disk(19, I=I18))
        disks.append(ot.Disk(20, I=I19))
        disks.append(ot.Disk(21, I=I20))
        disks.append(ot.Disk(22, I=I21))
        disks.append(ot.Disk(23, I=I22, c=d22))

        """ Gear elements """
        gear1 = ot.Gear(11, 0, 10)
        gear2 = ot.Gear(12, 0, 30, parent=gear1)

        gear3 = ot.Gear(17, 0, 10)
        gear4 = ot.Gear(18, 0, 40, parent=gear3)

        gears = [gear1, gear2, gear3, gear4]

        """ Assembly """
        assembly = ot.Assembly(shafts, disks, gear_elements=gears)

        return assembly



def load_feather(plot: bool = False, N: int = 1) -> np.ndarray:
    if N < 1:
        raise ValueError("N must be >= 1")

    measurements = pd.read_feather('examples/opentorsion_22_disk/data/testbench_evaluation_dataset.feather')
 
    # Group by sample_id
    sample_groups = measurements.groupby('sample_id')
    # List to store the slices for each sample
    samples_list = []
 
    for sample_id, group in sample_groups:
        print(sample_id)
        # Ensure the group is sorted by time if needed
        group = group.sort_values('time')
 
        # Select rows [1000:6000] for the current sample
        sample_slice = group.iloc[:]
        print(sample_slice.shape)
        # Extract the columns in order: T1, T2, u_m, u_p
        time = sample_slice["time"].to_numpy()

        cutoff = np.argmin(np.abs(time - 6.0))

        # Roll u_m back to align it in time with the other signals
        dt = np.mean(np.diff(time))
        u_m_shift = int(round(0.013 / dt))
        u_m_full = np.roll(sample_slice["u_m"].to_numpy(), -u_m_shift)

        time = time[cutoff:]
        e1 = sample_slice["E1"].to_numpy()[cutoff:]  # input
        e2 = sample_slice["E2"].to_numpy()[cutoff:]  # input
        t1 = sample_slice["T1"].to_numpy()[cutoff:]  # input
        t2 = sample_slice["T2"].to_numpy()[cutoff:]  # evaluation
        u_p = sample_slice["u_p"].to_numpy()[cutoff:]
        u_m = sample_slice["u_m"].to_numpy()[cutoff:]
        #u_m = u_m_full[cutoff:]

        if N != 1:
            time = time[::N]
            e1 = e1[::N]
            e2 = e2[::N]
            t1 = t1[::N]
            t2 = t2[::N]
            u_p = u_p[::N]
            u_m = u_m[::N]

        # print(group["sample_id"].unique)
        # Append as a tuple (or list) into our master list

        if plot:
            plt.figure(figsize=(12, 8))
            plt.plot(time, (e1-np.mean(e1))/np.max(e1-np.mean(e1)), label="e1")
            plt.plot(time, (e2-np.mean(e2))/np.max(e2-np.mean(e2)), label="e2")
            plt.plot(time, t2/np.max(t2), label="t2")
            plt.plot(time, u_m/np.max(u_m), label="u_m")
            plt.plot(time, u_p/np.max(u_p), label="u_p")
            plt.plot(time, t1/np.max(t1), label="t1")
            plt.xlabel("Time (s)")
            plt.ylabel("Torque (Nm)")
            plt.legend()
            plt.show()
 
        samples_list.append((e1, e2, t1, t2, u_m, u_p))
 
    # Now samples_list is a list where each element corresponds to one sample's measurements slice.
    print(f"Generated {len(samples_list)} samples.")

    return time, np.column_stack((e1, e2, t1, t2)), u_m, u_p

if __name__ == "__main__":
    sim_t, measurement_data, motor, propeller = load_feather(plot=False, N=1)
    dt = np.mean(np.diff(sim_t))

    # Rearrange measurement data to be in the same order as the model
    all_measurements = measurement_data[:, [2,3,0,1]]

    # Possible sensors:
    # torque1, velocity1, velocity2

    sensors = ["torque1", "velocity1"]
    use_input = False

    state_index_connection = {
        "torque1": 8,
        "velocity1": 27,
        "velocity2": 28,
    }

    state_sensor_connection = {
        "torque1": 0,
        "velocity1": 2,
        "velocity2": 3,
    }

    measured_states = [state_index_connection[sensor] for sensor in sensors]
    state_sensor_indices = [state_sensor_connection[sensor] for sensor in sensors]

    # openmhe expects (channels, samples); the loaded data is (samples, channels)
    y = all_measurements[:, state_sensor_indices].T

    test_bench = TestBench(measured_states)
    mhe_system = mhe.SystemModel.from_matrices(
        test_bench.A,
        test_bench.B,
        test_bench.C,
        test_bench.D,
        is_discrete=False,
        dt=dt,
    )

    n_window = 35
    ny, nx, nu = mhe_system.ny, mhe_system.nx, mhe_system.nu

    w_cov = np.zeros(nx)
    w_cov[21] = 0.1
    w_cov[32] = 0.1
    w_cov[38] = 0.1
    w_cov[42] = 0.1

    v_cov = 0.5
    load_lambda = 0.1

    # Known motor torque drives input 0; load (input 1) is estimated.
    u = np.zeros((nu, y.shape[1]))
    u[0, :] = motor

    mhe_objective = mhe.ObjectiveBuilder()

    mhe_objective.add(
        mhe.MeasurementTerm(
            penalty=mhe.L2Penalty(),
            weight=mhe.NoiseWeight(dim=ny, cov=v_cov),
        )
    )
    mhe_objective.add(
        mhe.ProcessTerm(
            penalty=mhe.L2Penalty(),
            weight=mhe.NoiseWeight(dim=nx, cov=w_cov),
        )
    )

    mhe_objective.add(
            mhe.InputRandomWalk(target_idx=[1], lambda_u=load_lambda),
        )

    if use_input:
        mhe_objective.add(
            mhe.InputTrackingTerm(target_idx=[0], weight=mhe.NoiseWeight(dim=1, cov=w_cov[0]), reference="measured")
        )
    else:
        mhe_objective.add(
            mhe.InputRandomWalk(target_idx=[0], lambda_u=load_lambda)
        )

    mhe_objective.add(mhe.EKFArrivalCost(mhe_system, builder=mhe_objective))

    print("Compiling ACADOS solver...")
    solver = mhe.build_mhe_solver(
        mhe_system,
        n_window,
        mhe_objective,
        dt=dt,
        already_discrete=True,
        qp_solver="PARTIAL_CONDENSING_HPIPM",
    )

    print("Running MHE (C solver, -O3 -march=native -ffast-math)...")
    u_hat, x_hat = mhe.run_c_solver(solver, y, u)

    # run_solver returns estimates at the last stage of each window (index k-1).
    n_est = x_hat.shape[1]
    sl = slice(n_window - 1, n_window - 1 + n_est)
    t_est = sim_t[sl]
    t2_true = all_measurements[sl, 1]
    t2_est = x_hat[18, :]
    prop_true = propeller[sl]
    prop_est = -1 * u_hat[1, :]

    t2_rmse = np.sqrt(np.nanmean((t2_est - t2_true) ** 2))
    prop_rmse = np.sqrt(np.nanmean((prop_est - prop_true) ** 2))
    print(f"t2 RMSE (MHE): {t2_rmse:.4f}")
    print(f"Propeller u[1] RMSE (MHE): {prop_rmse:.4f}")

    zoom = (8.6, 9.0)
    zoom_mask = (t_est >= zoom[0]) & (t_est <= zoom[1])

    series = [
        ("t2 torque", t2_true, t2_est, (-1, 28)),
        ("Propeller u[1]", prop_true, prop_est, (-5, 22)),
    ]

    n_panels = len(series)
    fig, axes = plt.subplots(
        n_panels,
        2,
        figsize=(14, 2.2 * n_panels),
        sharex="col",
        width_ratios=[1.4, 1],
    )
    if n_panels == 1:
        axes = np.array([axes])

    for row, (ylabel, true, est, ylim) in enumerate(series):
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
            ax.set_ylim(*ylim)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{subtitle}  |  RMSE = {rmse:.4f}", fontsize=9, loc="left")
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(
        f"MHE estimates ({', '.join(sensors)})",
        y=1.002,
    )
    fig.tight_layout()
    plt.show()

    