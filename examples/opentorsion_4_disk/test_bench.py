"""OpenTorsion plant model for the OpenMHE shaft-line demo."""

import opentorsion as ot
import scipy.signal as signal
import numpy as np


class TestBench:
    """OpenTorsion four-disk shaft line used by the MHE demo."""

    def __init__(self):
        """Assemble a four-disk shaft line with motor and load inputs."""
        self.inertias = [0.1, 0.3, 0.5, 0.1]
        self.stiffnesses = [1e4, 2e4, 3e4]
        self.damping = [0.1, 0.0, 0.2]

        disks = []
        shafts = []

        for i in range(len(self.inertias)):
            disks.append(ot.Disk(i, self.inertias[i]))

        for i in range(len(self.stiffnesses)):
            shafts.append(ot.Shaft(i, i + 1, k=self.stiffnesses[i], c=self.damping[i]))

        self.assembly = ot.Assembly(disk_elements=disks, shaft_elements=shafts)

        self.A, self.B, self.C, self.D = self.assembly.state_space(
            minimal_representation=True
        )

        self.B = self.B[:, [0, -1]]

        self.measured_states = [1, 5]
        self.C = self.C[self.measured_states, :]
        self.D = self.D[self.measured_states, :][:, [0, -1]]

    def simulate(self, dt, u, w, v):
        """Discrete-time simulation with process noise ``w`` and measurement noise ``v``."""
        (A_d, B_d, C_d, D_d, dt) = signal.cont2discrete(
            (self.A, self.B, self.C, self.D), dt=dt
        )

        x = np.zeros((A_d.shape[0], u.shape[1]))
        y = np.zeros((C_d.shape[0], u.shape[1]))

        for i in range(1, u.shape[1]):
            x[:, i] = (
                A_d @ x[:, i - 1]
                + B_d @ u[:, i]
                + np.random.default_rng().multivariate_normal(np.zeros(x.shape[0]), w)
            )
            y[:, i] = (
                C_d @ x[:, i]
                + D_d @ u[:, i]
                + np.random.default_rng().multivariate_normal(np.zeros(y.shape[0]), v)
            )

        return x, y


if __name__ == "__main__":
    TestBench()
