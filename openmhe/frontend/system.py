"""Discrete-time plant models for MHE."""

import casadi as ca
import opentorsion as ot
from scipy.signal import StateSpace, cont2discrete


class SystemModel:
    """Discrete-time LTI model for Acados MHE."""

    def __init__(self, Ad, Bd, Cd, Dd, dt):
        """
        Parameters
        ----------
        Ad, Bd, Cd, Dd : ndarray
            Discrete-time state-space matrices.
        dt : float
            Sample period in seconds.
        """
        self.dt = dt
        self.nx = Ad.shape[0]
        self.nu = Bd.shape[1]
        self.ny = Cd.shape[0]

        self.A = Ad
        self.B = Bd
        self.C = Cd
        self.D = Dd

        self.x = ca.SX.sym("x", self.nx)
        self.u = ca.SX.sym("u", self.nu)
        self.disc_dyn_expr = self.A @ self.x + self.B @ self.u

    @classmethod
    def from_scipy(cls, sys: StateSpace, dt=None):
        """Build from a :class:`scipy.signal.StateSpace` object.

        Continuous models are discretized with zero-order hold when ``sys.dt``
        is None; ``dt`` must be supplied in that case.
        """
        if sys.dt is None:
            if dt is None:
                raise ValueError(
                    "Continuous SciPy systems require a 'dt' for discretization."
                )
            sys_d = sys.to_discrete(dt, method="zoh")
            return cls(sys_d.A, sys_d.B, sys_d.C, sys_d.D, dt)
        return cls(sys.A, sys.B, sys.C, sys.D, sys.dt)

    @classmethod
    def from_matrices(cls, A, B, C, D, is_discrete=True, dt=None):
        """Build from numpy state-space matrices.

        Parameters
        ----------
        is_discrete : bool
            If False, continuous matrices are discretized with ZOH at ``dt``.
        dt : float
            Required when ``is_discrete`` is False (or for discrete labeling).
        """
        if not is_discrete:
            if dt is None:
                raise ValueError(
                    "Continuous matrices require a 'dt' for discretization."
                )
            Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), dt, method="zoh")
            return cls(Ad, Bd, Cd, Dd, dt)
        if dt is None:
            raise ValueError(
                "Discrete systems still need a known 'dt' for the solver window."
            )
        return cls(A, B, C, D, dt)

    @classmethod
    def from_opentorsion(cls, assembly: ot.Assembly, dt: float):
        """Build from an OpenTorsion :class:`~opentorsion.Assembly` (requires ``opentorsion``)."""
        A, B, C, D = assembly.state_space(minimal_representation=True)
        Ad, Bd, Cd, Dd, dt = cont2discrete((A, B, C, D), dt, method="zoh")
        return cls(Ad, Bd, Cd, Dd, dt)
