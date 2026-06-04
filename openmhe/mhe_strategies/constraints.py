"""Optional hard bound helpers for Acados OCP configuration."""

import numpy as np


class HardBounds:
    """Hard box constraints on selected Acados inputs (``lbu`` / ``ubu``)."""

    def __init__(self):
        """Create empty bound dictionaries for states and inputs."""
        self.x_min = {}
        self.x_max = {}
        self.u_min = {}
        self.u_max = {}
        
    def set_u_bounds(self, min_val, max_val):
        """Set uniform lower and upper bounds on all control channels."""
        self.u_min = min_val
        self.u_max = max_val

    def apply_to_ocp(self, ocp):
        """Write ``lbu`` / ``ubu`` / ``idxbu`` on an :class:`~acados_template.AcadosOcp`."""
        # ACADOS uses lbu (lower bound u) and ubu (upper bound u)
        if self.u_min and self.u_max:
            ocp.constraints.lbu = np.array(self.u_min)
            ocp.constraints.ubu = np.array(self.u_max)
            # Tell ACADOS which indices these apply to
            ocp.constraints.idxbu = np.arange(len(self.u_min))