"""Dataclass placeholder for future high-level Acados MHE configuration."""

from dataclasses import dataclass
from typing import Optional

# These would be imported from your 'blocks/' folder
# from blocks.arrival_cost import BaseArrivalCost
# from blocks.penalties import BasePenalty


@dataclass
class AcadosConfig:
    """Horizon, timing, and solver options for future codegen helpers."""
    # 1. Horizon & Timing
    N: int                     # Number of steps in the sliding window
    dt: float                  # Time step (seconds)
    
    # 2. Solver Options (Defaults provided)
    solver_type: str = "SQP"   # 'SQP' for testing, 'SQP_RTI' for real-time C deployment
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    max_iter: int = 50
    tol_stat: float = 1e-4     # Optimization tolerance
    
    # 3. The "Building Blocks" (Attached by the user)
    arrival_cost: Optional[any] = None  # e.g., EKFArrivalCost()
    cost_penalty: Optional[any] = None  # e.g., HuberPenalty()
    
    # 4. Hardware Deployment Flags
    target_hardware: str = "SIL"        # 'SIL' (PC), 'DSPACE', 'GENERIC_MICRO'
    
    def validate(self) -> None:
        """Raise if horizon, timing, or attached blocks are invalid for codegen."""
        if self.N <= 1:
            raise ValueError(f"Horizon length N must be > 1. Received: {self.N}")
        
        if self.dt <= 0.0:
            raise ValueError(f"Time step dt must be strictly positive. Received: {self.dt}")
            
        if self.solver_type not in ["SQP", "SQP_RTI"]:
            raise ValueError(f"Unsupported solver_type: {self.solver_type}")
            
        # Ensure a penalty block was actually attached
        if self.cost_penalty is None:
            raise ValueError("You must attach a cost_penalty block (e.g., L2Penalty).")