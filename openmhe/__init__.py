"""OpenMHE: moving horizon estimation with Acados.

Build a sliding-window MHE from LTI dynamics and composable cost terms
(measurement, process noise, known inputs, input regularization). Unknown
inputs can be modeled with :class:`~openmhe.InputRandomWalk` or input regulators
(:class:`~openmhe.InputFirstDiffReg`, :class:`~openmhe.InputSecondDiffReg`).

Example
-------
>>> import openmhe as mhe
>>> model = mhe.SystemModel.from_matrices(A, B, C, D, is_discrete=False, dt=0.001)
>>> obj = mhe.ObjectiveBuilder()
>>> obj.add(mhe.MeasurementTerm(mhe.L2Penalty(), mhe.NoiseWeight(dim=ny, std=0.05)))
>>> solver = mhe.build_mhe_solver(model.A, model.B, model.C, model.D, 50, obj, dt=0.001)
>>> u_hat, x_hat = mhe.run_solver(solver, y, u)
"""

__version__ = "0.1.0"

from .frontend import AcadosConfig, SystemModel
from .mhe_strategies import (
    BaseArrivalCost,
    BasePenalty,
    DeadzonePenalty,
    EKFArrivalCost,
    HardBounds,
    HuberPenalty,
    InputFirstDiffReg,
    InputRandomWalk,
    InputRegTerm,
    InputSecondDiffReg,
    InputTrackingTerm,
    KnownInput,
    SteadyStateArrivalCost,
    UKFArrivalCost,
    noise_covs_from_builder,
    weight_from_lambda_u,
    L1Penalty,
    L2Penalty,
    MeasurementTerm,
    NoiseWeight,
    ObjectiveBuilder,
    ProcessTerm,
    steady_state_cov,
)
from .builder import WindowStep, build_mhe_solver, run_c_solver, run_solver
from .export import LatexSymbols, objective_to_latex
from .paths import get_codegen_dir, get_data_root, get_mhe_json_dir, mhe_json_path
from .frontend.acados_runtime import ensure_acados_environment

knownInput = KnownInput

__all__ = [
    "__version__",
    "SystemModel",
    "AcadosConfig",
    "BaseArrivalCost",
    "SteadyStateArrivalCost",
    "EKFArrivalCost",
    "UKFArrivalCost",
    "noise_covs_from_builder",
    "steady_state_cov",
    "BasePenalty",
    "WindowStep",
    "build_mhe_solver",
    "run_solver",
    "run_c_solver",
    "get_data_root",
    "get_mhe_json_dir",
    "get_codegen_dir",
    "mhe_json_path",
    "ensure_acados_environment",
    "HardBounds",
    "L1Penalty",
    "L2Penalty",
    "HuberPenalty",
    "DeadzonePenalty",
    "NoiseWeight",
    "ObjectiveBuilder",
    "MeasurementTerm",
    "ProcessTerm",
    "InputFirstDiffReg",
    "InputSecondDiffReg",
    "InputRandomWalk",
    "InputRegTerm",
    "InputTrackingTerm",
    "KnownInput",
    "knownInput",
    "weight_from_lambda_u",
    "objective_to_latex",
    "LatexSymbols",
]
