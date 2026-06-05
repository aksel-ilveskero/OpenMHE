"""Tests for ObjectiveBuilder.to_latex / objective_to_latex."""

import numpy as np
import pytest
import openmhe as mhe
from openmhe.export.latex import LatexSymbols


def _demo_objective():
    """Builder with measurement, process, RW load, and tracked motor."""
    obj = mhe.ObjectiveBuilder()
    obj.add(mhe.MeasurementTerm(mhe.L2Penalty(), mhe.NoiseWeight(2, cov=0.5)))
    obj.add(mhe.ProcessTerm(mhe.L2Penalty(), mhe.NoiseWeight(7, cov=0.001)))
    obj.add(mhe.InputRandomWalk([1], lambda_u=1.0))
    obj.add(mhe.InputTrackingTerm([0], mhe.NoiseWeight(1, cov=1e-6), reference="measured"))
    return obj


def test_contains_min_and_weight_symbols():
    """Substituted form includes min operator and inverse-covariance labels."""
    tex = _demo_objective().to_latex()
    assert r"\min" in tex
    assert "R^{-1}" in tex
    assert "Q^{-1}" in tex
    assert "U^{-1}" in tex
    assert "P^{-1}" not in tex
    assert r"\lambda" in tex


def test_steady_state_arrival_latex():
    """Steady-state arrival cost renders when added to the objective."""
    obj = _demo_objective()
    obj.add(mhe.SteadyStateArrivalCost())
    tex = obj.to_latex(underbrace=True)
    assert r"x_{t-N}" in tex
    assert "P^{-1}" in tex
    assert r"\text{steady-state arrival cost}" in tex
    assert r"\hat{x}" not in tex


def test_ekf_arrival_latex():
    """EKF arrival cost uses filter prior notation."""
    obj = _demo_objective()
    system = mhe.SystemModel.from_matrices(
        np.array([[0.9]]),
        np.array([[0.1, 0.05]]),
        np.array([[1.0, 0.0]]),
        None,
        is_discrete=True,
        dt=0.01,
    )
    arrival = mhe.EKFArrivalCost(system, Q=np.array([[0.001]]), R=np.array([[0.5]]))
    obj.add(arrival)
    tex = obj.to_latex(underbrace=True)
    assert r"x_{t-N} - \hat{x}_{t-N|t-N-1}" in tex
    assert r"P_{t-N|t-N-1}^{-1}" in tex
    assert r"\text{EKF arrival cost}" in tex


def test_ukf_arrival_latex():
    """UKF arrival cost is labeled distinctly from EKF."""
    obj = _demo_objective()
    system = mhe.SystemModel.from_matrices(
        np.array([[0.9]]),
        np.array([[0.1, 0.05]]),
        np.array([[1.0, 0.0]]),
        None,
        is_discrete=True,
        dt=0.01,
    )
    arrival = mhe.UKFArrivalCost(system, Q=np.array([[0.001]]), R=np.array([[0.5]]))
    obj.add(arrival)
    tex = obj.to_latex(underbrace=True)
    assert r"\text{UKF arrival cost}" in tex
    assert r"P_{t-N|t-N-1}^{-1}" in tex


def test_measurement_residual_rendered():
    """Measurement residual is inlined as ``y - (C x + D u)``."""
    tex = _demo_objective().to_latex()
    assert "y_{k} - (C x_{k} + D u_{k})" in tex


def test_underbrace_toggle():
    """``underbrace=True`` adds labeled term descriptions."""
    obj = _demo_objective()
    obj.add(mhe.SteadyStateArrivalCost())
    with_ub = obj.to_latex(underbrace=True)
    without_ub = obj.to_latex(underbrace=False)
    assert r"\underbrace" in with_ub
    assert r"\text{measurement error}" in with_ub
    assert r"\text{process noise}" in with_ub
    assert r"\text{steady-state arrival cost}" in with_ub
    assert r"\underbrace" not in without_ub


def test_multiline_defaults_to_underbrace():
    """Multiline ``aligned`` layout follows ``underbrace``."""
    obj = _demo_objective()
    obj.add(mhe.SteadyStateArrivalCost())
    assert r"\begin{aligned}" in obj.to_latex(underbrace=True)
    assert r"\begin{aligned}" not in obj.to_latex(underbrace=False)


def test_no_arrival_without_cost():
    """Arrival term is omitted when no arrival cost was added."""
    obj = _demo_objective()
    assert "P^{-1}" not in obj.to_latex()
    assert "P^{-1}" not in obj.to_latex(form="constrained")


def test_input_tracking_reference_variants():
    """``reference='zero'`` renders as subtraction from zero."""
    obj = mhe.ObjectiveBuilder()
    obj.add(mhe.MeasurementTerm(mhe.L2Penalty(), mhe.NoiseWeight(1, cov=0.5)))
    obj.add(mhe.ProcessTerm(mhe.L2Penalty(), mhe.NoiseWeight(2, cov=0.001)))
    obj.add(mhe.InputTrackingTerm([0], mhe.NoiseWeight(1, cov=1e-6), reference="zero"))
    tex = obj.to_latex()
    assert "u^{(0)}_{k} - 0" in tex


def test_second_diff_residual():
    """Second-difference regulator residual appears in substituted form."""
    obj = mhe.ObjectiveBuilder()
    obj.add(mhe.MeasurementTerm(mhe.L2Penalty(), mhe.NoiseWeight(1, cov=0.5)))
    obj.add(mhe.ProcessTerm(mhe.L2Penalty(), mhe.NoiseWeight(2, cov=0.001)))
    obj.add(mhe.InputSecondDiffReg([0], lambda_u=10.0))
    tex = obj.to_latex()
    assert "u^{(0)}_{k} - 2 u^{(0)}_{k-1} + u^{(0)}_{k-2}" in tex


def test_non_l2_penalty_rendering_and_definitions():
    """Huber/L1 terms and optional definition preamble render correctly."""
    obj = mhe.ObjectiveBuilder()
    obj.add(mhe.MeasurementTerm(mhe.HuberPenalty(delta=1.5), mhe.NoiseWeight(1, cov=0.5)))
    obj.add(mhe.ProcessTerm(mhe.L1Penalty(), mhe.NoiseWeight(2, cov=0.001)))
    tex = obj.to_latex(define_penalties=True)
    assert r"\mathcal{H}_{1.5}" in tex
    assert "Q^{-1}, 1" in tex
    assert r"\mathcal{H}_{\delta}" in tex  # definition preamble


def test_standalone_document():
    """``standalone=True`` wraps output in a full LaTeX document."""
    tex = _demo_objective().to_latex(standalone=True)
    assert tex.startswith(r"\documentclass{article}")
    assert r"\begin{document}" in tex
    assert r"\end{document}" in tex


def test_symbol_override():
    """Custom ``LatexSymbols`` replace default measurement symbols."""
    syms = LatexSymbols(output="z", cov_meas="\\Sigma")
    obj = _demo_objective()
    tex = obj.to_latex(symbols=syms)
    assert "z_{k}" in tex
    assert "\\Sigma^{-1}" in tex


def test_module_function_matches_method():
    """``objective_to_latex`` matches ``ObjectiveBuilder.to_latex``."""
    obj = _demo_objective()
    assert mhe.objective_to_latex(obj) == obj.to_latex()


def _constrained_objective():
    """Builder for constrained-form LaTeX tests."""
    obj = mhe.ObjectiveBuilder()
    obj.add(mhe.MeasurementTerm(mhe.L2Penalty(), mhe.NoiseWeight(1, cov=0.5)))
    obj.add(mhe.ProcessTerm(mhe.L2Penalty(), mhe.NoiseWeight(7, cov=0.001)))
    obj.add(mhe.InputSecondDiffReg([1], lambda_u=1.0))
    obj.add(mhe.InputTrackingTerm([0], mhe.NoiseWeight(1, cov=1e-6), reference="measured"))
    return obj


def test_constrained_has_minimize_and_subject_to():
    """Constrained form uses minimize / subject to headers."""
    tex = _constrained_objective().to_latex(form="constrained")
    assert r"\text{minimize}" in tex
    assert r"\text{subject to}" in tex
    assert r"\underset{x,\, u,\, \delta}" in tex


def test_constrained_dynamics_and_measurement_constraints():
    """Constrained form lists state and measurement equalities."""
    tex = _constrained_objective().to_latex(form="constrained")
    assert "x_{k} = A x_{k-1} + B u_{k-1} + w_{k}" in tex
    assert "y_{k} = C x_{k} + D u_{k} + v_{k}" in tex


def test_constrained_defect_constraint():
    """Second-difference defect appears under subject to."""
    tex = _constrained_objective().to_latex(form="constrained")
    assert r"\delta_{k} = u^{(1)}_{k} - 2 u^{(1)}_{k-1} + u^{(1)}_{k-2}" in tex


def test_constrained_objective_uses_noise_variables():
    """Constrained cost is written in v, w, and delta variables."""
    tex = _constrained_objective().to_latex(form="constrained")
    assert r"\left\lVert v_{k} \right\rVert^2_{R^{-1}}" in tex
    assert r"\left\lVert w_{k} \right\rVert^2_{Q^{-1}}" in tex
    assert r"\lambda\, \left\lVert \delta_{k} \right\rVert_{2}^{2}" in tex


def test_constrained_input_error_no_double_subscript():
    """Input error uses q^{(i)}_{u,k} notation (no double subscript)."""
    tex = _constrained_objective().to_latex(form="constrained")
    assert "q^{(0)}_{u,k}" in tex
    assert "q_u^{(0)}_{k}" not in tex


def test_constrained_alias_and_arrival():
    """``form='subject_to'`` aliases constrained; arrival follows ``obj.add(...)``."""
    obj = _constrained_objective()
    obj.add(mhe.SteadyStateArrivalCost())
    assert obj.to_latex(form="subject_to") == obj.to_latex(form="constrained")
    assert "P^{-1}" in obj.to_latex(form="constrained")
    obj.arrival_cost = None
    assert "P^{-1}" not in obj.to_latex(form="constrained")


def test_duplicate_arrival_cost_raises():
    """Only one arrival cost may be added."""
    obj = _demo_objective()
    obj.add(mhe.SteadyStateArrivalCost())
    with pytest.raises(ValueError, match="at most one arrival cost"):
        obj.add(mhe.SteadyStateArrivalCost())


def test_constrained_random_walk_defect():
    """Random-walk defect is first difference in constrained form."""
    obj = mhe.ObjectiveBuilder()
    obj.add(mhe.MeasurementTerm(mhe.L2Penalty(), mhe.NoiseWeight(1, cov=0.5)))
    obj.add(mhe.ProcessTerm(mhe.L2Penalty(), mhe.NoiseWeight(2, cov=0.001)))
    obj.add(mhe.InputRandomWalk([1], lambda_u=1.0))
    tex = obj.to_latex(form="constrained")
    assert r"\delta_{k} = u^{(1)}_{k} - u^{(1)}_{k-1}" in tex
