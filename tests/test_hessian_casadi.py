"""Tests for CasADi LTI decision Hessian extraction."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

import openmhe as mhe


def _tiny_lti_solver(N: int = 5):
    A = np.array([[0.9]])
    B = np.array([[0.1, 0.05]])
    C = np.array([[1.0]])
    D = np.zeros((1, 2))
    dt = 0.1
    system = mhe.SystemModel.from_matrices(A, B, C, D, is_discrete=False, dt=dt)

    builder = mhe.ObjectiveBuilder()
    builder.add(
        mhe.MeasurementTerm(
            penalty=mhe.L2Penalty(),
            weight=mhe.NoiseWeight(dim=1, cov=0.5),
        )
    )
    builder.add(
        mhe.ProcessTerm(
            penalty=mhe.L2Penalty(),
            weight=mhe.NoiseWeight(dim=1, cov=0.01),
        )
    )
    builder.add(mhe.InputRandomWalk(target_idx=[1], lambda_u=1.0))
    builder.add(mhe.InputRandomWalk(target_idx=[0], lambda_u=1.0))
    builder.add(mhe.SteadyStateArrivalCost())

    solver = mhe.build_mhe_solver(system, N, builder, dt=dt, lti_linear_ls_fast=False)
    return solver


def test_lti_ls_decision_hessian_shape_and_spd():
    pytest.importorskip("acados_template")
    solver = _tiny_lti_solver(N=4)
    out = mhe.lti_ls_decision_hessian(solver)

    N = solver._N
    nu_err = out["nu_err"]
    nx = out["nx"]
    assert out["H"].shape == (N * nu_err + nx, N * nu_err + nx)
    eig = np.linalg.eigvalsh(out["H"])
    assert np.all(eig > -1e-10), f"expected PSD Hessian, min eig {eig.min()}"


def test_rts_mid_window_tighter_than_steady_state_kf():
    A = np.array([[0.9, 0.05], [0.0, 0.95]])
    C = np.array([[1.0, 0.0]])
    Q = np.diag([0.01, 0.01])
    R = np.diag([0.1])
    P0 = np.eye(2) * 0.5
    rts = mhe.fixed_interval_smoother_covariances(A, C, Q, R, P0, N=10)
    P_ss, _ = mhe.kalman_present_covariance(A, C, Q, R)
    assert rts["P_smooth"][5][0, 0] < P_ss[0, 0]


def test_rts_singular_predicted_covariance():
    """Backward pass must tolerate null directions in P_{k+1|k}."""
    import warnings

    A = np.diag([0.95, 1.0, 0.9])
    C = np.array([[1.0, 0.0, 0.0]])
    Q = np.diag([0.01, 0.0, 0.01])
    R = np.diag([0.1])
    P0 = np.diag([0.5, 1e6, 0.5])
    with warnings.catch_warnings():
        warnings.simplefilter("error", sla.LinAlgWarning)
        rts = mhe.fixed_interval_smoother_covariances(A, C, Q, R, P0, N=5)
    assert rts["P_smooth"][2][0, 0] > 0.0
    assert rts["P_smooth"][2][2, 2] > 0.0
    assert rts["P_smooth"][2][0, 0] <= rts["P_filt"][2][0, 0]


def test_decision_variable_labels_match_hessian_size():
    pytest.importorskip("acados_template")
    solver = _tiny_lti_solver(N=4)
    labels = mhe.decision_variable_labels(solver)
    hess = mhe.lti_ls_decision_hessian(solver)
    assert len(labels) == hess["H"].shape[0]
    assert labels[0]["group"] == "process_noise"
    assert labels[-1]["group"] == "initial_state"
