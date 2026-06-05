"""Tests for arrival-cost strategies."""

import numpy as np
import pytest
import scipy.linalg as sla

import openmhe as mhe


def _tiny_system():
    A = np.array([[0.9, 0.0], [0.0, 0.8]])
    B = np.array([[0.1, 0.0], [0.0, 0.05]])
    C = np.array([[1.0, 0.0]])
    D = np.zeros((1, 2))
    return mhe.SystemModel.from_matrices(A, B, C, D, is_discrete=True, dt=0.01)


def _base_objective(ny=1, nx=2):
    obj = mhe.ObjectiveBuilder()
    obj.add(mhe.MeasurementTerm(mhe.L2Penalty(), mhe.NoiseWeight(ny, cov=0.1)))
    obj.add(mhe.ProcessTerm(mhe.L2Penalty(), mhe.NoiseWeight(nx, cov=0.01)))
    return obj


def test_noise_covs_from_builder_round_trip():
    """Diagonal weights invert to the supplied covariances."""
    system = _tiny_system()
    obj = _base_objective()
    Q, R = mhe.noise_covs_from_builder(obj, system.nx, system.ny)
    np.testing.assert_allclose(Q, np.diag([0.01, 0.01]))
    np.testing.assert_allclose(R, np.diag([0.1]))


def test_steady_state_matches_dare():
    """SteadyStateArrivalCost matches scipy DARE on a tiny LTI model."""
    system = _tiny_system()
    obj = _base_objective()
    Q, R = mhe.noise_covs_from_builder(obj, system.nx, system.ny)
    P_ref = sla.solve_discrete_are(system.A.T, system.C.T, Q, R)
    arrival = mhe.SteadyStateArrivalCost()
    P = arrival.initial_covariance(system, obj)
    np.testing.assert_allclose(P, P_ref, rtol=1e-10)


def test_ekf_window_prior_hand_steps():
    """EKF window_prior matches explicit predict/update steps."""
    system = _tiny_system()
    obj = _base_objective()
    Q, R = mhe.noise_covs_from_builder(obj, system.nx, system.ny)
    ekf = mhe.EKFArrivalCost(system, builder=obj)

    y = np.array([[0.1, 0.2, 0.15, 0.05]])
    u = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])

    x, P = ekf.window_prior(0, y, u)
    x_hand = system.A @ np.zeros(2) + system.B @ u[:, 0]
    P0 = np.diag(np.diag(Q))
    P_hand = system.A @ P0 @ system.A.T + Q
    np.testing.assert_allclose(x, x_hand)
    np.testing.assert_allclose(P, P_hand)

    ekf.reset()
    x1, P1 = ekf.window_prior(1, y, u)
    x_pred = system.A @ x_hand + system.B @ u[:, 1]
    innov = y[:, 0] - (system.C @ x_hand + system.D @ u[:, 0])
    S = system.C @ P_hand @ system.C.T + R
    K = P_hand @ system.C.T @ np.linalg.inv(S)
    x_post = x_hand + K @ innov
    P_post = (np.eye(2) - K @ system.C) @ P_hand @ (np.eye(2) - K @ system.C).T + K @ R @ K.T
    x_hand1 = system.A @ x_post + system.B @ u[:, 1]
    P_hand1 = system.A @ P_post @ system.A.T + Q
    np.testing.assert_allclose(x1, x_hand1, rtol=1e-10)
    np.testing.assert_allclose(P1, P_hand1, rtol=1e-10)


def test_ukf_matches_ekf_on_lti():
    """UKF and EKF agree on the same LTI data sequence."""
    system = _tiny_system()
    obj = _base_objective()
    ekf = mhe.EKFArrivalCost(system, builder=obj)
    ukf = mhe.UKFArrivalCost(system, builder=obj)

    rng = np.random.default_rng(0)
    y = rng.normal(size=(1, 8))
    u = rng.normal(size=(2, 8))

    for t_start in range(4):
        ekf.reset()
        ukf.reset()
        x_e, P_e = ekf.window_prior(t_start, y, u)
        x_u, P_u = ukf.window_prior(t_start, y, u)
        np.testing.assert_allclose(x_e, x_u, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(P_e, P_u, rtol=1e-5, atol=1e-7)


def test_build_solver_with_steady_state_arrival():
    """Smoke test: solver builds with SteadyStateArrivalCost."""
    pytest.importorskip("acados_template")
    system = _tiny_system()
    obj = _base_objective()
    obj.add(mhe.KnownInput([0, 1]))
    arrival = mhe.SteadyStateArrivalCost()
    obj.add(arrival)
    solver = mhe.build_mhe_solver(
        system,
        5,
        obj,
        dt=0.01,
        already_discrete=True,
    )
    assert solver._arrival_cost is obj.arrival_cost
    assert solver._W0_template is not None
