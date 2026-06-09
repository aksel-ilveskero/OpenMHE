"""Tests for strict-kinematics (sparse) process noise."""

import numpy as np
import pytest

import openmhe as mhe
from openmhe.builder.input_regs import plant_process_cov, sparse_process_noise_config
from openmhe.mhe_strategies.arrival_cost import invert_arrival_covariance


def test_sparse_process_noise_config_shrinks_active_channels():
    nx_base = 45
    n_rw = 0
    W_full = np.zeros((nx_base, nx_base))
    W_full[0, 0] = 100.0
    W_full[43, 43] = 100.0

    nw, W_sparse, G = sparse_process_noise_config(W_full, nx_base, n_rw)

    assert nw == 2
    assert W_sparse.shape == (2, 2)
    assert G.shape == (45, 2)
    assert G[0, 0] == 1.0
    assert G[43, 1] == 1.0
    np.testing.assert_allclose(W_sparse, np.diag([100.0, 100.0]))


def test_plant_process_cov_from_sparse_g():
    nx_base = 45
    W_full = np.zeros((nx_base, nx_base))
    W_full[0, 0] = 100.0
    W_full[43, 43] = 50.0
    _, W_sparse, G = sparse_process_noise_config(W_full, nx_base, 0)

    Q = plant_process_cov(G, W_sparse, nx_base)

    assert Q.shape == (45, 45)
    np.testing.assert_allclose(Q[0, 0], 0.01)
    np.testing.assert_allclose(Q[43, 43], 0.02)
    assert np.count_nonzero(Q) == 2


def test_noise_weight_preserves_zero_cov_entries():
    w = mhe.NoiseWeight(dim=4, cov=[0.01, 0.0, 0.0, 0.02])
    np.testing.assert_allclose(np.diag(w.W), [100.0, 0.0, 0.0, 50.0])


def test_invert_arrival_covariance_sparse_p():
    P = np.diag([0.01, 0.0, 0.02, 0.0])
    W = invert_arrival_covariance(P)
    np.testing.assert_allclose(W, np.diag([100.0, 0.0, 50.0, 0.0]))
    assert np.max(W) <= 1e6


def test_plant_arrival_state_indices_sparse():
    from openmhe.builder.input_regs import plant_arrival_state_indices

    G = np.zeros((45, 2))
    G[21, 0] = 1.0
    G[42, 1] = 1.0
    idx = plant_arrival_state_indices(G, 45)
    np.testing.assert_array_equal(idx, [21, 42])


def test_build_solver_ekf_with_sparse_process():
    pytest.importorskip("acados_template")
    system = mhe.SystemModel.from_matrices(
        np.diag([0.9, 0.8, 0.7]),
        np.ones((3, 1)),
        np.array([[1.0, 0.0, 0.0]]),
        np.zeros((1, 1)),
        is_discrete=True,
        dt=0.01,
    )
    obj = mhe.ObjectiveBuilder()
    obj.add(mhe.MeasurementTerm(mhe.L2Penalty(), mhe.NoiseWeight(1, cov=0.1)))
    obj.add(
        mhe.ProcessTerm(
            mhe.L2Penalty(),
            weight=mhe.NoiseWeight(3, cov=[0.01, 0.0, 0.02]),
        )
    )
    obj.add(mhe.InputRandomWalk(target_idx=[0], lambda_u=0.5))
    obj.add(mhe.EKFArrivalCost(system, builder=obj))
    solver = mhe.build_mhe_solver(system, 3, obj, dt=0.01, already_discrete=True)
    assert solver._W0_template is not None
    assert solver._nw == 3
    assert solver._n_arrival == 2
    assert solver._arrival_state_idx is not None


def test_ekf_arrival_cost_with_sparse_process_before_solver_build():
    system = mhe.SystemModel.from_matrices(
        np.diag([0.9, 0.8, 0.7]),
        np.ones((3, 1)),
        np.array([[1.0, 0.0, 0.0]]),
        np.zeros((1, 1)),
        is_discrete=True,
        dt=0.01,
    )
    obj = mhe.ObjectiveBuilder()
    obj.add(mhe.MeasurementTerm(mhe.L2Penalty(), mhe.NoiseWeight(1, cov=0.1)))
    obj.add(
        mhe.ProcessTerm(
            mhe.L2Penalty(),
            weight=mhe.NoiseWeight(3, cov=[0.01, 0.0, 0.02]),
        )
    )
    mhe.EKFArrivalCost(system, builder=obj)


def test_noise_covs_from_builder_sparse_process():
    system = mhe.SystemModel.from_matrices(
        np.diag([0.9, 0.8, 0.7]),
        np.ones((3, 1)),
        np.array([[1.0, 0.0, 0.0]]),
        np.zeros((1, 1)),
        is_discrete=True,
        dt=0.01,
    )
    obj = mhe.ObjectiveBuilder()
    obj.add(mhe.MeasurementTerm(mhe.L2Penalty(), mhe.NoiseWeight(1, cov=0.1)))
    obj.add(
        mhe.ProcessTerm(
            mhe.L2Penalty(),
            weight=mhe.NoiseWeight(3, cov=[0.01, 0.0, 0.02]),
        )
    )
    obj.add(mhe.InputRandomWalk(target_idx=[0], lambda_u=0.5))

    solver = mhe.build_mhe_solver(system, 3, obj, dt=0.01, already_discrete=True)
    assert solver._nw == 3
    assert solver._nw_full == 4
    assert solver._G_proc.shape == (4, 3)

    Q, R = mhe.noise_covs_from_builder(obj, system.nx, system.ny)
    np.testing.assert_allclose(Q, np.diag([0.01, 0.0, 0.02]))
    np.testing.assert_allclose(R, np.diag([0.1]))


@pytest.mark.slow
def test_sparse_process_noise_solver_smoke():
    pytest.importorskip("acados_template")
    A = np.diag([0.9, 0.8, 0.7])
    B = np.array([[0.1], [0.05], [0.02]])
    C = np.array([[1.0, 0.0, 0.0]])
    D = np.zeros((1, 1))
    system = mhe.SystemModel.from_matrices(A, B, C, D, is_discrete=True, dt=0.1)

    obj = mhe.ObjectiveBuilder()
    obj.add(mhe.MeasurementTerm(mhe.L2Penalty(), mhe.NoiseWeight(1, cov=0.5)))
    obj.add(
        mhe.ProcessTerm(
            mhe.L2Penalty(),
            weight=mhe.NoiseWeight(3, cov=[0.01, 0.0, 0.02]),
        )
    )
    obj.add(mhe.KnownInput([0]))

    solver = mhe.build_mhe_solver(system, 5, obj, dt=0.1, already_discrete=True)
    assert solver._nw == 2
    assert solver._nu_ctrl + solver._nw < 1 + system.nx

    rng = np.random.default_rng(0)
    n_steps = 25
    y = rng.normal(size=(1, n_steps))
    u = rng.normal(size=(1, n_steps))

    u_hat, x_hat = mhe.run_solver(solver, y, u)
    assert np.any(np.isfinite(u_hat))
    assert np.any(np.isfinite(x_hat))
