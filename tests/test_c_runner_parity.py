"""Python vs C sliding-window MHE driver parity."""

from __future__ import annotations

import numpy as np
import pytest

import openmhe as mhe


def _tiny_system():
    A = np.array([[0.9, 0.0], [0.0, 0.8]])
    B = np.array([[0.1, 0.0], [0.0, 0.05]])
    C = np.array([[1.0, 0.0]])
    D = np.zeros((1, 2))
    return mhe.SystemModel.from_matrices(A, B, C, D, is_discrete=True, dt=0.01)


def _tiny_objective(system, arrival_factory):
    obj = mhe.ObjectiveBuilder()
    obj.add(mhe.MeasurementTerm(mhe.L2Penalty(), mhe.NoiseWeight(1, cov=0.1)))
    obj.add(mhe.ProcessTerm(mhe.L2Penalty(), mhe.NoiseWeight(2, cov=0.01)))
    obj.add(mhe.KnownInput([0, 1]))
    obj.add(arrival_factory(system, obj))
    return obj


@pytest.mark.parametrize(
    "arrival_factory",
    [
        lambda s, o: mhe.SteadyStateArrivalCost(),
        lambda s, o: mhe.EKFArrivalCost(s, builder=o),
        lambda s, o: mhe.UKFArrivalCost(s, builder=o),
    ],
    ids=["steady", "ekf", "ukf"],
)
def test_c_solver_matches_python(arrival_factory):
    pytest.importorskip("acados_template")
    system = _tiny_system()
    obj = _tiny_objective(system, arrival_factory)
    N = 5
    solver = mhe.build_mhe_solver(
        system, N, obj, dt=0.01, already_discrete=True
    )
    rng = np.random.default_rng(1)
    n_steps = N + 12
    y = rng.normal(size=(system.ny, n_steps))
    u = rng.normal(size=(system.nu, n_steps))

    u_py, x_py = mhe.run_solver(solver, y, u)
    u_c, x_c = mhe.run_c_solver(solver, y, u, rebuild=True)

    mask = np.isfinite(u_py) & np.isfinite(u_c)
    assert np.any(mask), "no finite estimates to compare"
    np.testing.assert_allclose(u_c[mask], u_py[mask], rtol=1e-5, atol=1e-5)
    mask_x = np.isfinite(x_py) & np.isfinite(x_c)
    np.testing.assert_allclose(x_c[mask_x], x_py[mask_x], rtol=1e-4, atol=0.05)


def test_ukf_filter_matrices_cached():
    """``build_mhe_solver`` stores plant filter matrices and UKF tuning for in-C UKF."""
    pytest.importorskip("acados_template")
    system = _tiny_system()
    obj = _tiny_objective(system, lambda s, o: mhe.UKFArrivalCost(s, builder=o))
    solver = mhe.build_mhe_solver(system, 5, obj, dt=0.01, already_discrete=True)
    assert solver._filter_kind == "ukf"
    assert solver._filter_A.shape == (2, 2)
    assert solver._filter_alpha == pytest.approx(1e-3)
    assert solver._filter_beta == pytest.approx(2.0)
    assert solver._filter_kappa == pytest.approx(0.0)


def test_filter_setup_w0_points_at_persistent_fortran_buffer():
    """``_build_filter_setup`` must not copy ``W0_template`` (Fortran layout)."""
    pytest.importorskip("acados_template")
    from openmhe.builder.c_runner import _build_filter_setup

    system = _tiny_system()
    obj = _tiny_objective(system, lambda s, o: mhe.EKFArrivalCost(s, builder=o))
    solver = mhe.build_mhe_solver(system, 5, obj, dt=0.01, already_discrete=True)
    setup = _build_filter_setup(solver, has_arrival=True)
    assert setup is not None
    assert setup.W0_template == solver._W0_template_f.ctypes.data


def test_ekf_filter_matrices_cached_for_sparse_process():
    """``build_mhe_solver`` stores plant filter matrices for in-C EKF arrival."""
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
    obj.add(mhe.KnownInput([0]))
    obj.add(mhe.EKFArrivalCost(system, builder=obj))
    solver = mhe.build_mhe_solver(system, 3, obj, dt=0.01, already_discrete=True)
    assert solver._filter_kind == "ekf"
    assert solver._arrival_state_idx is not None
    assert solver._filter_A.shape == (3, 3)
    np.testing.assert_allclose(
        np.diag(solver._filter_Q), [0.01, 0.0, 0.02], atol=0.0
    )
