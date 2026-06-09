"""Regression tests for LTI + LINEAR_LS fast C solve path."""

from __future__ import annotations

import numpy as np
import pytest

import openmhe as mhe


def _tiny_lti_problem():
    """Minimal 1-state LTI plant with one measurement."""
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

    solver = mhe.build_mhe_solver(system, 5, builder, dt=dt, lti_linear_ls_fast=True)
    return solver, dt


def _run_both(solver, n_steps: int = 40):
    rng = np.random.default_rng(0)
    t = np.arange(n_steps) * solver._dt
    u = np.vstack([0.2 * np.sin(t), 0.1 * np.cos(0.5 * t)])
    y = 0.3 * np.sin(t).reshape(1, -1) + 0.02 * rng.standard_normal((1, n_steps))

    u_ref, x_ref = mhe.run_solver(solver, y, u)
    u_fast, x_fast = mhe.run_c_solver(solver, y, u, lti_linear_ls_fast=True, rebuild=True)
    u_slow, x_slow = mhe.run_c_solver(solver, y, u, lti_linear_ls_fast=False, rebuild=False)
    return (u_ref, x_ref), (u_fast, x_fast), (u_slow, x_slow)


@pytest.mark.slow
def test_lti_fast_matches_python_and_full_c():
    solver, _ = _tiny_lti_problem()
    (u_ref, x_ref), (u_fast, x_fast), (u_slow, x_slow) = _run_both(solver)

    np.testing.assert_allclose(u_fast, u_slow, rtol=0, atol=1e-6)
    np.testing.assert_allclose(x_fast, x_slow, rtol=0, atol=1e-6)
    mask = np.isfinite(u_ref) & np.isfinite(u_fast)
    np.testing.assert_allclose(u_fast[mask], u_ref[mask], rtol=1e-5, atol=1e-4)
    mask_x = np.isfinite(x_ref) & np.isfinite(x_fast)
    np.testing.assert_allclose(x_fast[mask_x], x_ref[mask_x], rtol=1e-4, atol=0.05)


@pytest.mark.slow
def test_lti_fast_known_inputs():
    pytest.importorskip("acados_template")
    system = mhe.SystemModel.from_matrices(
        np.array([[0.9, 0.0], [0.0, 0.8]]),
        np.array([[0.1, 0.0], [0.0, 0.05]]),
        np.array([[1.0, 0.0]]),
        np.zeros((1, 2)),
        is_discrete=True,
        dt=0.01,
    )
    builder = mhe.ObjectiveBuilder()
    builder.add(mhe.MeasurementTerm(mhe.L2Penalty(), mhe.NoiseWeight(1, cov=0.1)))
    builder.add(mhe.ProcessTerm(mhe.L2Penalty(), mhe.NoiseWeight(2, cov=0.01)))
    builder.add(mhe.KnownInput([0, 1]))
    builder.add(mhe.SteadyStateArrivalCost())
    solver = mhe.build_mhe_solver(system, 5, builder, dt=0.01, already_discrete=True)
    rng = np.random.default_rng(1)
    y = rng.normal(size=(1, 17))
    u = rng.normal(size=(2, 17))
    u_fast, _ = mhe.run_c_solver(solver, y, u, lti_linear_ls_fast=True, rebuild=True)
    u_slow, _ = mhe.run_c_solver(solver, y, u, lti_linear_ls_fast=False, rebuild=False)
    np.testing.assert_allclose(u_fast, u_slow, rtol=0, atol=1e-6)


@pytest.mark.slow
def test_lti_fast_ekf_arrival():
    pytest.importorskip("acados_template")
    A = np.array([[0.9]])
    B = np.array([[0.1, 0.05]])
    C = np.array([[1.0]])
    D = np.zeros((1, 2))
    system = mhe.SystemModel.from_matrices(A, B, C, D, is_discrete=True, dt=0.1)
    builder = mhe.ObjectiveBuilder()
    builder.add(mhe.MeasurementTerm(mhe.L2Penalty(), mhe.NoiseWeight(1, cov=0.5)))
    builder.add(mhe.ProcessTerm(mhe.L2Penalty(), mhe.NoiseWeight(1, cov=0.01)))
    builder.add(mhe.InputRandomWalk(target_idx=[0, 1], lambda_u=1.0))
    builder.add(mhe.EKFArrivalCost(system, builder=builder))
    solver = mhe.build_mhe_solver(system, 5, builder, dt=0.1, already_discrete=True)
    rng = np.random.default_rng(2)
    n_steps = 30
    y = rng.normal(size=(1, n_steps))
    u = rng.normal(size=(2, n_steps))
    u_fast, x_fast = mhe.run_c_solver(solver, y, u, lti_linear_ls_fast=True, rebuild=True)
    u_slow, x_slow = mhe.run_c_solver(solver, y, u, lti_linear_ls_fast=False, rebuild=False)
    np.testing.assert_allclose(u_fast, u_slow, rtol=0, atol=1e-5)
    np.testing.assert_allclose(x_fast, x_slow, rtol=0, atol=1e-4)


@pytest.mark.slow
def test_lti_fast_disabled_for_non_rti_solver():
    solver, dt = _tiny_lti_problem()
    system = solver._system
    builder = solver._builder
    with pytest.warns(UserWarning, match="SQP_RTI"):
        sqp_solver = mhe.build_mhe_solver(
            system,
            solver._N,
            builder,
            dt=dt,
            nlp_solver_type="SQP",
            lti_linear_ls_fast=True,
        )
    assert sqp_solver._lti_linear_ls_fast is False

    rng = np.random.default_rng(3)
    n_steps = 25
    t = np.arange(n_steps) * dt
    u = np.vstack([0.2 * np.sin(t), 0.1 * np.cos(0.5 * t)])
    y = 0.3 * np.sin(t).reshape(1, -1) + 0.02 * rng.standard_normal((1, n_steps))
    with pytest.warns(UserWarning, match="SQP_RTI"):
        u_try_fast, _ = mhe.run_c_solver(
            sqp_solver, y, u, lti_linear_ls_fast=True, rebuild=True
        )
    u_full, _ = mhe.run_c_solver(
        sqp_solver, y, u, lti_linear_ls_fast=False, rebuild=False
    )
    np.testing.assert_allclose(u_try_fast, u_full, rtol=0, atol=1e-6)


@pytest.mark.slow
def test_codegen_extra_header_written():
    solver, _ = _tiny_lti_problem()
    from openmhe.paths import get_codegen_dir

    header = get_codegen_dir() / "openmhe_mhe_extra.h"
    source = get_codegen_dir() / "openmhe_mhe_extra.c"
    assert header.is_file()
    assert source.is_file()
    assert "OPENMHE_MHE_LINEAR_LS 1" in header.read_text(encoding="utf-8")
    assert solver._linear_ls is True
