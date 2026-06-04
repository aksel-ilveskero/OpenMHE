"""Tests for B-column input partition validation."""

import pytest
import openmhe as mhe


def _base(ny=1, nx=2):
    """Minimal objective with measurement and process terms only."""
    obj = mhe.ObjectiveBuilder()
    obj.add(mhe.MeasurementTerm(mhe.L2Penalty(), mhe.NoiseWeight(ny, cov=0.1)))
    obj.add(mhe.ProcessTerm(mhe.L2Penalty(), mhe.NoiseWeight(nx, cov=0.01)))
    return obj


def test_full_partition_known_and_rw():
    """Valid partition (known motor + RW load) builds without error."""
    obj = _base(ny=1, nx=2)
    obj.add(mhe.KnownInput([0]))
    obj.add(mhe.InputRandomWalk([1], lambda_u=1.0))
    A = [[0.9, 0.0], [0.0, 0.9]]
    B = [[0.1, 0.05], [0.0, 0.1]]
    C = [[1.0, 0.0]]
    mhe.build_mhe_solver(A, B, C, None, 10, obj, dt=0.01, already_discrete=True, P_arrival=None)


def test_missing_input_raises():
    """Unassigned input index raises at solver build time."""
    obj = _base()
    obj.add(mhe.InputRandomWalk([1], lambda_u=1.0))
    with pytest.raises(ValueError, match="Unassigned: \\[0\\]"):
        mhe.build_mhe_solver(
            [[0.9]], [[0.1, 0.05]], [[1, 0]], None,
            10, obj, dt=0.01, already_discrete=True, P_arrival=None,
        )


def test_overlap_known_and_rw_raises():
    """Same index cannot be both KnownInput and regulated."""
    obj = _base()
    obj.add(mhe.KnownInput([1]))
    obj.add(mhe.InputRandomWalk([1], lambda_u=1.0))
    with pytest.raises(ValueError, match="KnownInput and regulated"):
        mhe.build_mhe_solver(
            [[0.9]], [[0.1, 0.05]], [[1, 0]], None,
            10, obj, dt=0.01, already_discrete=True, P_arrival=None,
        )


def test_overlap_tracked_and_known_raises():
    """Same index cannot be both KnownInput and InputTrackingTerm."""
    obj = _base()
    obj.add(mhe.KnownInput([0]))
    obj.add(mhe.InputTrackingTerm([0], mhe.NoiseWeight(1, cov=1e-6)))
    obj.add(mhe.InputRandomWalk([1], lambda_u=1.0))
    with pytest.raises(ValueError, match="KnownInput and InputTrackingTerm"):
        mhe.build_mhe_solver(
            [[0.9]], [[0.1, 0.05]], [[1, 0]], None,
            10, obj, dt=0.01, already_discrete=True, P_arrival=None,
        )


def test_known_input_pins_motor():
    """KnownInput keeps motor channel equal to passed ``u`` in run_solver."""
    import numpy as np

    obj = _base(ny=1, nx=2)
    obj.add(mhe.KnownInput([0]))
    obj.add(mhe.InputRandomWalk([1], lambda_u=1.0))
    A = [[0.9, 0.0], [0.0, 0.9]]
    B = [[0.1, 0.05], [0.0, 0.1]]
    C = [[1.0, 0.0]]
    s = mhe.build_mhe_solver(A, B, C, None, 5, obj, dt=0.01, already_discrete=True, P_arrival=None)
    u = np.array([[1.0] * 20, [0.5] * 20])
    y = np.ones((1, 20)) * 0.1
    uh, _ = mhe.run_solver(s, y, u)
    assert np.sqrt(np.nanmean((uh[0, :] - 1.0) ** 2)) < 1e-6
