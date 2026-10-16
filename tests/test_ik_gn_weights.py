"""Analytic weighted least-squares references for Gauss-Newton IK."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import SE3

import roboticstoolbox as rtb
from roboticstoolbox.ets.ETS import ETS


@pytest.mark.parametrize("pinv", [False, True])
@pytest.mark.parametrize("weights", [[9, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
def test_underactuated_gn_step_minimises_weighted_pose_error(
    pinv: bool, weights: list[float]
) -> None:
    angle = 0.4
    ets = rtb.ET.Rz(angle) * rtb.ET.tx(qlim=[-5, 5])
    q = np.array([0.0])
    target = SE3(0.3, -0.2, 0) * SE3.Rz(angle)
    solver = rtb.IK_GN(pinv=pinv, mask=weights)

    residual, result = solver.step(ets, target.A, q)

    # A single prismatic axis has displacement q * [cos(a), sin(a)].
    # Differentiate wx*(q*cos(a)-x)**2 + wy*(q*sin(a)-y)**2.
    c, s = np.cos(angle), np.sin(angle)
    wx, wy = weights[:2]
    expected = (wx * c * 0.3 - wy * s * 0.2) / (wx * c**2 + wy * s**2)
    nt.assert_allclose(result, [expected], atol=1e-14)
    expected_error = 0.5 * (
        wx * (c * expected - 0.3) ** 2 + wy * (s * expected + 0.2) ** 2
    )
    nt.assert_allclose(residual, expected_error, atol=1e-14)


@pytest.mark.parametrize("pinv", [False, True])
def test_gn_solves_only_the_selected_cartesian_coordinate(pinv: bool) -> None:
    ets = rtb.ET.Rz(0.4) * rtb.ET.tx(qlim=[-5, 5])
    target = SE3(0.3, -0.2, 0) * SE3.Rz(0.4)
    solver = rtb.IK_GN(
        pinv=pinv, mask=[1, 0, 0, 0, 0, 0], slimit=1, ilimit=2, tol=1e-20
    )

    result = solver.solve(ets, target, q0=[0.0])

    assert result.success
    nt.assert_allclose(ets.eval(result.q)[0, 3], 0.3, atol=1e-14)
    assert result.iterations == 1


def test_masked_gn_nullspace_preserves_selected_task() -> None:
    ets = ETS(
        [
            rtb.ET.tx(qlim=[-1, 1]),
            rtb.ET.ty(qlim=[-1, 1]),
        ]
    )
    q = np.array([0.0, 0.9])
    solver = rtb.IK_GN(pinv=True, mask=[1, 0, 0, 0, 0, 0], kq=1.0, ps=0.1, pi=0.4)

    _, result = solver.step(ets, SE3(0.2, 0.9, 0).A, q)

    # The ignored y coordinate is available for joint-limit avoidance, while
    # x remains the exact primary solution.
    nt.assert_allclose(result, [0.2, -0.1], atol=1e-14)
