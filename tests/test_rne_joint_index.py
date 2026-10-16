"""Inverse-dynamics efforts use the same coordinates as the input state."""

import numpy as np
import numpy.testing as npt
import pytest

import roboticstoolbox as rtb


def _arm(indices: tuple[int, int]) -> rtb.Robot:
    first = rtb.Link(
        rtb.ET.Rz(jindex=indices[0]),
        name="first",
        m=1.2,
        r=[0.4, 0.1, 0],
        I=[0.02, 0.03, 0.04],
        Jm=0.01,
        G=3,
        B=0.03,
        Tc=[0.1, -0.2],
    )
    second = rtb.Link(
        rtb.ET.tx(0.7) * rtb.ET.Ry(jindex=indices[1]),
        name="second",
        parent=first,
        m=2.3,
        r=[0.2, 0.1, 0.3],
        I=[0.06, 0.07, 0.08],
        Jm=0.02,
        G=-2,
        B=0.04,
        Tc=[0.2, -0.1],
    )
    return rtb.Robot([first, second], gravity=[1.1, -2.3, -8.7])


@pytest.mark.parametrize("trajectory", [False, True])
def test_inverse_dynamics_is_covariant_under_joint_permutation(
    trajectory: bool,
) -> None:
    reference = _arm((0, 1))
    robot = _arm((1, 0))
    q = np.array([[0.2, -0.4], [-0.5, 0.3]])
    qd = np.array([[0.4, -0.2], [-0.3, 0.6]])
    qdd = np.array([[-0.2, 0.7], [0.1, -0.5]])
    if not trajectory:
        q, qd, qdd = q[0], qd[0], qdd[0]

    expected = reference.rne(q[..., ::-1], qd[..., ::-1], qdd[..., ::-1])
    npt.assert_allclose(robot.rne(q, qd, qdd), expected[..., ::-1], atol=1e-13)


def test_inertia_and_forward_dynamics_keep_coordinate_order() -> None:
    reference = _arm((0, 1))
    robot = _arm((1, 0))
    q = np.array([0.2, -0.4])
    qd = np.array([0.3, -0.5])
    tau = np.array([1.2, -0.7])
    expected = reference.inertia(q[::-1])[::-1, ::-1]

    npt.assert_allclose(robot.inertia(q), expected, atol=1e-13)
    npt.assert_allclose(robot.inertia(q), robot.inertia(q).T, atol=1e-13)
    npt.assert_allclose(
        robot.accel(q, qd, tau),
        reference.accel(q[::-1], qd[::-1], tau[::-1])[::-1],
        atol=1e-12,
    )
