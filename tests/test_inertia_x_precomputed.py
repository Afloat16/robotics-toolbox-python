"""Trajectory inertia honors the same supplied inverse as a single pose."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import DHRobot, models


@pytest.mark.parametrize("model", [models.DH.Puma560, models.DH.Panda])
@pytest.mark.parametrize("rows", [0, 1, 3])
def test_trajectory_uses_supplied_inverse(model: type[DHRobot], rows: int) -> None:
    robot = model()
    q0 = robot.qn if robot.n == 6 else robot.qr
    inverse = np.linalg.pinv(robot.jacob0_analytical(q0, "rpy/xyz"))
    saved_inverse = inverse.copy()
    q = np.array([q0 + 0.1 * k for k in range(rows)]).reshape(rows, robot.n)

    result = robot.inertia_x(q, Ji=inverse)
    if rows == 0:
        assert result.shape == (0, 6, 6)
    elif rows == 1:
        nt.assert_allclose(result, robot.inertia_x(q[0], Ji=inverse), atol=1e-12)
    else:
        expected = np.stack([robot.inertia_x(qk, Ji=inverse) for qk in q])
        nt.assert_allclose(result, expected, atol=1e-12)
        # The supplied velocity map must preserve kinetic energy at each pose.
        velocity = np.array([0.2, -0.1, 0.3, -0.2, 0.4, 0.1])
        joint_velocity = inverse @ velocity
        for qk, inertia in zip(q, result):
            nt.assert_allclose(
                velocity @ inertia @ velocity,
                joint_velocity @ robot.inertia(qk) @ joint_velocity,
                atol=1e-12,
            )
    nt.assert_array_equal(inverse, saved_inverse)


def test_supplied_pseudoinverse_handles_a_singular_trajectory() -> None:
    robot = models.DH.Puma560()
    q = np.tile(robot.qz, (2, 1))
    inverse = np.linalg.pinv(robot.jacob0_analytical(robot.qz, "rpy/xyz"))
    expected = robot.inertia_x(robot.qz, Ji=inverse)
    nt.assert_allclose(robot.inertia_x(q, Ji=inverse), np.tile(expected, (2, 1, 1)))


@pytest.mark.parametrize("model", [models.DH.Puma560, models.DH.Panda])
def test_trajectory_without_inverse_evaluates_each_pose(model: type[DHRobot]) -> None:
    robot = model()
    q0 = robot.qn if robot.n == 6 else robot.qr
    q = np.stack([q0, q0 + 0.1, q0 - 0.1])
    expected = np.stack([robot.inertia_x(qk) for qk in q])
    nt.assert_allclose(robot.inertia_x(q), expected, atol=1e-12)
