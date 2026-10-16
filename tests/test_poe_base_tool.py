"""PoE kinematics honour the robot's placement and attached tool."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import SE3, Twist3
from spatialmath.base import vex

from roboticstoolbox import PoEPrismatic, PoERevolute, PoERobot
from roboticstoolbox.tools.types import NDArray


def _model() -> tuple[PoERobot, list[Twist3], SE3]:
    home = SE3.Trans(0.3, -0.2, 0.5) * SE3.RPY(0.3, -0.4, 0.2)
    twists = [
        Twist3.UnitRevolute([0, 1, 0], [0.1, 0, 0.3]),
        Twist3.UnitPrismatic([1, 0, 0]),
    ]
    robot = PoERobot(
        [PoERevolute([0, 1, 0], [0.1, 0, 0.3]), PoEPrismatic([1, 0, 0])], home
    )
    return robot, twists, home


def _expected(q: NDArray, twists: list[Twist3], home: SE3, base: SE3, tool: SE3) -> SE3:
    T = base.copy()
    for twist, angle in zip(twists, q):
        T *= twist.exp(angle)
    return T * home * tool


@pytest.mark.parametrize("placement", ["base", "tool", "both", "identity"])
def test_poe_kinematics_in_placed_tool_frame(placement: str) -> None:
    robot, twists, home = _model()
    base = (
        SE3.Trans(1.2, -0.7, 0.4) * SE3.RPY(-0.4, 0.2, 0.6)
        if placement in ("base", "both")
        else SE3()
    )
    tool = (
        SE3.Trans(0.2, -0.1, 0.4) * SE3.RPY(0.5, -0.3, 0.1)
        if placement in ("tool", "both")
        else SE3()
    )
    robot.base, robot.tool = base, tool
    q = np.array([0.3, -0.2])
    expected = _expected(q, twists, home, base, tool)
    nt.assert_allclose(robot.fkine(q).A, expected.A, atol=1e-12)

    h = 1e-6
    J = np.zeros((6, robot.n))
    for i in range(robot.n):
        dq = np.zeros(robot.n)
        dq[i] = h
        dT = (
            _expected(q + dq, twists, home, base, tool).A
            - _expected(q - dq, twists, home, base, tool).A
        ) / (2 * h)
        J[:3, i] = dT[:3, 3]
        J[3:, i] = vex(dT[:3, :3] @ expected.R.T)
    nt.assert_allclose(robot.jacob0(q), J, atol=2e-9)
    frame = np.zeros((6, 6))
    frame[:3, :3] = frame[3:, 3:] = expected.R.T
    nt.assert_allclose(robot.jacobe(q), frame @ J, atol=2e-9)


def test_poe_body_jacobian_is_invariant_to_base_placement() -> None:
    robot, _, _ = _model()
    q = [0.3, -0.2]
    robot.tool = SE3.Trans(0.2, -0.1, 0.4) * SE3.Ry(0.3)
    before = robot.jacobe(q)
    robot.base = SE3.Trans(1, 2, 3) * SE3.RPY(0.4, 0.2, -0.5)
    nt.assert_allclose(robot.jacobe(q), before, atol=1e-12)
