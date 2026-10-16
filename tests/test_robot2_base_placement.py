"""Planar forward kinematics includes the configured world placement."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import SE2

from roboticstoolbox import ET2, Robot2


@pytest.mark.parametrize("assignment", ["constructor", "property"])
@pytest.mark.parametrize("array_base", [False, True])
def test_planar_robot_base_placement(assignment: str, array_base: bool) -> None:
    ets = ET2.R() * ET2.tx(0.7) * ET2.ty()
    base = SE2(1.3, -0.8, 0.6)
    argument = base.A if array_base else base
    robot = Robot2(ets, base=argument) if assignment == "constructor" else Robot2(ets)
    original_tool = np.copy(robot._tool)
    if assignment == "property":
        robot.base = argument
    nt.assert_allclose(robot.base.A, base.A, atol=1e-12)
    nt.assert_allclose(robot._tool, original_tool, atol=0)
    q = [0.3, -0.2]
    expected = base * SE2(q[0]) * SE2.Tx(0.7) * SE2.Ty(q[1])
    nt.assert_allclose(
        np.asarray(robot.fkine(q).A, dtype=float), expected.A, atol=1e-12
    )
    nt.assert_allclose(robot.fkine_all(q)[-1].A, expected.A, atol=1e-12)


def test_planar_robot_placed_trajectory() -> None:
    robot = Robot2(ET2.R() * ET2.tx(0.7))
    robot.base = SE2(-0.2, 0.4, -0.3)
    poses = robot.fkine(np.array([[0.1], [0.4], [-0.2]]))
    for q, pose in zip([0.1, 0.4, -0.2], poses):
        expected = robot.base * SE2(q) * SE2.Tx(0.7)
        nt.assert_allclose(np.asarray(pose.A, dtype=float), expected.A, atol=1e-12)


def test_planar_identity_base_control() -> None:
    ets = ET2.tx(0.4) * ET2.R() * ET2.tx(0.7)
    robot = Robot2(ets)
    nt.assert_allclose(
        np.asarray(robot.fkine([0.3]).A, dtype=float),
        np.asarray(ets.fkine([0.3]).A, dtype=float),
        atol=1e-12,
    )
