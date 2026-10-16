"""Fixed-link insertion must not change the mechanics of the same bodies."""

import numpy as np
import numpy.testing as npt
import pytest
from spatialmath import SE3

import roboticstoolbox as rtb
from roboticstoolbox.tools.types import NDArray


def _segmented_arm(segmented: bool) -> rtb.Robot:
    root_ets = rtb.ET.Rx(0.4) * rtb.ET.tz(0.3)
    spacer_ets = rtb.ET.tx(0.7) * rtb.ET.Ry(-0.3)
    first_ets = rtb.ET.Rz(jindex=0)
    second_ets = rtb.ET.Rx(0.2) * rtb.ET.Ry(jindex=1)
    links = []
    parent = None
    if segmented:
        parent = rtb.Link(root_ets, name="mount")
        links.append(parent)
    else:
        first_ets = root_ets * first_ets
    first = rtb.Link(
        first_ets,
        name="first",
        parent=parent,
        m=1.3,
        r=[0.4, 0.1, -0.2],
        I=[0.05, 0.06, 0.07],
    )
    links.append(first)
    parent = first
    if segmented:
        parent = rtb.Link(spacer_ets, name="spacer", parent=first)
        links.append(parent)
    else:
        second_ets = spacer_ets * second_ets
    links.append(
        rtb.Link(
            second_ets,
            name="second",
            parent=parent,
            m=2.1,
            r=[0.3, -0.1, 0.2],
            I=[0.08, 0.09, 0.1],
        )
    )
    return rtb.Robot(links, base=SE3.Ry(0.2), gravity=[1.2, -2.4, -8.5])


@pytest.mark.parametrize("mode", ["gravity", "inertia", "moving"])
def test_fixed_transforms_equal_the_same_geometry_within_joint_links(mode: str) -> None:
    robot = _segmented_arm(True)
    reference = _segmented_arm(False)
    q = np.array([[0.2, -0.3], [-0.5, 0.7]])
    qd = np.array([[0.4, -0.2], [-0.1, 0.3]])
    qdd = np.array([[0.2, 0.5], [-0.3, 0.4]])
    npt.assert_allclose(robot.fkine(q).A, reference.fkine(q).A, atol=1e-14)
    if mode == "gravity":
        qd = qdd = np.zeros_like(q)
    elif mode == "inertia":
        qd = np.zeros_like(q)
        robot.gravity = reference.gravity = [0, 0, 0]
    npt.assert_allclose(robot.rne(q, qd, qdd), reference.rne(q, qd, qdd), atol=1e-12)


@pytest.mark.parametrize("reverse", [False, True])
def test_fixed_sibling_bodies_use_their_own_parent_transform(reverse: bool) -> None:
    root = rtb.Link(rtb.ET.Rz(), name="joint")
    left = rtb.Link(
        rtb.ET.tx(0.4) * rtb.ET.Rx(0.3),
        name="left",
        parent=root,
        m=1.2,
        r=[0.2, -0.1, 0.3],
        I=[0.03, 0.04, 0.05],
    )
    right = rtb.Link(
        rtb.ET.ty(-0.7) * rtb.ET.Ry(-0.2),
        name="right",
        parent=root,
        m=2.3,
        r=[-0.1, 0.2, 0.1],
        I=[0.06, 0.07, 0.08],
    )
    leaves = [left, right] if not reverse else [right, left]
    robot = rtb.Robot([root, *leaves], gravity=[2, -8, 1])
    q, qd, qdd = 0.4, 0.7, -0.3
    # Independent rigid-body energy and potential oracles about the z-axis.
    inertia = 0.0
    gravity_torque = 0.0
    for leaf in leaves:
        pose = SE3(leaf.A())
        centre = pose.R @ leaf.r + pose.t
        inertia += leaf.m * np.dot(centre[:2], centre[:2])
        inertia += (pose.R @ leaf.I @ pose.R.T)[2, 2]
        world_centre = SE3.Rz(q).R @ centre
        gravity_torque -= leaf.m * np.cross(world_centre, robot.gravity)[2]
    npt.assert_allclose(
        robot.rne([q], [qd], [qdd]), [inertia * qdd + gravity_torque], atol=1e-12
    )


def test_nested_fixed_branches_equal_rigid_body_superposition() -> None:
    joint = rtb.Link(rtb.ET.Rz(), name="joint")
    left = rtb.Link(
        rtb.ET.tx(0.4) * rtb.ET.Rx(0.3),
        name="left",
        parent=joint,
        m=1.2,
        r=[0.2, -0.1, 0.3],
        I=[0.03, 0.04, 0.05],
    )
    tip = rtb.Link(
        rtb.ET.ty(0.3),
        name="tip",
        parent=left,
        m=0.7,
        r=[0.1, 0.2, -0.1],
        I=[0.01, 0.02, 0.03],
    )
    right = rtb.Link(
        rtb.ET.tz(0.5) * rtb.ET.Ry(-0.2),
        name="right",
        parent=joint,
        m=2.3,
        r=[-0.1, 0.2, 0.1],
        I=[0.06, 0.07, 0.08],
    )
    robot = rtb.Robot([joint, left, tip, right], gravity=[2, -8, 1])
    q = [0.4]
    gravity_torque = 0.0
    inertia = 0.0
    for link in [left, tip, right]:
        pose = SE3(link.A())
        if link is tip:
            pose = SE3(left.A()) * pose
        centre = pose.R @ link.r + pose.t
        inertia += link.m * np.dot(centre[:2], centre[:2])
        inertia += (pose.R @ link.I @ pose.R.T)[2, 2]
        gravity_torque -= link.m * np.cross(SE3.Rz(q[0]).R @ centre, robot.gravity)[2]
    npt.assert_allclose(
        robot.rne(q, [0.7], [-0.3]), [inertia * -0.3 + gravity_torque], atol=1e-12
    )


@pytest.mark.parametrize("q", [[0.2, -0.3], [-0.5, 0.7]])
def test_fixed_mount_and_spacer_gravity_matches_potential_gradient(
    q: list[float],
) -> None:
    robot = _segmented_arm(True)
    step = 1e-6
    expected = np.zeros(2)

    def potential(state: NDArray) -> float:
        value = 0.0
        for link in robot.links:
            pose = robot.fkine(state, end=link)
            centre = pose.R @ link.r + pose.t
            value -= link.m * np.dot(robot.gravity, centre)
        return value

    for index in range(2):
        plus, minus = np.array(q), np.array(q)
        plus[index] += step
        minus[index] -= step
        expected[index] = (potential(plus) - potential(minus)) / (2 * step)

    npt.assert_allclose(robot.rne(q, [0, 0], [0, 0]), expected, atol=3e-9)
