"""Flipped DH coordinates obey virtual power and coordinate covariance."""

import numpy as np
import numpy.testing as npt
import pytest
from spatialmath import SE3

import roboticstoolbox as rtb
from roboticstoolbox.robot.frne import _C_AVAILABLE


def _arm(mdh: bool, flipped: tuple[bool, bool], motors: bool = False) -> rtb.DHRobot:
    revolute = rtb.RevoluteMDH if mdh else rtb.RevoluteDH
    prismatic = rtb.PrismaticMDH if mdh else rtb.PrismaticDH
    parameters = [
        {
            "a": 0.7,
            "d": 0.2,
            "alpha": 0.4,
            "offset": 0.1,
            "m": 1.3,
            "r": [0.2, -0.1, 0.3],
            "I": [0.04, 0.05, 0.06],
        },
        {
            "a": 0.3,
            "theta": -0.2,
            "alpha": -0.5,
            "offset": 0.2,
            "m": 2.1,
            "r": [-0.1, 0.2, 0.1],
            "I": [0.07, 0.08, 0.09],
        },
    ]
    links = []
    for index, (constructor, values) in enumerate(
        zip([revolute, prismatic], parameters)
    ):
        if motors:
            values.update(
                Jm=0.03 + 0.01 * index,
                G=[-3, 2][index],
                B=0.02 + 0.01 * index,
                Tc=[0.2, -0.4],
            )
        links.append(constructor(**values, flip=flipped[index]))
    return rtb.DHRobot(links, base=SE3.Rx(0.3), gravity=[1.2, -2.1, -8.3])


@pytest.mark.parametrize("mdh", [False, True])
@pytest.mark.parametrize("motors", [False, True])
@pytest.mark.parametrize("trajectory", [False, True])
@pytest.mark.parametrize("method", ["rne_python", "rne"])
@pytest.mark.parametrize("flip_mask", [(True, False), (False, True), (True, True)])
def test_flipped_coordinates_transform_velocity_acceleration_and_effort(
    mdh: bool,
    motors: bool,
    trajectory: bool,
    method: str,
    flip_mask: tuple[bool, bool],
) -> None:
    robot = _arm(mdh, flip_mask, motors)
    reference = _arm(mdh, (False, False))
    direction = np.where(flip_mask, -1, 1)
    q = np.array([[0.3, 0.4], [-0.5, 0.2]])
    qd = np.array([[0.4, -0.2], [-0.3, 0.5]])
    qdd = np.array([[-0.2, 0.7], [0.3, -0.4]])
    if not trajectory:
        q, qd, qdd = q[0], qd[0], qdd[0]
    wrench = [0.2, -0.4, 0.1, 0.3, -0.2, 0.5]
    npt.assert_allclose(robot.fkine(q).A, reference.fkine(direction * q).A, atol=1e-14)
    physical, expected_base = reference.rne_python(
        direction * q, direction * qd, direction * qdd, fext=wrench, base_wrench=True
    )
    expected = direction * physical
    for index, link in enumerate(robot.links):
        expected[..., index] += link.G**2 * link.Jm * qdd[..., index]
        for row in range(1 if q.ndim == 1 else len(q)):
            velocity = qd[index] if q.ndim == 1 else qd[row, index]
            if q.ndim == 1:
                expected[index] -= link.friction(velocity)
            else:
                expected[row, index] -= link.friction(velocity)
    actual, actual_base = getattr(robot, method)(
        q, qd, qdd, fext=wrench, base_wrench=True
    )
    npt.assert_allclose(actual, expected, atol=2e-12)
    npt.assert_allclose(actual_base, expected_base, atol=2e-12)


@pytest.mark.parametrize("mdh", [False, True])
def test_flipped_gravity_effort_matches_potential_gradient(mdh: bool) -> None:
    link_type = rtb.PrismaticMDH if mdh else rtb.PrismaticDH
    robot = rtb.DHRobot([link_type(flip=True, offset=0.4, m=2)], gravity=[0, 0, -9.81])
    # U = -m g_z (-q + offset), so dU/dq = m g_z.
    npt.assert_allclose(robot.rne_python([0.2], [0], [0]), [-19.62], atol=1e-13)


@pytest.mark.skipif(not _C_AVAILABLE, reason="compiled _frne_c extension not built")
@pytest.mark.parametrize("mdh", [False, True])
def test_native_and_python_paths_both_execute_for_flipped_joints(mdh: bool) -> None:
    robot = _arm(mdh, (True, True), True)
    q, qd, qdd = [0.2, 0.3], [0.4, -0.2], [-0.1, 0.5]
    native = robot.rne(q, qd, qdd)
    assert robot._frne is not None
    npt.assert_allclose(native, robot.rne_python(q, qd, qdd), atol=1e-12)


def _ets_arm(flipped: tuple[bool, bool], motors: bool) -> rtb.Robot:
    links = []
    for index in range(2):
        ets = (
            rtb.ET.Rx(0.3) * rtb.ET.Rz(jindex=0, flip=flipped[0])
            if index == 0
            else rtb.ET.tx(0.7) * rtb.ET.tz(jindex=1, flip=flipped[1])
        )
        links.append(
            rtb.Link(
                ets,
                name=f"joint{index}",
                parent=links[-1] if links else None,
                m=1.3 + index,
                r=[0.2, -0.1, 0.3],
                I=[0.04, 0.05, 0.06],
                Jm=0.03 if motors else 0,
                G=-3 if motors else 1,
                B=0.02 if motors else 0,
                Tc=[0.2, -0.4] if motors else [0, 0],
            )
        )
    return rtb.Robot(links, base=SE3.Ry(0.2), gravity=[1.2, -2.1, -8.3])


@pytest.mark.parametrize("flip_mask", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("motors", [False, True])
@pytest.mark.parametrize("trajectory", [False, True])
def test_ets_inverse_dynamics_obeys_the_same_flipped_coordinate_mapping(
    flip_mask: tuple[bool, bool],
    motors: bool,
    trajectory: bool,
) -> None:
    robot = _ets_arm(flip_mask, motors)
    reference = _ets_arm((False, False), False)
    direction = np.where(flip_mask, -1, 1)
    q = np.array([[0.3, 0.4], [-0.5, 0.2]])
    qd = np.array([[0.4, -0.2], [-0.3, 0.5]])
    qdd = np.array([[-0.2, 0.7], [0.3, -0.4]])
    if not trajectory:
        q, qd, qdd = q[0], qd[0], qdd[0]
    npt.assert_allclose(robot.fkine(q).A, reference.fkine(direction * q).A, atol=1e-14)
    expected = direction * reference.rne(direction * q, direction * qd, direction * qdd)
    for index, link in enumerate(robot.links):
        expected[..., index] += link.G**2 * link.Jm * qdd[..., index]
        if q.ndim == 1:
            expected[index] -= link.friction(qd[index])
        else:
            for row in range(len(q)):
                expected[row, index] -= link.friction(qd[row, index])
    npt.assert_allclose(robot.rne(q, qd, qdd), expected, atol=2e-12)


def test_ets_flipped_gravity_effort_matches_potential_gradient() -> None:
    robot = rtb.Robot([rtb.Link(rtb.ET.tz(flip=True), m=2)], gravity=[0, 0, -9.81])
    npt.assert_allclose(robot.rne([0.2], [0], [0]), [-19.62], atol=1e-13)
