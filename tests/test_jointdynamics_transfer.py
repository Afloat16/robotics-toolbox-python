"""The actuator transfer denominator is referred to the motor shaft once."""

import numpy as np
import numpy.testing as npt
import pytest

import roboticstoolbox as rtb


def _robot() -> rtb.DHRobot:
    return rtb.DHRobot(
        [
            rtb.RevoluteDH(
                a=0.8,
                m=1.5,
                r=[-0.2, 0, 0],
                I=[0.1, 0.2, 0.3],
                Jm=0.07,
                G=-4,
                B=0.02,
                Tc=[0.3, -0.5],
            ),
            rtb.PrismaticDH(m=2, Jm=0.04, G=3, B=0.06, Tc=[0.2, -0.4]),
        ],
        gravity=[0, 0, 0],
    )


def test_motor_inertia_is_counted_once() -> None:
    robot = _robot()
    q = [0.4, 0.2]
    result = robot.jointdynamics(q)
    # Rotating both masses at radii 0.6 and 0.8 gives the rigid-body inertia;
    # the prismatic joint translates only its own mass.
    rigid = [1.5 * 0.6**2 + 0.3 + 2 * 0.8**2, 2]
    for index, link in enumerate(robot.links):
        npt.assert_allclose(
            result[index][1],
            [
                link.Jm + rigid[index] / link.G**2,
                link.B,
            ],
            atol=1e-13,
        )


@pytest.mark.parametrize("qd", [[0.5, -0.4], [-0.2, 0.7], [0, 0]])
def test_equivalent_motor_damping_reproduces_the_friction_torque(
    qd: list[float],
) -> None:
    robot = _robot()
    result = robot.jointdynamics([0.4, 0.2], qd=np.array(qd))
    for index, link in enumerate(robot.links):
        damping = result[index][1][1]
        if qd[index] == 0:
            npt.assert_allclose(damping, link.B)
        else:
            # Multiply motor damping by joint velocity and the squared gear
            # ratio to recover the joint-referred friction resistance.
            npt.assert_allclose(
                damping * link.G**2 * qd[index],
                -link.friction(qd[index]),
                atol=1e-13,
            )


def test_fixed_links_do_not_become_actuator_transfer_functions() -> None:
    mount = rtb.Link(rtb.ET.tz(0.3), name="mount")
    joint = rtb.Link(
        rtb.ET.Rz(),
        parent=mount,
        name="joint",
        m=2,
        r=[0.4, 0, 0],
        I=[0.02, 0.03, 0.04],
        Jm=0.07,
        G=-3,
        B=0.05,
        Tc=[0.2, -0.4],
    )
    robot = rtb.Robot([mount, joint], gravity=[0, 0, 0])
    result = robot.jointdynamics([0.2], qd=[0.5])
    assert len(result) == 1
    npt.assert_allclose(
        result[0][1],
        [
            0.07 + (2 * 0.4**2 + 0.04) / 9,
            0.05 + 0.2 / (3 * 0.5),
        ],
        atol=1e-13,
    )
