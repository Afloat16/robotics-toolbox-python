"""URDF friction is specified at the joint, independently of motor gearing."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import Robot
from roboticstoolbox.models.URDF.URDFRobot import _links_from_urdf
from roboticstoolbox.tools.urdf import URDF, URDFError


def load_joint(
    joint_type: str = "revolute",
    dynamics: str = 'damping="0.7" friction="0.3"',
    reduction: float | None = None,
    transmission_name: str = "drive",
) -> Robot:
    transmission = ""
    if reduction is not None:
        transmission = f"""<transmission name="{transmission_name}">
          <type>transmission_interface/SimpleTransmission</type>
          <joint name="hinge"/>
          <actuator name="motor">
            <mechanicalReduction>{reduction}</mechanicalReduction>
          </actuator>
        </transmission>"""
    urdf = URDF.loadstr(
        f"""<robot name="friction">
          <link name="base"/><link name="body"/>
          <joint name="hinge" type="{joint_type}">
            <parent link="base"/><child link="body"/><axis xyz="0 0 1"/>
            <limit effort="10" velocity="2" lower="-1" upper="1"/>
            <dynamics {dynamics}/>
          </joint>{transmission}
        </robot>""",
        None,
    )
    links, name = _links_from_urdf(urdf)
    return Robot(links, name=name)


@pytest.mark.parametrize("joint_type", ["revolute", "continuous", "prismatic"])
@pytest.mark.parametrize("reduction", [None, 5.0, -5.0])
def test_joint_friction_matches_joint_side_force_law(
    joint_type: str, reduction: float | None
) -> None:
    robot = load_joint(joint_type=joint_type, reduction=reduction)
    link = next(link for link in robot.links if link.isjoint)
    nt.assert_allclose(link.G, 1.0 if reduction is None else reduction)
    for velocity in [-1.8, -0.2, 0.0, 0.2, 1.8]:
        expected = -0.7 * velocity - 0.3 * np.sign(velocity)
        nt.assert_allclose(link.friction(velocity), expected, atol=1e-14)
        nt.assert_allclose(
            link.friction(velocity, coulomb=False), -0.7 * velocity, atol=1e-14
        )


@pytest.mark.parametrize(
    ("dynamics", "damping", "friction"),
    [('damping="0.7"', 0.7, 0.0), ('friction="0.3"', 0.0, 0.3), ("", 0.0, 0.0)],
)
def test_omitted_friction_terms_remain_zero(
    dynamics: str, damping: float, friction: float
) -> None:
    robot = load_joint(dynamics=dynamics)
    link = next(link for link in robot.links if link.isjoint)
    for velocity in [-0.4, 0.0, 0.4]:
        nt.assert_allclose(
            link.friction(velocity),
            -damping * velocity - friction * np.sign(velocity),
        )


@pytest.mark.parametrize("transmission_name", ["drive", "hinge"])
def test_transmission_name_does_not_change_its_joint_binding(
    transmission_name: str,
) -> None:
    robot = load_joint(reduction=4.0, transmission_name=transmission_name)
    link = next(link for link in robot.links if link.isjoint)
    nt.assert_allclose(link.G, 4.0)
    nt.assert_allclose(link.B, 0.7 / 16.0)
    nt.assert_allclose(link.Tc, [0.075, -0.075])


@pytest.mark.parametrize("reduction", [0.0, float("nan"), float("inf")])
def test_invalid_reduction_cannot_produce_non_finite_friction(reduction: float) -> None:
    with pytest.raises(URDFError, match="mechanicalReduction.*finite.*non-zero"):
        load_joint(reduction=reduction)
