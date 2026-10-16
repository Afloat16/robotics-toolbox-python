"""Integer driver commands retain fractional physical control limits."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import Bicycle, DiffSteer, Unicycle


@pytest.mark.parametrize("model", [Bicycle, Unicycle, DiffSteer])
@pytest.mark.parametrize("sign", [-1, 1])
def test_integer_controls_preserve_fractional_speed_limits(
    model: type[Bicycle | Unicycle | DiffSteer], sign: int
) -> None:
    robot = model(speed_max=0.4)
    command = [sign, sign] if model is DiffSteer else [sign, 0]
    limited = robot.u_limited(command)
    expected = [sign * 0.4, sign * 0.4] if model is DiffSteer else [sign * 0.4, 0.0]
    nt.assert_allclose(limited, expected)
    nt.assert_allclose(robot.deriv([0, 0, 0], command), [sign * 0.4, 0.0, 0.0])


@pytest.mark.parametrize("model", [Bicycle, Unicycle, DiffSteer])
@pytest.mark.parametrize("sign", [-1, 1])
def test_integer_driver_commands_accelerate_instead_of_stalling(
    model: type[Bicycle | Unicycle | DiffSteer], sign: int
) -> None:
    robot = model(dt=0.1, accel_max=1.0)
    command = [sign, sign] if model is DiffSteer else [sign, 0]
    robot.control = lambda vehicle, t, x: command
    expected_position = 0.0
    for step in range(1, 6):
        robot.step(animate=False)
        expected_position += 0.1 * sign * (step * 0.1)
        nt.assert_allclose(robot.x, [expected_position, 0.0, 0.0], atol=1e-14)


@pytest.mark.parametrize("model", [Bicycle, Unicycle])
def test_integer_commands_preserve_fractional_steering_limits(
    model: type[Bicycle | Unicycle],
) -> None:
    robot = model(steer_max=0.3)
    nt.assert_allclose(robot.u_limited([1, 1]), [1.0, 0.3])
    expected_turn = np.tan(0.3) if model is Bicycle else 0.3
    nt.assert_allclose(robot.deriv([0, 0, 0], [1, 1])[2], expected_turn)


@pytest.mark.parametrize("model", [Bicycle, Unicycle, DiffSteer])
def test_float_and_integer_commands_have_same_limited_motion(
    model: type[Bicycle | Unicycle | DiffSteer],
) -> None:
    outputs = []
    for dtype in [int, float]:
        robot = model(dt=0.1, accel_max=1.0, speed_max=0.35)
        command = np.array([1, 1] if model is DiffSteer else [1, 0], dtype=dtype)
        original = command.copy()
        robot.control = lambda vehicle, t, x, command=command: command
        outputs.append(robot.run(T=0.6, animate=False))
        nt.assert_array_equal(command, original)
    nt.assert_allclose(outputs[0], outputs[1], atol=1e-14)
