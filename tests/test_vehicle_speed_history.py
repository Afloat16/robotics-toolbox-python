"""Acceleration limits remember the applied speed after saturation."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import Bicycle, DiffSteer, Unicycle
from roboticstoolbox.mobile.Vehicle import VehicleBase


@pytest.mark.parametrize("vehicle_type", [Bicycle, Unicycle, DiffSteer])
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_derivative_brakes_after_speed_saturation(
    vehicle_type: type[VehicleBase], direction: int, reverse: bool
) -> None:
    vehicle = vehicle_type(speed_max=1, accel_max=0.5, dt=0.1)
    command = np.array([10.0, 10.0 if vehicle_type is DiffSteer else 0]) * direction
    for _ in range(25):
        vehicle.deriv([0.0, 0, 0], command)

    braking = -command if reverse else np.zeros(2)
    actual = [vehicle.deriv([0.0, 0, 0], braking)[0] for _ in range(25)]
    expected = 1 - 0.05 * np.arange(1, 26)
    if not reverse:
        expected = np.maximum(expected, 0)
    nt.assert_allclose(actual, direction * expected, atol=1e-12)


@pytest.mark.parametrize("vehicle_type", [Bicycle, Unicycle, DiffSteer])
@pytest.mark.parametrize("direction", [-1, 1])
def test_step_retains_its_saturated_braking_behavior(
    vehicle_type: type[VehicleBase], direction: int
) -> None:
    vehicle = vehicle_type(speed_max=1, accel_max=0.5, dt=0.1)
    command = np.array([10.0, 10.0 if vehicle_type is DiffSteer else 0]) * direction
    for _ in range(25):
        vehicle.step(command, animate=False)

    before = vehicle.x.copy()
    vehicle.step(np.zeros(2), animate=False)
    nt.assert_allclose(vehicle.x - before, [direction * 0.095, 0, 0], atol=1e-12)


def test_differential_wheels_retain_independent_speed_history() -> None:
    vehicle = DiffSteer(speed_max=1, accel_max=0.5, dt=0.1, W=1)
    for _ in range(25):
        vehicle.deriv([0.0, 0, 0], [10.0, 0.3])
    derivative = vehicle.deriv([0.0, 0, 0], [0.0, 0.3])
    # Left wheel slows from 1 to .95 while the right wheel stays at .3.
    nt.assert_allclose(derivative, [0.625, 0, -0.65], atol=1e-12)
