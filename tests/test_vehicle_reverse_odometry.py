"""Signed encoder travel must reconstruct noiseless reverse vehicle motion."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import EKF, Bicycle, DiffSteer, Unicycle


@pytest.mark.parametrize("model", [Bicycle, Unicycle, DiffSteer])
@pytest.mark.parametrize("speed", [-1.2, 0.0, 1.2])
@pytest.mark.parametrize("heading", [0.0, 0.7, -2.4])
def test_odometry_reconstructs_each_vehicle_step(
    model: type[Bicycle | Unicycle | DiffSteer], speed: float, heading: float
) -> None:
    robot = model(dt=0.1, x0=[2.0, -1.0, heading])
    controls = [speed, speed] if model is DiffSteer else [speed, 0.2]
    for _ in range(4):
        previous = robot.x.copy()
        odo = robot.step(controls, animate=False)
        nt.assert_allclose(odo[0], 0.1 * speed, atol=1e-15)
        nt.assert_allclose(robot.f(previous, odo), robot.x, atol=1e-14)


@pytest.mark.parametrize("model", [Bicycle, Unicycle, DiffSteer])
def test_dead_reckoning_follows_reverse_motion(
    model: type[Bicycle | Unicycle | DiffSteer],
) -> None:
    start = np.array([1.0, -2.0, 0.6])
    robot = model(dt=0.1, x0=start)
    robot.control = [-1.0, -1.0] if model is DiffSteer else [-1.0, 0.2]
    estimator = EKF(
        robot=(robot, np.zeros((2, 2))),
        P0=np.zeros((3, 3)),
        x0=start,
        animate=False,
    )
    estimator.run(1.0)
    nt.assert_allclose(estimator.get_xyt(), robot.x_hist, atol=1e-14)
    nt.assert_allclose(estimator.P_est, 0.0)
