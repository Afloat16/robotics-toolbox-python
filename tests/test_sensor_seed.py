"""The sensor seed must survive the resets used by repeated simulations."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import EKF, Bicycle, LandmarkMap, RangeBearingSensor


def make_sensor(seed: int) -> RangeBearingSensor:
    landmarks = LandmarkMap(np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]))
    return RangeBearingSensor(Bicycle(), landmarks, covar=[0.02, 0.01], seed=seed)


def read_sequence(sensor: RangeBearingSensor) -> np.ndarray:
    readings = []
    for _ in range(12):
        measurement, landmark = sensor.reading()
        assert measurement is not None
        readings.append([landmark, *measurement])
    return np.asarray(readings)


@pytest.mark.parametrize("seed", [0, 7, 23])
def test_reset_replays_requested_sensor_seed(seed: int) -> None:
    sensor = make_sensor(seed)
    first = read_sequence(sensor)

    sensor.init()
    nt.assert_array_equal(read_sequence(sensor), first)
    sensor.init()
    nt.assert_array_equal(read_sequence(sensor), first)


def test_different_sensor_seeds_remain_distinct_after_reset() -> None:
    a, b = make_sensor(7), make_sensor(23)
    a.init()
    b.init()

    assert not np.array_equal(read_sequence(a), read_sequence(b))


def run_localisation(seed: int) -> np.ndarray:
    sensor = make_sensor(seed)
    robot = sensor.robot
    robot.control = (0.5, 0.05)
    covariance = np.diag([0.01, 0.005]) ** 2
    ekf = EKF(
        robot=(robot, covariance),
        sensor=(sensor, sensor.W),
        map=sensor.map,
        P0=np.diag([0.02, 0.02, 0.01]),
        x_est=[0.1, 0.1, 0.0],
    )
    ekf.run(T=1, animate=False)
    return ekf.get_xyt()


def test_ekf_runs_use_the_requested_sensor_seed() -> None:
    first = run_localisation(7)
    nt.assert_array_equal(run_localisation(7), first)
    assert not np.array_equal(run_localisation(23), first)
