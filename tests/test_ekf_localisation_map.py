"""Localisation uses the known map, independently of the sensor's true world."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import EKF, Bicycle, LandmarkMap, RangeBearingSensor
from roboticstoolbox.tools.types import NDArray


def observation(pose: NDArray, point: NDArray) -> NDArray:
    delta = point - pose[:2]
    angle = np.arctan2(delta[1], delta[0]) - pose[2]
    return np.array([np.linalg.norm(delta), np.arctan2(np.sin(angle), np.cos(angle))])


@pytest.mark.parametrize("joseph", [False, True])
@pytest.mark.parametrize("map_offset", [[0.0, 0.0], [0.6, -0.8], [-1.0, 0.4]])
def test_localisation_update_uses_estimator_map(
    joseph: bool, map_offset: list[float]
) -> None:
    true_point = np.array([6.0, 2.0])
    known_point = true_point + map_offset
    robot = Bicycle()
    robot.control = [0.0, 0.0]
    sensor = RangeBearingSensor(robot, LandmarkMap(true_point[:, None]))
    pose = np.array([0.3, -0.2, 0.1])
    prior = np.diag([0.4, 0.3, 0.08])
    noise = np.array([[0.04, 0.002], [0.002, 0.01]])
    estimator = EKF(
        robot=(robot, np.zeros((2, 2))),
        sensor=(sensor, noise),
        map=LandmarkMap(known_point[:, None]),
        P0=prior,
        x0=pose,
        joseph=joseph,
        animate=False,
    )
    epsilon = 1e-6
    H = np.column_stack(
        [
            (
                observation(pose + epsilon * e, known_point)
                - observation(pose - epsilon * e, known_point)
            )
            / (2 * epsilon)
            for e in np.eye(3)
        ]
    )
    innovation = observation(np.zeros(3), true_point) - observation(pose, known_point)
    S = H @ prior @ H.T + noise
    K = np.linalg.solve(S, H @ prior).T
    expected_state = pose + K @ innovation
    expected_covariance = prior - K @ S @ K.T

    estimator.step()

    nt.assert_allclose(estimator.history[-1].innov, innovation, atol=1e-12)
    nt.assert_allclose(estimator.x_est, expected_state, atol=1e-9)
    nt.assert_allclose(estimator.P_est, expected_covariance, atol=1e-9)
