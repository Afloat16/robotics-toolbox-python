"""Both covariance updates must use noise in observation coordinates."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import EKF, Bicycle, LandmarkMap, RangeBearingSensor
from roboticstoolbox.tools.types import ArrayLike, NDArray


class CalibratedSensor(RangeBearingSensor):
    """Range/bearing noise obtained from correlated raw sensor channels."""

    def Hw(self, x: ArrayLike, landmark: int | ArrayLike | None) -> NDArray:
        return np.array([[2.0, 0.4], [-0.1, 0.5]])


class CartesianNoiseSensor(RangeBearingSensor):
    """Range/bearing observations of landmarks perturbed in world coordinates."""

    def Hw(self, x: ArrayLike, landmark: int | ArrayLike | None) -> NDArray:
        if isinstance(landmark, (int, np.integer)):
            point = self.map[int(landmark)]
        else:
            point = np.asarray(landmark)
        delta = point - np.asarray(x)[:2]
        squared_range = delta @ delta
        distance = np.sqrt(squared_range)
        return np.array(
            [
                [delta[0] / distance, delta[1] / distance],
                [-delta[1] / squared_range, delta[0] / squared_range],
            ]
        )


def inverse_observation(pose: NDArray, observation: NDArray) -> NDArray:
    """Invert range/bearing geometry without calling the sensor implementation."""
    angle = pose[2] + observation[1]
    return pose[:2] + observation[0] * np.array([np.cos(angle), np.sin(angle)])


@pytest.mark.parametrize("slam", [False, True])
@pytest.mark.parametrize("joseph", [False, True])
def test_landmark_birth_propagates_cartesian_noise_and_cross_covariance(
    slam: bool, joseph: bool
) -> None:
    robot = Bicycle(x0=[0.1, -0.2, 0.35])
    robot.control = [0.0, 0.0]
    landmarks = LandmarkMap(np.array([[4.0, -3.0], [2.0, 5.0]]))
    sensor = CartesianNoiseSensor(robot, landmarks)
    noise = np.array([[0.05, 0.012], [0.012, 0.03]])
    factor = np.array([[0.3, 0.0, 0.0], [0.1, 0.2, 0.0], [0.03, -0.01, 0.1]])
    estimator = EKF(
        robot=(robot, np.zeros((2, 2)) if slam else None),
        sensor=(sensor, noise),
        P0=factor @ factor.T,
        x0=[1.0, 0.2, -0.4],
        animate=False,
        joseph=joseph,
    )
    seen = set()
    for _ in range(10):
        previous_state = estimator.x_est.copy()
        prior = estimator.P_est.copy()
        pose = previous_state[:3] if slam else robot.x.copy()
        estimator.step()
        record = estimator.history[-1]
        if record.lm in seen:
            continue
        seen.add(record.lm)
        assert record.z is not None
        point = inverse_observation(pose, record.z)
        epsilon = 1e-6
        Gx = np.column_stack(
            [
                (
                    inverse_observation(pose + epsilon * e, record.z)
                    - inverse_observation(pose - epsilon * e, record.z)
                )
                / (2 * epsilon)
                for e in np.eye(3)
            ]
        )
        # Inverting a range/bearing observation of p + w returns p + w:
        # raw Cartesian noise therefore contributes exactly W to map covariance.
        if slam:
            cross = prior[:, :3] @ Gx.T
            variance = Gx @ prior[:3, :3] @ Gx.T + noise
        else:
            cross = np.zeros((len(prior), 2))
            variance = noise
        expected_covariance = np.block([[prior, cross], [cross.T, variance]])
        nt.assert_allclose(estimator.x_est, np.r_[previous_state, point], atol=1e-12)
        nt.assert_allclose(estimator.P_est, expected_covariance, atol=1e-9)
        if len(seen) == 2:
            break
    assert len(seen) == 2


@pytest.mark.parametrize("mode", ["localisation", "mapping", "slam"])
def test_joseph_matches_conditional_covariance_for_transformed_noise(mode: str) -> None:
    outputs = []
    for joseph in [False, True]:
        robot = Bicycle()
        robot.control = [0.0, 0.0]
        landmarks = LandmarkMap(np.array([[4.0], [3.0]]))
        sensor = CalibratedSensor(robot, landmarks)
        noise = np.array([[0.04, 0.008], [0.008, 0.01]])
        estimator = EKF(
            robot=(robot, None if mode == "mapping" else np.zeros((2, 2))),
            sensor=(sensor, noise),
            map=landmarks if mode == "localisation" else None,
            P0=np.diag([0.3, 0.2, 0.05]),
            animate=False,
            joseph=joseph,
        )
        if mode != "localisation":
            estimator.step()  # first observation augments the map
        prior = estimator.P_est.copy()
        point = landmarks[0]
        if mode == "localisation":
            H = sensor.Hx(robot.x, point)
        elif mode == "mapping":
            H = sensor.Hp(robot.x, point)
        else:
            H = np.column_stack((sensor.Hx(robot.x, point), sensor.Hp(robot.x, point)))
        R = sensor.Hw(robot.x, point) @ noise @ sensor.Hw(robot.x, point).T
        expected = np.linalg.inv(np.linalg.inv(prior) + H.T @ np.linalg.solve(R, H))
        S = H @ prior @ H.T + R
        K = np.linalg.solve(S, H @ prior).T
        conditional = prior - K @ S @ K.T
        nt.assert_allclose(conditional, expected, atol=1e-12)
        estimator.step()
        nt.assert_allclose(estimator.P_est, conditional, atol=1e-12)
        outputs.append(estimator.P_est)
    nt.assert_allclose(outputs[0], outputs[1], atol=1e-12)
