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
