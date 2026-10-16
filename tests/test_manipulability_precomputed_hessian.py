"""The manipulability gradient accepts the kinematic Hessian's actual layout."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import ET, ETS, Robot
from roboticstoolbox.tools.types import NDArray


def _model(n: int) -> Robot:
    axes = [ET.Rz, ET.Ry, ET.Rx, ET.Rz, ET.Ry, ET.Rx, ET.Rz]
    ets = ETS()
    for i in range(n):
        ets *= axes[i](jindex=i) * ET.tx(0.2 + 0.03 * i) * ET.tz(0.1 + 0.01 * i)
    return Robot(ets)


@pytest.mark.parametrize("n", [3, 6, 7])
def test_cached_manipulability_hessian_matches_derivative(n: int) -> None:
    robot = _model(n)
    q = np.linspace(0.2, 0.8, n)
    J = robot.jacob0(q)
    H = robot.hessian0(q)
    expected = np.zeros((n, 1))
    h = 1e-6
    for i in range(n):
        dq = np.zeros(n)
        dq[i] = h
        expected[i, 0] = (
            robot.manipulability(q + dq, axes="trans")
            - robot.manipulability(q - dq, axes="trans")
        ) / (2 * h)
    nt.assert_allclose(robot.jacobm(J=J, H=H, axes="trans"), expected, atol=2e-8)
    nt.assert_allclose(robot.jacobm(q=q, axes="trans"), expected, atol=2e-8)


def test_cached_subchain_manipulability_hessian() -> None:
    robot = _model(7)
    end = robot.links[2]
    q = np.linspace(0.2, 0.8, robot.n)
    J = robot.jacob0(q, end=end)
    H = robot.hessian0(q, end=end)
    expected = robot.jacobm(q=q, end=end, axes="trans")
    nt.assert_allclose(
        robot.jacobm(J=J, H=H, end=end, axes="trans"), expected, atol=1e-12
    )


@pytest.mark.parametrize("shape", [(6, 7, 7), (7, 7, 6), (7, 6, 6)])
def test_rejects_incompatible_cached_hessian(shape: tuple[int, ...]) -> None:
    robot = _model(7)
    q = np.linspace(0.2, 0.8, robot.n)
    H: NDArray = np.zeros(shape)
    with pytest.raises(ValueError):
        robot.jacobm(J=robot.jacob0(q), H=H)
