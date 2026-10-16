"""A terminal joint keeps its own axis when a tool frame is attached."""

import importlib

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import SE3
from spatialmath.base import vex

from roboticstoolbox import ET, ETS
from roboticstoolbox.tools.types import NDArray

fknm = importlib.import_module("roboticstoolbox.ets.fknm")


def _differentiate(ets: ETS, q: NDArray, tool: SE3) -> NDArray:
    h = 1e-6
    T = ets.fkine(q, tool=tool).A
    J = np.zeros((6, ets.n))
    for i in range(ets.n):
        dq = np.zeros(ets.n)
        dq[i] = h
        dT = (ets.fkine(q + dq, tool=tool).A - ets.fkine(q - dq, tool=tool).A) / (2 * h)
        J[:3, i] = dT[:3, 3]
        J[3:, i] = vex(dT[:3, :3] @ T[:3, :3].T)
    return J


@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.parametrize("kind", ["Rx", "Ry", "Rz", "tx", "ty", "tz"])
@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("array_tool", [False, True])
def test_terminal_joint_with_rotated_tool(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    kind: str,
    flip: bool,
    array_tool: bool,
) -> None:
    if backend == "native" and not fknm._C_AVAILABLE:
        pytest.skip("native kinematics extension is not available")
    if backend == "python":
        monkeypatch.setattr(fknm, "_C_AVAILABLE", False)

    ets = ET.Rx(0.3) * ET.ty(0.2) * getattr(ET, kind)(flip=flip)
    q = np.array([0.4])
    tool = SE3.Trans(0.2, -0.1, 0.3) * SE3.RPY(0.4, -0.5, 0.6)
    argument = tool.A if array_tool else tool
    expected = _differentiate(ets, q, tool)
    nt.assert_allclose(ets.jacob0(q, tool=argument), expected, atol=2e-9)
    R = ets.fkine(q, tool=tool).R
    change_frame = np.zeros((6, 6))
    change_frame[:3, :3] = change_frame[3:, 3:] = R.T
    nt.assert_allclose(ets.jacobe(q, tool=argument), change_frame @ expected, atol=2e-9)


def test_terminal_joint_matches_equivalent_constant_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fknm, "_C_AVAILABLE", False)
    ets = ET.Rx() * ET.ty(0.2) * ET.Rz()
    tool = SE3.Trans(0.2, 0.1, -0.3) * SE3.RPY(0.5, -0.4, 0.2)
    folded = ets * ET.SE3(tool)
    q = [0.3, -0.2]
    nt.assert_allclose(ets.jacob0(q, tool=tool), folded.jacob0(q), atol=1e-12)
    nt.assert_allclose(ets.jacobe(q, tool=tool), folded.jacobe(q), atol=1e-12)
    nt.assert_allclose(ets.hessian0(q, tool=tool), folded.hessian0(q), atol=1e-12)
