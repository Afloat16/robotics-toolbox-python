"""Manipulability must describe the same selected chain as its Jacobian."""

import numpy as np
import pytest
from spatialmath import SE3

import roboticstoolbox as rtb


@pytest.mark.parametrize(
    "start,end",
    [
        (None, None),
        (None, "panda_hand"),
        (None, "panda_link6"),
        ("panda_link2", None),
        ("panda_link2", "panda_link6"),
    ],
)
@pytest.mark.parametrize("method", ["yoshikawa", "invcondition", "minsingular"])
@pytest.mark.parametrize("axes", ["all", "trans", "rot"])
@pytest.mark.parametrize("translated_tool", [False, True])
def test_manipulability_uses_selected_chain(
    start: str | None, end: str | None, method: str, axes: str, translated_tool: bool
) -> None:
    panda = rtb.models.Panda()
    if translated_tool:
        panda.tool = SE3.Trans(0.4, -0.2, 0.3) * SE3.RPY(0.2, -0.4, 0.7)
    q = np.array([0.3, -0.5, 0.2, -1.8, -0.4, 1.4, 0.6])
    trajectory = np.stack([q, q + 0.1])
    expected = []
    for row in trajectory:
        jacobian = panda.jacob0(row, start=start, end=end)
        if axes == "trans":
            jacobian = jacobian[:3]
        elif axes == "rot":
            jacobian = jacobian[3:]
        if method == "yoshikawa":
            if jacobian.shape[0] == jacobian.shape[1]:
                value = abs(np.linalg.det(jacobian))
            else:
                value = np.sqrt(abs(np.linalg.det(jacobian @ jacobian.T)))
        else:
            singular = np.linalg.svd(jacobian, compute_uv=False)
            value = singular[-1]
            if method == "invcondition":
                value /= singular[0]
        expected.append(value)

    options = {"start": start, "end": end, "method": method, "axes": axes}
    np.testing.assert_allclose(
        panda.manipulability(q, **options), expected[0], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        panda.manipulability(trajectory, **options), expected, rtol=1e-10, atol=1e-12
    )


def test_manipulability_accepts_link_and_gripper_objects() -> None:
    panda = rtb.models.Panda()
    q = np.array([0.3, -0.5, 0.2, -1.8, -0.4, 1.4, 0.6])
    start = panda.links[2]
    end = panda.links[6]
    expected = np.linalg.svd(panda.jacob0(q, start=start, end=end), compute_uv=False)[
        -1
    ]
    assert panda.manipulability(
        q, start=start, end=end, method="minsingular"
    ) == pytest.approx(expected)
    for method in ("yoshikawa", "invcondition", "minsingular"):
        assert panda.manipulability(
            q, end=panda.grippers[0], method=method
        ) == pytest.approx(panda.manipulability(q, method=method))


def test_manipulability_gripper_asada_with_dynamic_parameters() -> None:
    panda = rtb.models.Panda()
    for link in panda.links:
        link.m = 1.0
        link.I = np.eye(3) * 0.1
    q = np.array([0.3, -0.5, 0.2, -1.8, -0.4, 1.4, 0.6])
    expected = panda.manipulability(q, method="asada")
    assert expected > 0
    assert panda.manipulability(
        q, end=panda.grippers[0], method="asada"
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "start,end",
    [
        (None, None),
        (None, "panda_link6"),
        (None, "panda_link5"),
        ("panda_link2", None),
    ],
)
@pytest.mark.parametrize("axes", ["all", "trans", "rot"])
@pytest.mark.parametrize("translated_tool", [False, True])
def test_asada_subchain_matches_inverse_dynamics(
    start: str | None, end: str | None, axes: str, translated_tool: bool
) -> None:
    panda = rtb.models.Panda()
    for link in panda.links:
        link.m = 1.0
        link.I = np.eye(3) * 0.1
    if translated_tool:
        panda.tool = SE3.Trans(0.4, -0.2, 0.3) * SE3.RPY(0.2, -0.4, 0.7)
    q = np.array([0.3, -0.5, 0.2, -1.8, -0.4, 1.4, 0.6])
    trajectory = np.stack([q, q + 0.1])
    expected = []
    for row in trajectory:
        jacobian = panda.jacob0(row, start=start, end=end)
        if np.linalg.matrix_rank(jacobian) < 6:
            value = 0.0
        else:
            # Embed unit task-space accelerations into the full robot, holding
            # unselected joints fixed, then obtain task inertia directly from RNE.
            accelerations = np.zeros((6, panda.n))
            accelerations[:, panda.ets(start, end).jindices] = np.linalg.pinv(
                jacobian
            ).T
            torques = panda.rne(
                np.tile(row, (6, 1)),
                np.zeros((6, panda.n)),
                accelerations,
                gravity=[0, 0, 0],
            )
            task_inertia = accelerations @ torques.T
            if axes == "trans":
                task_inertia = task_inertia[:3, :3]
            elif axes == "rot":
                task_inertia = task_inertia[3:, 3:]
            eigenvalues = np.linalg.eigvalsh((task_inertia + task_inertia.T) / 2)
            value = eigenvalues[0] / eigenvalues[-1]
        expected.append(value)
        assert panda.manipulability(
            row, J=jacobian, start=start, end=end, method="asada", axes=axes
        ) == pytest.approx(value)

    np.testing.assert_allclose(
        panda.manipulability(
            trajectory, start=start, end=end, method="asada", axes=axes
        ),
        expected,
        rtol=1e-10,
        atol=1e-12,
    )
