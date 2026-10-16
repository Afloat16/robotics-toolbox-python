"""Joint-limit avoidance must use the same active coordinates as FK."""

import numpy as np
import numpy.testing as nt
import pytest

import roboticstoolbox as rtb
from roboticstoolbox.robot.IK import _null_Σ


def chain(indices: list[int]) -> rtb.ETS:
    return rtb.ETS([rtb.ET.tx(jindex=index, qlim=[-1, 1]) for index in indices])


@pytest.mark.parametrize("indices", [[0, 1], [2, 5], [5, 2]])
@pytest.mark.parametrize("influence", [0.4, np.array([0.4, 0.5])])
def test_limit_gradient_reads_active_joint_coordinates(indices, influence):
    ets = chain(indices)
    q = np.full(max(indices) + 1, 0.23)
    q[indices] = [0.9, 0.0]

    # Only the first joint lies within the influence region. Its upper-limit
    # avoidance gradient is -((1 - 0.9 - 0.4) / (0.4 - 0.1))**2 = -1.
    nt.assert_allclose(_null_Σ(ets, q, 0.1, influence), [[-1.0], [0.0]])


@pytest.mark.parametrize(
    "solver_type, options",
    [
        (rtb.IK_NR, {"pinv": True}),
        (rtb.IK_GN, {"pinv": True}),
        (rtb.IK_LM, {"method": "wampler", "k": 0.1}),
    ],
)
def test_sparse_step_redistributes_motion_away_from_limit(solver_type, options):
    indices = [2, 5]
    ets = chain(indices)
    q = np.full(6, 0.23)
    q[indices] = [0.9, 0.0]
    before = q.copy()
    target = ets.eval(np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.5]))
    solver = solver_type(kq=1.0, ps=0.1, pi=0.4, **options)

    residual, active = solver.step(ets, target, q)

    # For two coincident prismatic axes the null space redistributes equal
    # and opposite motion: projection of [-1, 0] is [-0.5, 0.5]. The primary
    # translation update is shared equally, with LM's configured damping.
    denominator = 2.1 if solver_type is rtb.IK_LM else 2.0
    primary = 0.1 / denominator
    nt.assert_allclose(active, [0.9 + primary - 0.5, primary + 0.5])
    nt.assert_array_equal(q[[0, 1, 3, 4]], before[[0, 1, 3, 4]])
    error = 1.0 - np.sum(active)
    nt.assert_allclose(residual, 0.5 * error**2, atol=1e-15)
