"""Both sides of a joint's active velocity damper constrain the QP."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import SE3

import roboticstoolbox as rtb
from tests import skip_no_qp


@skip_no_qp
@pytest.mark.parametrize("influence", [0.1, 0.3, 1.0])
@pytest.mark.parametrize("gain", [0.5, 2.0])
@pytest.mark.parametrize("direction", [-1, 1])
def test_qp_preserves_both_overlapping_limit_inequalities(influence, gain, direction):
    ets = rtb.ETS(rtb.ET.tx(qlim=[-0.05, 0.05]))
    initial = direction * 0.04
    q = np.array([initial])
    displacement = direction * 0.5
    solver = rtb.IK_QP(kq=gain, pi=influence, ps=0.005, kj=0.01, ks=1.0)

    _, result = solver.step(ets, SE3.Tx(initial + displacement).A, q)

    # The one-dimensional QP eliminates slack as delta = e - dq.
    # Its unconstrained minimiser is ks*e / (ks + kj*abs(e)); both
    # independently active damper inequalities clip it to this interval.
    lower = (-0.05 - initial + solver.ps) / (influence - solver.ps) / gain
    upper = (0.05 - initial - solver.ps) / (influence - solver.ps) / gain
    unconstrained = displacement / (1 + solver.kj * abs(displacement))
    expected_step = np.clip(unconstrained, lower, upper)
    nt.assert_allclose(result, [initial + expected_step], atol=1e-12)
    assert lower - 1e-12 <= result[0] - initial <= upper + 1e-12


@skip_no_qp
def test_two_sided_dampers_keep_heterogeneous_joints_independent():
    ets = rtb.ETS(
        [
            rtb.ET.tx(qlim=[-0.05, 0.05]),
            rtb.ET.ty(qlim=[-2.0, 2.0]),
        ]
    )
    q = np.array([0.04, -1.9])
    solver = rtb.IK_QP(kq=1.0, pi=[1.0, 0.3], ps=0.0)

    _, result = solver.step(ets, SE3(0.54, -2.4, 0).A, q)

    # First joint has both limits active, second only its lower limit.
    nt.assert_allclose(result, [0.05, -1.9 - 1 / 3], atol=1e-12)
