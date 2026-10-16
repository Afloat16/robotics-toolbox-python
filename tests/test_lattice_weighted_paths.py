"""Lattice edge weights must determine path cost independently of SE(2) units."""

import itertools

import numpy as np
import pytest
from spatialmath import SE2

from roboticstoolbox import LatticePlanner


@pytest.mark.parametrize("scale", [0.01, 0.05, 0.1])
@pytest.mark.parametrize("side", [-1, 1])
def test_small_move_costs_keep_the_four_move_optimum(scale, side):
    planner = LatticePlanner(costs=(scale, scale, scale))
    planner.plan(iterations=6)
    path, status = planner.query((0, 0, 0), (1, side * 3, np.pi))
    # S,L,S,L (or S,R,S,R) reaches this pose in four moves. Independently
    # enumerate the geometric action words to rule out any shorter route.
    moves = (SE2(1, 0, 0), SE2(1, 1, np.pi / 2), SE2(1, -1, -np.pi / 2))
    target = SE2(1, side * 3, np.pi)
    for count in range(4):
        for word in itertools.product(moves, repeat=count):
            pose = SE2()
            for move in word:
                pose *= move
            assert not np.allclose(pose.A, target.A)
    assert status.cost == pytest.approx(4 * scale)
    assert len(status.segments) == 4
    assert status.cost == pytest.approx(sum(edge.cost for edge in status.edges))
    np.testing.assert_allclose(path[-1, :2], (1, side * 3), atol=1e-12)


@pytest.mark.parametrize("scale", [0.02, 0.1, 3.0])
def test_uniformly_rescaling_move_costs_preserves_optimum(scale):
    base = LatticePlanner(costs=(1, 1, 1))
    scaled = LatticePlanner(costs=(scale, scale, scale))
    base.plan(iterations=5)
    scaled.plan(iterations=5)
    start = (0, 0, 0)
    for name in ("0SLSL", "0SRSR", "0SSLR", "0LLS", "0RRS"):
        goal = base.graph[name].coord
        _, original = base.query(start, goal)
        _, result = scaled.query(start, goal)
        assert result.cost == pytest.approx(scale * original.cost)
