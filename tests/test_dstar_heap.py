"""D* key decreases must preserve minimum-first frontier extraction."""

import heapq
from unittest.mock import patch

import numpy as np
import numpy.testing as nt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from roboticstoolbox import DstarPlanner
from roboticstoolbox.mobile.DstarPlanner import _Dstar, _Map


def test_decreased_keys_are_extracted_in_priority_order() -> None:
    planner = _Dstar(_Map(1, 20))
    states = planner._Map._Map[0]
    priorities = [
        11,
        20,
        28,
        26,
        17,
        21,
        22,
        27,
        25,
        12,
        13,
        14,
        15,
        18,
        10,
        19,
        24,
        23,
        16,
        29,
    ]
    for state, priority in zip(states, priorities):
        planner.insert(state, priority)
    for index, priority in [(7, 3), (19, 5), (9, 9), (10, 3), (0, 7)]:
        planner.insert(states[index], priority)
        priorities[index] = priority

    extracted = [heapq.heappop(planner.open_list) for _ in states]
    assert [key for key, _ in extracted] == sorted(priorities)
    assert {state.x for _, state in extracted} == set(range(len(states)))


def test_unchanged_key_does_not_duplicate_an_open_state() -> None:
    planner = _Dstar(_Map(1, 3))
    states = planner._Map._Map[0]
    for state, priority in zip(states, [3, 1, 2]):
        planner.insert(state, priority)
    planner.insert(states[1], 5)
    planner.insert(states[1], 1)
    assert len(planner.open_list) == 3
    assert [heapq.heappop(planner.open_list)[0] for _ in states] == [1, 2, 3]


def test_weighted_map_expands_the_current_minimum() -> None:
    cost = np.random.default_rng(0).integers(1, 40, (8, 8)).astype(float)
    planner = DstarPlanner(cost, goal=(7, 7))
    process = planner._Dstar.process__State

    def check_minimum(verbose: bool = False) -> float:
        frontier = planner._Dstar.open_list
        if frontier:
            assert frontier[0][0] == min(key for key, _ in frontier)
        return process(verbose=verbose)

    with patch.object(planner._Dstar, "process__State", side_effect=check_minimum):
        planner.plan()
    path, status = planner.query((0, 0))

    # Independent shortest-path oracle with the documented eight-neighbor
    # edge weight: average endpoint cost times Euclidean step distance.
    rows, columns, weights = [], [], []
    for y, x in np.ndindex(cost.shape):
        for dy, dx in np.ndindex((3, 3)):
            yy, xx = y + dy - 1, x + dx - 1
            if (yy, xx) == (y, x) or not (0 <= yy < 8 and 0 <= xx < 8):
                continue
            rows.append(8 * y + x)
            columns.append(8 * yy + xx)
            weights.append((cost[y, x] + cost[yy, xx]) * np.hypot(yy - y, xx - x) / 2)
    graph = csr_matrix((weights, (rows, columns)), shape=(64, 64))
    expected = dijkstra(graph, directed=False, indices=0)[63]
    nt.assert_allclose(status.cost, expected, atol=1e-12)
    nt.assert_allclose(path[0], [0, 0])
    nt.assert_allclose(path[-1], [7, 7])
