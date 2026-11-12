"""The convergence cost must use the same information matrices as the solve."""

from pathlib import Path

import pytest

from roboticstoolbox import PoseGraph


@pytest.mark.parametrize(
    "information,expected",
    [
        ("1 0 0 1 0 1", 5.09),
        ("4 1 0 9 0 16", 45.44),
        ("9 0 0 9 0 9", 45.81),
    ],
)
def test_posegraph_cost_uses_measurement_information(
    tmp_path: Path, information: str, expected: float
) -> None:
    """Cost is e.T Omega e, including covariance coupling between axes."""
    path = tmp_path / "weighted.g2o"
    path.write_text(
        f"VERTEX_SE2 0 0 0 0\nVERTEX_SE2 1 2 2 0.3\nEDGE_SE2 0 1 1 0 0 {information}\n"
    )
    graph = PoseGraph(str(path))

    # The pre-update residual is [1, 2, 0.3]. For the second matrix this is
    # 4 * 1**2 + 2 * 1 * 2 + 9 * 2**2 + 16 * 0.3**2 = 45.44.
    assert graph.linearize_and_solve() == pytest.approx(expected)


def test_posegraph_continues_while_weighted_error_decreases(tmp_path: Path) -> None:
    """An uncertain loop closure must not stop the well-weighted odometry fit."""
    path = tmp_path / "loop.g2o"
    path.write_text(
        "VERTEX_SE2 0 0 0 0\n"
        "VERTEX_SE2 1 1.23283 1.24392 2.13137\n"
        "VERTEX_SE2 2 2.7 0.49438 0.648741\n"
        "VERTEX_SE2 3 1.53307 1.01206 1.93234\n"
        "EDGE_SE2 0 1 1.06049 0.0918499 0.270294 100 0 0 100 0 200\n"
        "EDGE_SE2 1 2 1.28784 0.796069 0.730525 100 0 0 100 0 200\n"
        "EDGE_SE2 2 3 0.129531 1.40321 1.02514 100 0 0 100 0 200\n"
        "EDGE_SE2 3 0 -1.44459 1.74589 -2.14763 100 0 0 100 0 200\n"
        "EDGE_SE2 0 2 3.86914 1.63234 0.184713 1 0 0 1 0 2\n"
        "EDGE_SE2 1 3 0.94446 2.10938 0.404437 1 0 0 1 0 2\n"
    )
    graph = PoseGraph(str(path))
    graph.optimize(iterations=12)

    # At the first step the unweighted squared error rises from about 31.5 to
    # 36.3, while the information-weighted cost falls from 3013 to 1498.
    # Stopping on the former leaves cost 31.97; continuing the same normal
    # equations converges to about 10.53. No change to those equations is needed.
    cost = 0.0
    for edge in graph.graph.edges():
        error, _, _ = edge.linear_factors()
        cost += error @ edge.info @ error
    assert cost < 11.0
