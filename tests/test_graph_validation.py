"""
Unit tests for input validation, edge normalization, and dataset resolution.
Phase 2 & Phase 5 of the Code Quality & Robustness Audit.
"""

import tempfile
from pathlib import Path
import pytest
from scipy.io import mmwrite
from scipy.sparse import coo_matrix

from graph_utils import (
    resolve_dataset_path,
    validate_and_normalize_graph,
    load_graph_from_mtx,
)
from cbmp import solve
from baselines.cpsat_cbmp import solve_cpsat
from baselines.cplex_cbmp import solve_cplex


# ============================================================================
# 1. GRAPH VALIDATION & NORMALIZATION TESTS
# ============================================================================

def test_valid_graphs_normalization():
    # Empty graph
    N, edges = validate_and_normalize_graph(5, [])
    assert N == 5
    assert edges == []

    # Single vertex
    N, edges = validate_and_normalize_graph(1, [])
    assert N == 1
    assert edges == []

    # Path graph
    N, edges = validate_and_normalize_graph(4, [(2, 1), (0, 1), (3, 2)])
    assert N == 4
    assert edges == [(0, 1), (1, 2), (2, 3)]

    # Complete graph
    raw_edges = [(1, 0), (2, 0), (3, 0), (2, 1), (3, 1), (3, 2)]
    N, edges = validate_and_normalize_graph(4, raw_edges)
    assert N == 4
    assert len(edges) == 6
    assert edges == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    # Disconnected graph
    N, edges = validate_and_normalize_graph(4, [(3, 2), (0, 1)])
    assert N == 4
    assert edges == [(0, 1), (2, 3)]


def test_edge_normalization_duplicates_and_self_loops():
    # Duplicate and reversed edges
    raw_edges = [(1, 0), (0, 1), (1, 2), (2, 1), (0, 1)]
    N, edges = validate_and_normalize_graph(3, raw_edges)
    assert edges == [(0, 1), (1, 2)]

    # Self-loops ignored
    raw_edges_with_loops = [(0, 0), (0, 1), (1, 1), (2, 2)]
    N, edges = validate_and_normalize_graph(3, raw_edges_with_loops)
    assert edges == [(0, 1)]


def test_invalid_n():
    # N <= 0
    with pytest.raises(ValueError, match="positive integer"):
        validate_and_normalize_graph(0, [])
    with pytest.raises(ValueError, match="positive integer"):
        validate_and_normalize_graph(-3, [])

    # Non-integer N
    with pytest.raises(TypeError, match="integer"):
        validate_and_normalize_graph(4.5, [])
    with pytest.raises(TypeError, match="integer"):
        validate_and_normalize_graph("4", [])

    # Booleans are not allowed as integers
    with pytest.raises(TypeError, match="integer"):
        validate_and_normalize_graph(True, [])
    with pytest.raises(TypeError, match="integer"):
        validate_and_normalize_graph(False, [])


def test_invalid_edges():
    # Not iterable or non-tuple/list element
    with pytest.raises(TypeError):
        validate_and_normalize_graph(3, "not-edges")

    with pytest.raises(ValueError, match="pair"):
        validate_and_normalize_graph(3, [(0,)])

    with pytest.raises(ValueError, match="pair"):
        validate_and_normalize_graph(3, [(0, 1, 2)])

    with pytest.raises(TypeError, match="must be integers"):
        validate_and_normalize_graph(3, [(0, "1")])

    with pytest.raises(TypeError, match="must be integers"):
        validate_and_normalize_graph(3, [(True, 1)])

    # Vertex indices out of bounds
    with pytest.raises(ValueError, match="out of bounds"):
        validate_and_normalize_graph(4, [(-1, 0)])

    with pytest.raises(ValueError, match="out of bounds"):
        validate_and_normalize_graph(4, [(0, 4)])

    with pytest.raises(ValueError, match="out of bounds"):
        validate_and_normalize_graph(4, [(10, 2)])


# ============================================================================
# 2. DATASET PATH RESOLUTION & MTX LOADING TESTS
# ============================================================================

def test_resolve_dataset_path():
    # Base name
    p1, name1 = resolve_dataset_path("14_bus")
    assert Path(p1).is_file()
    assert Path(p1).name == "14_bus.mtx"
    assert name1 == "14_bus"

    # With extension
    p2, name2 = resolve_dataset_path("14_bus.mtx")
    assert p2 == p1
    assert name2 == "14_bus"

    # Relative path
    p3, name3 = resolve_dataset_path("datasets/14_bus.mtx")
    assert p3 == p1
    assert name3 == "14_bus"

    # Path object
    p4, name4 = resolve_dataset_path(Path("datasets/14_bus.mtx"))
    assert p4 == p1
    assert name4 == "14_bus"

    # Nonexistent
    with pytest.raises(FileNotFoundError):
        resolve_dataset_path("non_existent_dataset_xyz")


def test_load_graph_from_mtx_valid():
    N, edges, dataset_name = load_graph_from_mtx("14_bus")
    assert dataset_name == "14_bus"
    assert N == 14
    assert len(edges) > 0
    # Edges should be sorted and normalized
    for u, v in edges:
        assert 0 <= u < v < N


def test_load_graph_from_mtx_non_square():
    with tempfile.NamedTemporaryFile(suffix=".mtx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        # 3x4 non-square matrix
        row = [0, 1]
        col = [1, 2]
        data = [1.0, 1.0]
        mat = coo_matrix((data, (row, col)), shape=(3, 4))
        mmwrite(str(tmp_path), mat)

        with pytest.raises(ValueError, match="square"):
            load_graph_from_mtx(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_load_graph_from_mtx_asymmetric():
    with tempfile.NamedTemporaryFile(suffix=".mtx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        # 3x3 asymmetric matrix: edge (0, 1) has value 1, but (1, 0) has value 0
        row = [0]
        col = [1]
        data = [1.0]
        mat = coo_matrix((data, (row, col)), shape=(3, 3))
        mmwrite(str(tmp_path), mat)

        with pytest.raises(ValueError, match="symmetric"):
            load_graph_from_mtx(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ============================================================================
# 3. CROSS-SOLVER CONSISTENCY ON EDGE CASES
# ============================================================================

def test_cross_solver_empty_graph():
    N = 4
    edges = []
    r_sat, s_sat = solve(N=N, edges=edges, verbose=False)
    r_cp, s_cp = solve_cpsat(N=N, edges=edges, verbose=False)
    r_cplex, s_cplex = solve_cplex(N=N, edges=edges, verbose=False)

    assert r_sat == 0 and s_sat["status"] == "OPTIMAL"
    assert r_cp == 0 and s_cp["status"] == "OPTIMAL"
    assert r_cplex == 0 and s_cplex["status"] == "OPTIMAL"


def test_cross_solver_single_vertex():
    N = 1
    edges = []
    r_sat, s_sat = solve(N=N, edges=edges, verbose=False)
    r_cp, s_cp = solve_cpsat(N=N, edges=edges, verbose=False)
    r_cplex, s_cplex = solve_cplex(N=N, edges=edges, verbose=False)

    assert r_sat == 0 and s_sat["status"] == "OPTIMAL"
    assert r_cp == 0 and s_cp["status"] == "OPTIMAL"
    assert r_cplex == 0 and s_cplex["status"] == "OPTIMAL"


def test_cross_solver_disconnected_graph():
    N = 4
    edges = [(0, 1), (2, 3)]
    r_sat, s_sat = solve(N=N, edges=edges, verbose=False)
    r_cp, s_cp = solve_cpsat(N=N, edges=edges, verbose=False)
    r_cplex, s_cplex = solve_cplex(N=N, edges=edges, verbose=False)

    assert r_sat == r_cp == r_cplex == 1
    assert s_sat["status"] == s_cp["status"] == s_cplex["status"] == "OPTIMAL"
