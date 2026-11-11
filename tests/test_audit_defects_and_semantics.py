"""
Regression tests for Code Quality & Robustness Audit defects:
1. Pairwise and Order encoding correctness (N=1, N=2, empty graph, isolated vertices, wrap-around, injectivity).
2. CP-SAT and CPLEX API call conventions and invalid inputs.
3. CP-SAT and CPLEX status preservation (OPTIMAL, FEASIBLE, INFEASIBLE, TIMEOUT, SIZE_LIMIT, LICENSE_ERROR, ERROR).
4. Independent solution validation (permutation, edge bounds, objective mismatch).
5. CPLEX Community Edition vs Full-license handling.
6. Main process status preservation.
7. Experiment pipeline canonical keys and encoding separation.
"""

import pytest
import os
import json
from unittest.mock import MagicMock, patch

from cbmp import solve, SolveResult, compute_cyclic_bandwidth
from graph_utils import validate_solution, validate_and_normalize_graph
from baselines.cpsat_cbmp import solve_cpsat
from baselines.cplex_cbmp import solve_cplex, is_cplex_available
from tests.brute_force import cyclic_bandwidth_bruteforce
from experiments.run_experiments import get_canonical_key


# ============================================================================
# 1. ENCODING CORRECTNESS: SMALL & SPECIAL GRAPHS
# ============================================================================

def test_encoding_empty_graph():
    """Empty graph on N=4 vertices: optimum must be 0 for both encodings."""
    res_ord, stats_ord = solve(N=4, edges=[], encoding="order", verbose=False)
    res_pw, stats_pw = solve(N=4, edges=[], encoding="pairwise", verbose=False)
    assert res_ord == 0 and stats_ord["status"] == "OPTIMAL"
    assert res_pw == 0 and stats_pw["status"] == "OPTIMAL"

def test_encoding_isolated_vertices():
    """Graph with 2 connected vertices and 2 isolated vertices (N=4, edge [(0, 1)])."""
    edges = [(0, 1)]
    expected = cyclic_bandwidth_bruteforce(4, edges)
    assert expected == 1

    res_ord, stats_ord = solve(N=4, edges=edges, encoding="order", verbose=False)
    res_pw, stats_pw = solve(N=4, edges=edges, encoding="pairwise", verbose=False)
    assert res_ord == 1 and stats_ord["status"] == "OPTIMAL"
    assert res_pw == 1 and stats_pw["status"] == "OPTIMAL"

def test_encoding_single_vertex_n1():
    """N=1 vertex, no edges: bandwidth must be 0."""
    res_ord, stats_ord = solve(N=1, edges=[], encoding="order", verbose=False)
    res_pw, stats_pw = solve(N=1, edges=[], encoding="pairwise", verbose=False)
    assert res_ord == 0 and stats_ord["status"] == "OPTIMAL"
    assert res_pw == 0 and stats_pw["status"] == "OPTIMAL"

def test_encoding_n2_single_edge():
    """N=2 vertices, edge (0, 1): bandwidth must be 1."""
    edges = [(0, 1)]
    res_ord, stats_ord = solve(N=2, edges=edges, encoding="order", verbose=False)
    res_pw, stats_pw = solve(N=2, edges=edges, encoding="pairwise", verbose=False)
    assert res_ord == 1 and stats_ord["status"] == "OPTIMAL"
    assert res_pw == 1 and stats_pw["status"] == "OPTIMAL"

def test_encoding_triangle_n3():
    """N=3 triangle K_3: cyclic bandwidth is 1."""
    edges = [(0, 1), (1, 2), (0, 2)]
    res_ord, stats_ord = solve(N=3, edges=edges, encoding="order", verbose=False)
    res_pw, stats_pw = solve(N=3, edges=edges, encoding="pairwise", verbose=False)
    assert res_ord == 1 and stats_ord["status"] == "OPTIMAL"
    assert res_pw == 1 and stats_pw["status"] == "OPTIMAL"

def test_encoding_cyclic_wrap_around_c5():
    """Cycle C_5 has optimal cyclic bandwidth = 1."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    res_ord, stats_ord = solve(N=5, edges=edges, encoding="order", verbose=False)
    res_pw, stats_pw = solve(N=5, edges=edges, encoding="pairwise", verbose=False)
    assert res_ord == 1 and stats_ord["status"] == "OPTIMAL"
    assert res_pw == 1 and stats_pw["status"] == "OPTIMAL"


# ============================================================================
# 2. INDEPENDENT SOLUTION VALIDATION
# ============================================================================

def test_validate_solution_valid():
    """Valid permutation matching reported objective."""
    N = 4
    edges = [(0, 1), (1, 2), (2, 3)]
    assignment = [0, 1, 2, 3]
    valid, bw, err = validate_solution(N, edges, assignment, reported_objective=1)
    assert valid is True
    assert bw == 1
    assert err is None

def test_validate_solution_invalid_length():
    """Assignment with wrong length."""
    valid, bw, err = validate_solution(4, [(0, 1)], [0, 1, 2])
    assert valid is False
    assert "length" in err.lower()

def test_validate_solution_non_permutation():
    """Assignment with duplicate labels."""
    valid, bw, err = validate_solution(4, [(0, 1)], [0, 1, 1, 3])
    assert valid is False
    assert "duplicate" in err.lower() or "injective" in err.lower()

def test_validate_solution_out_of_bounds_label():
    """Assignment with label >= N."""
    valid, bw, err = validate_solution(4, [(0, 1)], [0, 1, 2, 4])
    assert valid is False
    assert "domain" in err.lower() or "out of" in err.lower()

def test_validate_solution_objective_mismatch():
    """Reported objective does not match actual cyclic bandwidth."""
    valid, bw, err = validate_solution(4, [(0, 1), (1, 2)], [0, 1, 2, 3], reported_objective=3)
    assert valid is False
    assert bw == 1
    assert "does not match" in err.lower()


# ============================================================================
# 3. CP-SAT & CPLEX API CONVENTIONS & INVALID INPUTS
# ============================================================================

def test_cpsat_api_calling_forms():
    """Verify all supported call forms for solve_cpsat."""
    edges = [(0, 1), (1, 2)]
    r1, s1 = solve_cpsat(N=3, edges=edges, timeout=5.0, verbose=False)
    r2, s2 = solve_cpsat(3, edges, timeout=5.0, verbose=False)
    r3, s3 = solve_cpsat(3, edges=edges, timeout=5.0, verbose=False)
    assert r1 == r2 == r3 == 1
    assert s1["status"] == s2["status"] == s3["status"] == "OPTIMAL"

def test_cplex_api_calling_forms():
    """Verify all supported call forms for solve_cplex."""
    pytest.importorskip("cplex")
    edges = [(0, 1), (1, 2)]
    r1, s1 = solve_cplex(N=3, edges=edges, timeout=5.0, verbose=False)
    r2, s2 = solve_cplex(3, edges, timeout=5.0, verbose=False)
    r3, s3 = solve_cplex(3, edges=edges, timeout=5.0, verbose=False)
    assert r1 == r2 == r3 == 1
    assert s1["status"] == s2["status"] == s3["status"] == "OPTIMAL"

def test_cpsat_invalid_arguments():
    """Verify solve_cpsat rejects ambiguous or missing arguments."""
    with pytest.raises(ValueError):
        solve_cpsat()
    with pytest.raises(ValueError):
        solve_cpsat(dataset="30_bus", N=6)
    with pytest.raises(ValueError):
        solve_cpsat(N=4, edges=[(0, 5)])  # vertex index out of bounds
    with pytest.raises(ValueError):
        solve_cpsat(N=-1, edges=[])

def test_cplex_invalid_arguments():
    """Verify solve_cplex rejects ambiguous or missing arguments."""
    with pytest.raises(ValueError):
        solve_cplex()
    with pytest.raises(ValueError):
        solve_cplex(dataset="30_bus", N=6)
    with pytest.raises(ValueError):
        solve_cplex(N=4, edges=[(0, 5)])  # vertex index out of bounds
    with pytest.raises(ValueError):
        solve_cplex(N=-1, edges=[])


# ============================================================================
# 4. CP-SAT STATUS PRESERVATION & MOCKING
# ============================================================================

def test_cpsat_status_feasible_mocked(monkeypatch):
    """When CP-SAT stops with FEASIBLE, status must remain FEASIBLE (not TIMEOUT)."""
    from ortools.sat.python import cp_model

    def mock_solve(self, model):
        return cp_model.FEASIBLE

    def mock_val(self, var):
        # Return 1-indexed valid placement
        return 1

    def mock_obj(self):
        return 2.0

    def mock_bound(self):
        return 1.0

    monkeypatch.setattr(cp_model.CpSolver, "Solve", mock_solve)
    monkeypatch.setattr(cp_model.CpSolver, "ObjectiveValue", mock_obj)
    monkeypatch.setattr(cp_model.CpSolver, "BestObjectiveBound", mock_bound)

    # Mock Value to return distinct values for a 3-vertex path
    val_map = {}
    def mock_val_distinct(self, var):
        name = str(var)
        if name not in val_map:
            val_map[name] = len(val_map) + 1
        return val_map[name]

    monkeypatch.setattr(cp_model.CpSolver, "Value", mock_val_distinct)

    res, stats = solve_cpsat(N=3, edges=[(0, 1), (1, 2)], timeout=1.0, verbose=False)
    assert stats["status"] == "FEASIBLE"
    assert stats["best_ub"] == 1  # actual recomputed bandwidth
    assert stats["best_lb"] == 1
    assert stats["gap"] == 0.0

def test_cpsat_status_infeasible_mocked(monkeypatch):
    """When CP-SAT returns INFEASIBLE, status must be INFEASIBLE."""
    from ortools.sat.python import cp_model

    monkeypatch.setattr(cp_model.CpSolver, "Solve", lambda self, m: cp_model.INFEASIBLE)
    res, stats = solve_cpsat(N=3, edges=[(0, 1), (1, 2)], timeout=1.0, verbose=False)
    assert stats["status"] == "INFEASIBLE"
    assert stats["best_ub"] == -1

def test_cpsat_status_unknown_mocked(monkeypatch):
    """When CP-SAT returns UNKNOWN (timeout before solution), status must be TIMEOUT."""
    from ortools.sat.python import cp_model

    monkeypatch.setattr(cp_model.CpSolver, "Solve", lambda self, m: cp_model.UNKNOWN)
    res, stats = solve_cpsat(N=3, edges=[(0, 1), (1, 2)], timeout=1.0, verbose=False)
    assert stats["status"] == "TIMEOUT"
    assert stats["best_ub"] == -1

def test_cpsat_validation_failure_sets_error(monkeypatch):
    """When CP-SAT returns OPTIMAL but assignment is not a permutation, status is ERROR."""
    from ortools.sat.python import cp_model

    monkeypatch.setattr(cp_model.CpSolver, "Solve", lambda self, m: cp_model.OPTIMAL)
    monkeypatch.setattr(cp_model.CpSolver, "ObjectiveValue", lambda self: 1.0)
    # Return identical label 1 for all vertices (invalid permutation)
    monkeypatch.setattr(cp_model.CpSolver, "Value", lambda self, var: 1)

    res, stats = solve_cpsat(N=3, edges=[(0, 1), (1, 2)], timeout=1.0, verbose=False)
    assert stats["status"] == "ERROR"
    assert stats["best_ub"] == -1
    assert "validation_error" in stats


# ============================================================================
# 5. CPLEX STATUS, LIMITS & LICENSE HANDLING
# ============================================================================

def test_cplex_community_edition_docplex_limit_mocked(monkeypatch):
    """Verify DOcplexLimitsExceeded properly sets SIZE_LIMIT."""
    pytest.importorskip("cplex")
    from docplex.mp.model import Model
    import docplex.mp.utils as utils

    def mock_solve_ce(self, *args, **kwargs):
        raise utils.DOcplexLimitsExceeded(1050, 1100)

    monkeypatch.setattr(Model, "solve", mock_solve_ce)
    res, stats = solve_cplex(N=3, edges=[(0, 1)], timeout=1.0, verbose=False)
    assert stats["status"] == "SIZE_LIMIT"
    assert res == -1

def test_cplex_full_license_mocked(monkeypatch):
    """Verify full licensed CPLEX successfully solves models without size limit rejection."""
    pytest.importorskip("cplex")
    from docplex.mp.model import Model

    class FakeDetails:
        status = "integer optimal solution"
        status_code = 101
        best_bound = 1.0
        def has_hit_limit(self):
            return False

    class FakeSol:
        def get_value(self, var):
            vname = str(var)
            if "k" in vname:
                return 1.0
            # For linear expr f[u] = sum(i * x_{u}_{i})
            for u in range(3):
                if f"x_{u}_" in vname:
                    return float(u + 1)
            return 1.0

    def mock_solve_full(self, *args, **kwargs):
        self._solve_details = FakeDetails()
        return FakeSol()

    monkeypatch.setattr(Model, "solve", mock_solve_full)
    res, stats = solve_cplex(N=3, edges=[(0, 1), (1, 2)], timeout=1.0, verbose=False)
    assert stats["status"] == "OPTIMAL"
    assert res == 1

def test_cplex_license_error_mocked(monkeypatch):
    """Verify CPLEX license failure reports LICENSE_ERROR."""
    pytest.importorskip("cplex")
    from docplex.mp.model import Model

    def mock_solve_lic(self, *args, **kwargs):
        raise RuntimeError("CPLEX Error 1014: License not found or invalid.")

    monkeypatch.setattr(Model, "solve", mock_solve_lic)
    res, stats = solve_cplex(N=3, edges=[(0, 1)], timeout=1.0, verbose=False)
    assert stats["status"] == "LICENSE_ERROR"
    assert res == -1


# ============================================================================
# 6. MAIN PROCESS STATUS PRESERVATION
# ============================================================================

def test_main_process_preserves_optimal():
    from main import main
    stats = main(
        dataset="14_bus",
        sb_rotation=True,
        sb_reflection=False,
        use_heuristic_initial_ub=False,
        timeout=10.0,
        encoding="order",
    )
    assert stats["status"] == "OPTIMAL"
    assert stats["best_ub"] >= 1

def test_main_process_preserves_pairwise():
    from main import main
    stats = main(
        dataset="14_bus",
        sb_rotation=True,
        sb_reflection=False,
        use_heuristic_initial_ub=False,
        timeout=10.0,
        encoding="pairwise",
    )
    assert stats["status"] == "OPTIMAL"
    assert stats["best_ub"] >= 1


# ============================================================================
# 7. EXPERIMENTS & CANONICAL KEYS
# ============================================================================

def test_canonical_key_distinguishes_encodings():
    """Ensure Order and Pairwise configurations have strictly distinct canonical keys."""
    task_order = {
        "dataset": "14_bus",
        "method": "sat",
        "solver": "cadical195",
        "encoding": "order",
        "solving_mode": "incremental",
        "eo_strategy": "commander",
        "sb_rotation": True,
        "sb_reflection": True,
        "heuristic_ub": True,
        "relabel_seed": None,
    }
    task_pw = dict(task_order, encoding="pairwise")

    key_order = get_canonical_key(task_order)
    key_pw = get_canonical_key(task_pw)

    assert key_order != key_pw
    assert "pairwise" in key_pw[3]
    assert "pairwise" not in key_order[3]

def test_canonical_key_distinguishes_solving_modes():
    """Ensure incremental and iterative solving modes produce distinct canonical keys."""
    task_inc = {
        "dataset": "14_bus",
        "method": "sat",
        "solver": "cadical195",
        "encoding": "pairwise",
        "solving_mode": "incremental",
        "eo_strategy": "commander",
        "sb_rotation": True,
        "sb_reflection": True,
        "heuristic_ub": True,
        "relabel_seed": None,
    }
    task_iter = dict(task_inc, solving_mode="iterative")

    assert get_canonical_key(task_inc) != get_canonical_key(task_iter)
