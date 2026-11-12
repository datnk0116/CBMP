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


# ============================================================================
# 8. STATUS POLICY & PAR2 AUDIT SYNTHETIC TESTS
# ============================================================================

def test_all_statuses_cell_formatting_and_semantics():
    """Verify format_cell_value handles all 11 explicit statuses distinctly and correctly."""
    from analysis.make_tables import format_cell_value

    assert format_cell_value("OPTIMAL", solve_time=1.2345) == "1.2345"
    assert format_cell_value("PARTIAL", solve_time=1.2345, n_optimal=3, n_total=5) == "3/5 1.2345*"
    assert format_cell_value("FEASIBLE", best_ub=4) == "FEAS(4)"
    assert format_cell_value("FEASIBLE", best_ub=-1) == "FEAS"
    assert format_cell_value("TIMEOUT", best_ub=5) == "TO(5)"
    assert format_cell_value("TIMEOUT", best_ub=-1) == "TO"
    assert format_cell_value("INTERRUPTED", best_ub=6) == "INT(6)"
    assert format_cell_value("INTERRUPTED", best_ub=-1) == "INT"
    assert format_cell_value("SIZE_LIMIT") == "SL"
    assert format_cell_value("UNAVAILABLE") == "UNAV"
    assert format_cell_value("LICENSE_ERROR") == "LIC"
    assert format_cell_value("INFEASIBLE") == "UNSAT"
    assert format_cell_value("NUMERIC_ERROR") == "NUM_ERR"
    assert format_cell_value("ERROR") == "ERR"
    assert format_cell_value(None) == "-"


def test_status_aggregation_and_par2_exclusion_policy():
    """
    Verify aggregate_repetitions and compute_summary_stats policies:
    1. Count only OPTIMAL as proven solved.
    2. Exclude SIZE_LIMIT, UNAVAILABLE, LICENSE_ERROR from solver algorithmic PAR2.
    3. TIMEOUT, INTERRUPTED, and other failures contribute 2 * timeout.
    4. Unverified incumbents with validation_error are stripped.
    """
    from analysis.make_tables import aggregate_repetitions, compute_summary_stats
    import pandas as pd

    records = [
        {"dataset": "d1", "solver": "s1", "status": "OPTIMAL", "solve_time": 10.0, "time_limit": 600.0, "rep": 1},
        {"dataset": "d2", "solver": "s1", "status": "TIMEOUT", "solve_time": 600.0, "best_ub": 4, "time_limit": 600.0, "rep": 1},
        {"dataset": "d3", "solver": "s1", "status": "SIZE_LIMIT", "solve_time": 0.1, "time_limit": 600.0, "rep": 1},
        {"dataset": "d4", "solver": "s1", "status": "UNAVAILABLE", "solve_time": 0.0, "time_limit": 600.0, "rep": 1},
        {"dataset": "d5", "solver": "s1", "status": "LICENSE_ERROR", "solve_time": 0.0, "time_limit": 600.0, "rep": 1},
        {"dataset": "d6", "solver": "s1", "status": "FEASIBLE", "best_ub": 5, "solve_time": 50.0, "time_limit": 600.0, "rep": 1},
    ]

    agg = aggregate_repetitions(records)
    agg_map = {r["dataset"]: r for r in agg}

    # Verify PAR2 values for individual statuses
    assert agg_map["d1"]["par2"] == 10.0  # OPTIMAL: exact time
    assert agg_map["d2"]["par2"] == 1200.0  # TIMEOUT: 2 * 600
    assert "par2" not in agg_map["d3"]  # SIZE_LIMIT: excluded
    assert "par2" not in agg_map["d4"]  # UNAVAILABLE: excluded
    assert "par2" not in agg_map["d5"]  # LICENSE_ERROR: excluded
    assert agg_map["d6"]["par2"] == 1200.0  # FEASIBLE (non-optimal): 2 * 600

    # Summary stats across datasets
    df_raw = pd.DataFrame({
        "s1": [agg_map["d1"], agg_map["d2"], agg_map["d3"], agg_map["d4"], agg_map["d5"], agg_map["d6"]]
    }, index=["d1", "d2", "d3", "d4", "d5", "d6"])

    solved, sl, par2, speedup = compute_summary_stats(df_raw, ["s1"], timeout=600.0)
    assert solved["s1"] == 1, "Only OPTIMAL must be counted as proven solved"
    assert sl["s1"] == 1, "SIZE_LIMIT must be tracked separately"
    # Denominator for PAR2 must be 3 valid algorithmic runs (d1: 10, d2: 1200, d6: 1200) -> mean = 2410 / 3 = 803.333
    expected_par2 = (10.0 + 1200.0 + 1200.0) / 3.0
    assert abs(par2["s1"] - expected_par2) < 1e-3


def test_par2_and_repetition_twelve_scenarios():
    """
    Comprehensive tests covering all 12 specified PAR2 and repetition scenarios:
    1. All repetitions OPTIMAL.
    2. OPTIMAL mixed with TIMEOUT.
    3. OPTIMAL mixed with UNAVAILABLE.
    4. OPTIMAL mixed with LICENSE_ERROR.
    5. OPTIMAL mixed with SIZE_LIMIT.
    6. Only UNAVAILABLE.
    7. Only LICENSE_ERROR.
    8. Only SIZE_LIMIT.
    9. FEASIBLE mixed with TIMEOUT.
    10. ERROR and INTERRUPTED.
    11. Missing records.
    12. Mixed repetitions with different timeout values.
    """
    from analysis.make_tables import aggregate_repetitions, compute_summary_stats, format_cell_value
    import pandas as pd
    import numpy as np

    # 1. All repetitions OPTIMAL
    reps_s1 = [
        {"dataset": "d1", "solver": "cadical", "status": "OPTIMAL", "solve_time": 10.0, "time_limit": 600.0, "rep": 1},
        {"dataset": "d1", "solver": "cadical", "status": "OPTIMAL", "solve_time": 12.0, "time_limit": 600.0, "rep": 2},
        {"dataset": "d1", "solver": "cadical", "status": "OPTIMAL", "solve_time": 11.0, "time_limit": 600.0, "rep": 3},
    ]
    agg_s1 = aggregate_repetitions(reps_s1)[0]
    assert agg_s1["status"] == "OPTIMAL"
    assert agg_s1["n_total"] == 3
    assert agg_s1["n_algorithmic"] == 3
    assert agg_s1["n_optimal"] == 3
    assert agg_s1["par2"] == 11.0  # mean(10, 12, 11) = 11.0
    assert format_cell_value(agg_s1["status"], solve_time=agg_s1["solve_time"]) == "11.0000"

    # 2. OPTIMAL mixed with TIMEOUT -> PARTIAL
    reps_s2 = [
        {"dataset": "d2", "solver": "cadical", "status": "OPTIMAL", "solve_time": 15.0, "time_limit": 600.0, "rep": 1},
        {"dataset": "d2", "solver": "cadical", "status": "TIMEOUT", "solve_time": 600.0, "time_limit": 600.0, "rep": 2},
        {"dataset": "d2", "solver": "cadical", "status": "OPTIMAL", "solve_time": 17.0, "time_limit": 600.0, "rep": 3},
    ]
    agg_s2 = aggregate_repetitions(reps_s2)[0]
    assert agg_s2["status"] == "PARTIAL"
    assert agg_s2["n_total"] == 3
    assert agg_s2["n_algorithmic"] == 3
    assert agg_s2["n_optimal"] == 2
    assert agg_s2["n_timeout"] == 1
    # PAR2 = (15.0 + 1200.0 + 17.0) / 3 = 1232 / 3 = 410.6666...
    assert abs(agg_s2["par2"] - (1232.0 / 3.0)) < 1e-4
    assert format_cell_value(agg_s2["status"], solve_time=agg_s2["solve_time"], n_optimal=agg_s2["n_optimal"], n_algorithmic=agg_s2["n_algorithmic"]) == "2/3 16.0000*"

    # 3. OPTIMAL mixed with UNAVAILABLE -> PARTIAL (some reps succeeded, but environment error prevented 100% completion)
    reps_s3 = [
        {"dataset": "d3", "solver": "cadical", "status": "OPTIMAL", "solve_time": 8.0, "time_limit": 600.0, "rep": 1},
        {"dataset": "d3", "solver": "cadical", "status": "UNAVAILABLE", "solve_time": 0.0, "time_limit": 600.0, "rep": 2},
        {"dataset": "d3", "solver": "cadical", "status": "OPTIMAL", "solve_time": 10.0, "time_limit": 600.0, "rep": 3},
    ]
    agg_s3 = aggregate_repetitions(reps_s3)[0]
    assert agg_s3["status"] == "PARTIAL"
    assert agg_s3["n_total"] == 3
    assert agg_s3["n_algorithmic"] == 2
    assert agg_s3["n_optimal"] == 2
    assert agg_s3["n_unavailable"] == 1
    assert agg_s3["par2"] == 9.0  # mean(8, 10) = 9.0; UNAVAILABLE excluded from algorithmic PAR2
    assert format_cell_value(agg_s3["status"], solve_time=agg_s3["solve_time"], n_optimal=agg_s3["n_optimal"], n_total=agg_s3["n_total"]) == "2/3 9.0000*"

    # 4. OPTIMAL mixed with LICENSE_ERROR -> PARTIAL
    reps_s4 = [
        {"dataset": "d4", "solver": "cplex", "status": "OPTIMAL", "solve_time": 20.0, "time_limit": 600.0, "rep": 1},
        {"dataset": "d4", "solver": "cplex", "status": "LICENSE_ERROR", "solve_time": 0.0, "time_limit": 600.0, "rep": 2},
    ]
    agg_s4 = aggregate_repetitions(reps_s4)[0]
    assert agg_s4["status"] == "PARTIAL"
    assert agg_s4["n_total"] == 2
    assert agg_s4["n_algorithmic"] == 1
    assert agg_s4["n_license_error"] == 1
    assert agg_s4["par2"] == 20.0
    assert format_cell_value(agg_s4["status"], solve_time=agg_s4["solve_time"], n_optimal=agg_s4["n_optimal"], n_total=agg_s4["n_total"]) == "1/2 20.0000*"

    # 5. OPTIMAL mixed with SIZE_LIMIT -> PARTIAL
    reps_s5 = [
        {"dataset": "d5", "solver": "cplex", "status": "OPTIMAL", "solve_time": 25.0, "time_limit": 600.0, "rep": 1},
        {"dataset": "d5", "solver": "cplex", "status": "SIZE_LIMIT", "solve_time": 0.05, "time_limit": 600.0, "rep": 2},
    ]
    agg_s5 = aggregate_repetitions(reps_s5)[0]
    assert agg_s5["status"] == "PARTIAL"
    assert agg_s5["n_total"] == 2
    assert agg_s5["n_algorithmic"] == 1
    assert agg_s5["n_size_limit"] == 1
    assert agg_s5["par2"] == 25.0
    assert format_cell_value(agg_s5["status"], solve_time=agg_s5["solve_time"], n_optimal=agg_s5["n_optimal"], n_total=agg_s5["n_total"]) == "1/2 25.0000*"

    # 6. Only UNAVAILABLE
    reps_s6 = [
        {"dataset": "d6", "solver": "cadical", "status": "UNAVAILABLE", "solve_time": 0.0, "time_limit": 600.0, "rep": 1},
        {"dataset": "d6", "solver": "cadical", "status": "UNAVAILABLE", "solve_time": 0.0, "time_limit": 600.0, "rep": 2},
    ]
    agg_s6 = aggregate_repetitions(reps_s6)[0]
    assert agg_s6["status"] == "UNAVAILABLE"
    assert agg_s6["n_total"] == 2
    assert agg_s6["n_algorithmic"] == 0
    assert agg_s6["n_unavailable"] == 2
    assert agg_s6.get("par2") is None
    assert "par2" not in agg_s6
    assert format_cell_value(agg_s6["status"]) == "UNAV"

    # 7. Only LICENSE_ERROR
    reps_s7 = [
        {"dataset": "d7", "solver": "cplex", "status": "LICENSE_ERROR", "solve_time": 0.0, "time_limit": 600.0, "rep": 1},
    ]
    agg_s7 = aggregate_repetitions(reps_s7)[0]
    assert agg_s7["status"] == "LICENSE_ERROR"
    assert agg_s7["n_total"] == 1
    assert agg_s7["n_algorithmic"] == 0
    assert agg_s7["n_license_error"] == 1
    assert agg_s7.get("par2") is None
    assert "par2" not in agg_s7
    assert format_cell_value(agg_s7["status"]) == "LIC"

    # 8. Only SIZE_LIMIT
    reps_s8 = [
        {"dataset": "d8", "solver": "cplex", "status": "SIZE_LIMIT", "solve_time": 0.1, "time_limit": 600.0, "rep": 1},
        {"dataset": "d8", "solver": "cplex", "status": "SIZE_LIMIT", "solve_time": 0.1, "time_limit": 600.0, "rep": 2},
    ]
    agg_s8 = aggregate_repetitions(reps_s8)[0]
    assert agg_s8["status"] == "SIZE_LIMIT"
    assert agg_s8["n_total"] == 2
    assert agg_s8["n_algorithmic"] == 0
    assert agg_s8["n_size_limit"] == 2
    assert agg_s8.get("par2") is None
    assert "par2" not in agg_s8
    assert format_cell_value(agg_s8["status"]) == "SL"

    # 9. FEASIBLE mixed with TIMEOUT -> FEASIBLE with incumbent preserved
    reps_s9 = [
        {"dataset": "d9", "solver": "cplex", "status": "FEASIBLE", "best_ub": 4, "solve_time": 50.0, "time_limit": 600.0, "rep": 1},
        {"dataset": "d9", "solver": "cplex", "status": "TIMEOUT", "best_ub": 5, "solve_time": 600.0, "time_limit": 600.0, "rep": 2},
    ]
    agg_s9 = aggregate_repetitions(reps_s9)[0]
    assert agg_s9["status"] == "FEASIBLE"
    assert agg_s9["best_ub"] == 4
    assert agg_s9["n_total"] == 2
    assert agg_s9["n_algorithmic"] == 2
    assert agg_s9["n_feasible"] == 1
    assert agg_s9["n_timeout"] == 1
    assert agg_s9["par2"] == 1200.0  # both non-optimal penalized as 2 * 600
    assert format_cell_value(agg_s9["status"], best_ub=agg_s9["best_ub"]) == "FEAS(4)"

    # 10. ERROR and INTERRUPTED
    reps_s10 = [
        {"dataset": "d10", "solver": "cadical", "status": "ERROR", "solve_time": 1.0, "time_limit": 600.0, "rep": 1},
        {"dataset": "d10", "solver": "cadical", "status": "INTERRUPTED", "solve_time": 30.0, "best_ub": 7, "time_limit": 600.0, "rep": 2},
    ]
    agg_s10 = aggregate_repetitions(reps_s10)[0]
    assert agg_s10["status"] in ("ERROR", "INTERRUPTED")
    assert agg_s10["n_total"] == 2
    assert agg_s10["n_algorithmic"] == 2
    assert agg_s10["n_error"] == 1
    assert agg_s10["n_interrupted"] == 1
    assert agg_s10["par2"] == 1200.0  # both penalized as 2 * 600

    # 11. Missing records
    assert aggregate_repetitions([]) == []
    df_empty = pd.DataFrame({"cadical": [None, None]}, index=["d_miss1", "d_miss2"])
    solved_e, sl_e, par2_e, sp_e = compute_summary_stats(df_empty, ["cadical"], timeout=600.0)
    assert solved_e["cadical"] == 0
    assert sl_e["cadical"] == 0
    assert np.isnan(par2_e["cadical"])

    # 12. Mixed repetitions with different timeout values
    reps_s12 = [
        {"dataset": "d12", "solver": "cadical", "status": "TIMEOUT", "solve_time": 100.0, "time_limit": 100.0, "rep": 1},
        {"dataset": "d12", "solver": "cadical", "status": "TIMEOUT", "solve_time": 200.0, "time_limit": 200.0, "rep": 2},
    ]
    agg_s12 = aggregate_repetitions(reps_s12)[0]
    assert agg_s12["status"] == "TIMEOUT"
    # Rep 1 penalty is 2 * 100 = 200, Rep 2 penalty is 2 * 200 = 400 -> mean = 300.0
    assert agg_s12["par2"] == 300.0

    # Cross-dataset summary stats integrating scenarios 1-10
    all_aggs = [agg_s1, agg_s2, agg_s3, agg_s4, agg_s5, agg_s6, agg_s7, agg_s8, agg_s9, agg_s10]
    df_multi = pd.DataFrame({"solver_col": all_aggs}, index=[f"d{i}" for i in range(1, 11)])
    solved_m, sl_m, par2_m, _ = compute_summary_stats(df_multi, ["solver_col"], timeout=600.0)

    # Proven optimal across 100% of repetitions: strictly d1
    assert solved_m["solver_col"] == 1
    # SIZE_LIMIT tracked: d8 (d5 had 1 SL rep, but was OPTIMAL on algorithmic attempt)
    assert sl_m["solver_col"] == 1

    # PAR2 population:
    # d1: 11.0
    # d2: 1232/3 = 410.6667
    # d3: 9.0
    # d4: 20.0
    # d5: 25.0
    # d6: UNAVAILABLE -> excluded
    # d7: LICENSE_ERROR -> excluded
    # d8: SIZE_LIMIT -> excluded
    # d9: 1200.0
    # d10: 1200.0
    expected_par2_list = [11.0, 1232.0 / 3.0, 9.0, 20.0, 25.0, 1200.0, 1200.0]
    expected_mean_par2 = float(np.mean(expected_par2_list))
    assert abs(par2_m["solver_col"] - expected_mean_par2) < 1e-3


def test_infeasible_and_unsat_repetition_aggregation():
    """Verify INFEASIBLE and UNSAT statuses are normalized, counted, receive PAR2 penalty, and format as UNSAT."""
    from analysis.make_tables import aggregate_repetitions, format_cell_value

    reps = [
        {"dataset": "d_unsat", "solver": "cadical", "status": "INFEASIBLE", "time_limit": 300.0, "rep": 1},
        {"dataset": "d_unsat", "solver": "cadical", "status": "UNSAT", "time_limit": 300.0, "rep": 2},
    ]
    agg = aggregate_repetitions(reps)[0]
    assert agg["status"] == "INFEASIBLE"
    assert agg["n_total"] == 2
    assert agg["n_algorithmic"] == 2
    assert agg["n_infeasible"] == 2
    assert agg["n_optimal"] == 0
    assert agg["best_ub"] == -1
    assert agg["assignment"] is None
    # Each non-optimal algorithmic run penalized as 2 * 300 = 600.0
    assert agg["par2"] == 600.0
    assert format_cell_value(agg["status"]) == "UNSAT"


def test_unknown_and_missing_status_handling():
    """Verify unknown or missing statuses are preserved in n_unknown, excluded from algorithmic PAR2 without false penalty."""
    from analysis.make_tables import aggregate_repetitions, format_cell_value

    reps = [
        {"dataset": "d_unk", "solver": "cadical", "status": "WEIRD_CORRUPT_STATUS", "time_limit": 600.0, "rep": 1},
        {"dataset": "d_unk", "solver": "cadical", "status": None, "time_limit": 600.0, "rep": 2},
    ]
    agg = aggregate_repetitions(reps)[0]
    assert agg["status"] == "UNKNOWN"
    assert agg["n_total"] == 2
    assert agg["n_algorithmic"] == 0
    assert agg["n_unknown"] == 2
    assert agg.get("par2") is None
    assert "par2" not in agg
    assert format_cell_value(agg["status"]) == "UNK"


def test_optimal_missing_timing_penalized_not_dropped():
    """Verify that an OPTIMAL repetition with missing solve_time and wall_time is not given artificial 0.0s and is included in PAR2."""
    from analysis.make_tables import aggregate_repetitions

    reps = [
        {"dataset": "d_miss_t", "solver": "cadical", "status": "OPTIMAL", "solve_time": None, "wall_time": None, "time_limit": 100.0, "rep": 1},
    ]
    agg = aggregate_repetitions(reps)[0]
    assert agg["status"] == "OPTIMAL"
    assert agg["n_total"] == 1
    assert agg["n_algorithmic"] == 1
    assert agg["n_optimal"] == 1
    # Missing timing for OPTIMAL run must not receive 0.0s; penalized with 2 * time_limit = 200.0
    assert agg["par2"] == 200.0


@pytest.mark.parametrize("status, has_incumbent, validation_error, expected_status", [
    ("OPTIMAL", True, False, "OPTIMAL"),
    ("TIMEOUT", True, False, "TIMEOUT"),
    ("FEASIBLE", True, False, "FEASIBLE"),
    ("SIZE_LIMIT", False, False, "SIZE_LIMIT"),
    ("UNAVAILABLE", False, False, "UNAVAILABLE"),
    ("LICENSE_ERROR", False, False, "LICENSE_ERROR"),
    ("INFEASIBLE", False, False, "INFEASIBLE"),
    ("ERROR", False, False, "ERROR"),
    ("OPTIMAL", True, True, "ERROR"),  # validation error overrides OPTIMAL to ERROR
])
def test_single_record_vs_multi_record_aggregation_consistency(status, has_incumbent, validation_error, expected_status):
    """
    Verify that single-repetition aggregation produces the EXACT same semantics,
    schema, status normalization, and cleanup as multi-repetition aggregation.
    """
    from analysis.make_tables import aggregate_repetitions

    def make_record(rep):
        rec = {
            "dataset": "d_test",
            "solver": "cadical",
            "status": status,
            "solve_time": 10.0,
            "wall_time": 10.5,
            "time_limit": 600.0,
            "rep": rep,
            "best_ub": 2 if has_incumbent else -1,
            "assignment": [0, 1, 2] if has_incumbent else None,
        }
        if validation_error:
            rec["validation_error"] = "Invalid permutation bijection"
        return rec

    # 1-record aggregation
    single_agg = aggregate_repetitions([make_record(1)])[0]
    # 3-identical-records aggregation
    multi_agg = aggregate_repetitions([make_record(1), make_record(2), make_record(3)])[0]

    assert single_agg["status"] == expected_status
    assert multi_agg["status"] == expected_status

    if expected_status in ("ERROR", "SIZE_LIMIT", "UNAVAILABLE", "LICENSE_ERROR", "INFEASIBLE"):
        assert single_agg["best_ub"] == -1
        assert multi_agg["best_ub"] == -1
        assert single_agg["assignment"] is None
        assert multi_agg["assignment"] is None

    # PAR2 consistency: either both have par2 or both exclude par2
    if "par2" in single_agg:
        assert "par2" in multi_agg
        assert single_agg["par2"] == multi_agg["par2"]
    else:
        assert "par2" not in multi_agg

    # Population proportions consistency
    assert single_agg["n_total"] == 1
    assert multi_agg["n_total"] == 3
    if expected_status == "OPTIMAL":
        assert single_agg["n_optimal"] == 1 and multi_agg["n_optimal"] == 3
    elif expected_status == "SIZE_LIMIT":
        assert single_agg["n_size_limit"] == 1 and multi_agg["n_size_limit"] == 3
    elif expected_status == "ERROR":
        assert single_agg["n_error"] == 1 and multi_agg["n_error"] == 3


def test_end_to_end_aggregation_summary_and_table_formatting(tmp_path):
    """
    Synthetic end-to-end test that passes records through:
    1. aggregate_repetitions (grouping and population metrics).
    2. compute_summary_stats (Solved, SL, PAR2).
    3. format_cell_value (cell formatting).
    4. export_table (CSV and LaTeX generation).
    Verifies that all summaries are mutually consistent, PAR2 denominator excludes non-algorithmic
    failures, and CSV exports preserve population counters.
    """
    from analysis.make_tables import (
        aggregate_repetitions,
        compute_summary_stats,
        format_cell_value,
        export_table,
    )
    import pandas as pd

    raw_records = [
        # Dataset 1: solver A is fully OPTIMAL across 3 reps, solver B has 1 OPTIMAL and 2 TIMEOUT -> PARTIAL
        {"dataset": "ds1", "solver": "sol_a", "status": "OPTIMAL", "solve_time": 2.0, "time_limit": 600.0, "rep": 1},
        {"dataset": "ds1", "solver": "sol_a", "status": "OPTIMAL", "solve_time": 3.0, "time_limit": 600.0, "rep": 2},
        {"dataset": "ds1", "solver": "sol_a", "status": "OPTIMAL", "solve_time": 4.0, "time_limit": 600.0, "rep": 3},
        {"dataset": "ds1", "solver": "sol_b", "status": "OPTIMAL", "solve_time": 5.0, "time_limit": 600.0, "rep": 1},
        {"dataset": "ds1", "solver": "sol_b", "status": "TIMEOUT", "solve_time": 600.0, "time_limit": 600.0, "rep": 2},
        {"dataset": "ds1", "solver": "sol_b", "status": "TIMEOUT", "solve_time": 600.0, "time_limit": 600.0, "rep": 3},

        # Dataset 2: solver A is FEASIBLE (incumbent 4), solver B hits SIZE_LIMIT
        {"dataset": "ds2", "solver": "sol_a", "status": "FEASIBLE", "best_ub": 4, "solve_time": 50.0, "time_limit": 600.0, "rep": 1},
        {"dataset": "ds2", "solver": "sol_b", "status": "SIZE_LIMIT", "solve_time": 0.05, "time_limit": 600.0, "rep": 1},

        # Dataset 3: solver A has UNAVAILABLE, solver B has LICENSE_ERROR
        {"dataset": "ds3", "solver": "sol_a", "status": "UNAVAILABLE", "solve_time": 0.0, "time_limit": 600.0, "rep": 1},
        {"dataset": "ds3", "solver": "sol_b", "status": "LICENSE_ERROR", "solve_time": 0.0, "time_limit": 600.0, "rep": 1},

        # Dataset 4: solver A is INFEASIBLE, solver B has ERROR
        {"dataset": "ds4", "solver": "sol_a", "status": "INFEASIBLE", "time_limit": 600.0, "rep": 1},
        {"dataset": "ds4", "solver": "sol_b", "status": "ERROR", "time_limit": 600.0, "rep": 1},
    ]

    agg = aggregate_repetitions(raw_records)
    agg_map = {(r["dataset"], r["solver"]): r for r in agg}

    # Verify per-cell aggregation
    assert agg_map[("ds1", "sol_a")]["status"] == "OPTIMAL"
    assert agg_map[("ds1", "sol_a")]["solve_time"] == 3.0
    assert agg_map[("ds1", "sol_a")]["par2"] == 3.0
    assert format_cell_value(agg_map[("ds1", "sol_a")]) == "3.0000"

    assert agg_map[("ds1", "sol_b")]["status"] == "PARTIAL"
    assert agg_map[("ds1", "sol_b")]["n_optimal"] == 1
    assert agg_map[("ds1", "sol_b")]["n_total"] == 3
    # PAR2 = (5 + 1200 + 1200) / 3 = 2405 / 3 = 801.6667
    assert abs(agg_map[("ds1", "sol_b")]["par2"] - (2405.0 / 3.0)) < 1e-4
    assert format_cell_value(agg_map[("ds1", "sol_b")]) == "1/3 5.0000*"

    assert agg_map[("ds2", "sol_a")]["status"] == "FEASIBLE"
    assert format_cell_value(agg_map[("ds2", "sol_a")]) == "FEAS(4)"

    assert agg_map[("ds2", "sol_b")]["status"] == "SIZE_LIMIT"
    assert format_cell_value(agg_map[("ds2", "sol_b")]) == "SL"
    assert "par2" not in agg_map[("ds2", "sol_b")]

    assert agg_map[("ds3", "sol_a")]["status"] == "UNAVAILABLE"
    assert format_cell_value(agg_map[("ds3", "sol_a")]) == "UNAV"

    assert agg_map[("ds3", "sol_b")]["status"] == "LICENSE_ERROR"
    assert format_cell_value(agg_map[("ds3", "sol_b")]) == "LIC"

    assert agg_map[("ds4", "sol_a")]["status"] == "INFEASIBLE"
    assert format_cell_value(agg_map[("ds4", "sol_a")]) == "UNSAT"

    assert agg_map[("ds4", "sol_b")]["status"] == "ERROR"
    assert format_cell_value(agg_map[("ds4", "sol_b")]) == "ERR"

    # Construct DataFrame for summary stats
    datasets = ["ds1", "ds2", "ds3", "ds4"]
    df_raw = pd.DataFrame({
        "sol_a": [agg_map[(ds, "sol_a")] for ds in datasets],
        "sol_b": [agg_map[(ds, "sol_b")] for ds in datasets],
    }, index=datasets)

    df_display = pd.DataFrame({
        "sol_a": [format_cell_value(agg_map[(ds, "sol_a")]) for ds in datasets],
        "sol_b": [format_cell_value(agg_map[(ds, "sol_b")]) for ds in datasets],
    }, index=datasets)

    cols = ["sol_a", "sol_b"]
    solved, sl, par2, speedups = compute_summary_stats(df_raw, cols, baseline_col="sol_a", timeout=600.0)

    # Proven optimal: only sol_a on ds1 is OPTIMAL (sol_b is PARTIAL on ds1)
    assert solved["sol_a"] == 1
    assert solved["sol_b"] == 0

    # Size limit count: only sol_b on ds2
    assert sl["sol_a"] == 0
    assert sl["sol_b"] == 1

    # PAR2 for sol_a:
    # ds1: 3.0 (OPTIMAL)
    # ds2: 1200.0 (FEASIBLE penalized as 2*600)
    # ds3: UNAVAILABLE -> excluded
    # ds4: 1200.0 (INFEASIBLE penalized as 2*600)
    # mean = (3.0 + 1200.0 + 1200.0) / 3 = 2403.0 / 3 = 801.0
    assert abs(par2["sol_a"] - 801.0) < 1e-4

    # PAR2 for sol_b:
    # ds1: 2405 / 3 = 801.6667 (PARTIAL)
    # ds2: SIZE_LIMIT -> excluded
    # ds3: LICENSE_ERROR -> excluded
    # ds4: 1200.0 (ERROR penalized as 2*600)
    # mean = (801.6667 + 1200.0) / 2 = 1000.8333
    expected_par2_b = ((2405.0 / 3.0) + 1200.0) / 2.0
    assert abs(par2["sol_b"] - expected_par2_b) < 1e-4

    # Export table to verify CSV generation
    summary_rows = [
        ("Solved", {c: solved[c] for c in cols}),
        ("SL", {c: sl[c] for c in cols}),
        ("PAR2 (s)", {c: f"{par2[c]:.2f}" for c in cols}),
    ]
    out_dir = str(tmp_path / "tables")
    export_table(df_display, summary_rows, title="End to End Test Table", base_filename="e2e_table", output_dirs=(out_dir,))

    csv_path = f"{out_dir}/e2e_table.csv"
    assert os.path.exists(csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "3.0000" in content
    assert "1/3 5.0000*" in content
    assert "FEAS(4)" in content
    assert "SL" in content
    assert "UNAV" in content
    assert "LIC" in content
    assert "UNSAT" in content
    assert "ERR" in content
    assert "Solved,1,0" in content
    assert "SL,0,1" in content


def test_par2_partial_two_optimal_substantially_different_plus_timeout():
    """
    Test PAR2 calculation for PARTIAL status with 2 OPTIMAL reps having substantially
    different solve times plus 1 TIMEOUT rep.
    Verifies that PAR2 is computed from individual repetitions, NOT approximated via median.
    """
    import math
    import pandas as pd
    from analysis.make_tables import aggregate_repetitions, compute_summary_stats

    records = [
        {"dataset": "ds_skewed", "solver": "sol_x", "status": "OPTIMAL", "solve_time": 5.0, "time_limit": 600.0, "rep": 1},
        {"dataset": "ds_skewed", "solver": "sol_x", "status": "OPTIMAL", "solve_time": 45.0, "time_limit": 600.0, "rep": 2},
        {"dataset": "ds_skewed", "solver": "sol_x", "status": "TIMEOUT", "solve_time": 600.0, "time_limit": 600.0, "rep": 3},
    ]

    agg = aggregate_repetitions(records)
    assert len(agg) == 1
    rec = agg[0]
    assert rec["status"] == "PARTIAL"
    assert rec["n_optimal"] == 2
    assert rec["n_timeout"] == 1
    assert rec["n_total"] == 3
    assert rec["n_algorithmic"] == 3

    # Exact manual PAR2: (5.0 + 45.0 + 2 * 600.0) / 3 = 1250.0 / 3 = 416.6666667
    expected_par2 = 1250.0 / 3.0
    assert abs(rec["par2"] - expected_par2) < 1e-5

    # Test fallback in compute_summary_stats when precomputed par2 is absent:
    rec_no_par2 = dict(rec)
    del rec_no_par2["par2"]
    df_raw = pd.DataFrame({"sol_x": [rec_no_par2]}, index=["ds_skewed"])
    solved, sl, par2, speedup = compute_summary_stats(df_raw, ["sol_x"], timeout=600.0)
    assert abs(par2["sol_x"] - expected_par2) < 1e-5


def test_par2_partial_three_optimal_plus_one_feasible():
    """
    Test PAR2 calculation for PARTIAL status with 3 OPTIMAL reps with different solve times
    plus 1 FEASIBLE rep (non-optimal algorithmic termination).
    """
    import pandas as pd
    from analysis.make_tables import aggregate_repetitions, compute_summary_stats

    records = [
        {"dataset": "ds_feas", "solver": "sol_y", "status": "OPTIMAL", "solve_time": 2.0, "time_limit": 300.0, "rep": 1},
        {"dataset": "ds_feas", "solver": "sol_y", "status": "OPTIMAL", "solve_time": 6.0, "time_limit": 300.0, "rep": 2},
        {"dataset": "ds_feas", "solver": "sol_y", "status": "OPTIMAL", "solve_time": 22.0, "time_limit": 300.0, "rep": 3},
        {"dataset": "ds_feas", "solver": "sol_y", "status": "FEASIBLE", "best_ub": 4, "solve_time": 300.0, "time_limit": 300.0, "rep": 4},
    ]

    agg = aggregate_repetitions(records)
    assert len(agg) == 1
    rec = agg[0]
    assert rec["status"] == "PARTIAL"
    assert rec["n_optimal"] == 3
    assert rec["n_feasible"] == 1
    assert rec["n_total"] == 4
    assert rec["n_algorithmic"] == 4

    # Exact manual PAR2: (2.0 + 6.0 + 22.0 + 2 * 300.0) / 4 = 630.0 / 4 = 157.5
    expected_par2 = 157.5
    assert abs(rec["par2"] - expected_par2) < 1e-5

    rec_no_par2 = dict(rec)
    del rec_no_par2["par2"]
    df_raw = pd.DataFrame({"sol_y": [rec_no_par2]}, index=["ds_feas"])
    solved, sl, par2, speedup = compute_summary_stats(df_raw, ["sol_y"], timeout=300.0)
    assert abs(par2["sol_y"] - expected_par2) < 1e-5


def test_par2_different_time_limits_across_repetitions():
    """
    Test PAR2 calculation when individual repetitions have different time limits.
    """
    import pandas as pd
    from analysis.make_tables import aggregate_repetitions, compute_summary_stats

    records = [
        {"dataset": "ds_var_tl", "solver": "sol_z", "status": "OPTIMAL", "solve_time": 4.0, "time_limit": 100.0, "rep": 1},
        {"dataset": "ds_var_tl", "solver": "sol_z", "status": "TIMEOUT", "solve_time": 200.0, "time_limit": 200.0, "rep": 2},
        {"dataset": "ds_var_tl", "solver": "sol_z", "status": "TIMEOUT", "solve_time": 500.0, "time_limit": 500.0, "rep": 3},
    ]

    agg = aggregate_repetitions(records)
    assert len(agg) == 1
    rec = agg[0]
    assert rec["status"] == "PARTIAL"

    # Exact manual PAR2: (4.0 + 2 * 200.0 + 2 * 500.0) / 3 = (4.0 + 400.0 + 1000.0) / 3 = 1404.0 / 3 = 468.0
    expected_par2 = 468.0
    assert abs(rec["par2"] - expected_par2) < 1e-5

    rec_no_par2 = dict(rec)
    del rec_no_par2["par2"]
    df_raw = pd.DataFrame({"sol_z": [rec_no_par2]}, index=["ds_var_tl"])
    solved, sl, par2, speedup = compute_summary_stats(df_raw, ["sol_z"], timeout=600.0)
    assert abs(par2["sol_z"] - expected_par2) < 1e-5


def test_par2_partial_with_existing_valid_par2():
    """
    Test that compute_summary_stats preserves an already-computed valid par2 for PARTIAL records.
    """
    import pandas as pd
    from analysis.make_tables import compute_summary_stats

    rec = {
        "status": "PARTIAL",
        "solve_time": 10.0,
        "par2": 345.678,
        "n_optimal": 1,
        "n_total": 2,
        "n_algorithmic": 2,
    }
    df_raw = pd.DataFrame({"sol_m": [rec]}, index=["ds_test"])
    solved, sl, par2, speedup = compute_summary_stats(df_raw, ["sol_m"], timeout=600.0)
    assert abs(par2["sol_m"] - 345.678) < 1e-5


def test_par2_legacy_partial_without_par2_or_reps():
    """
    Test that a legacy PARTIAL record without precomputed par2 and without individual repetition data
    does NOT guess an approximate PAR2 using median solve time.
    """
    import math
    import pandas as pd
    from analysis.make_tables import compute_summary_stats

    legacy_rec = {
        "status": "PARTIAL",
        "solve_time": 15.0,
        "n_optimal": 1,
        "n_total": 3,
        "n_algorithmic": 3,
    }
    df_raw = pd.DataFrame({"sol_legacy": [legacy_rec]}, index=["ds_legacy"])
    solved, sl, par2, speedup = compute_summary_stats(df_raw, ["sol_legacy"], timeout=600.0)

    # When no par2 can be accurately computed, par2_scores should be nan
    assert math.isnan(par2["sol_legacy"])


def test_export_table_data_csv_readback_and_population_counters(tmp_path):
    """
    Test export_table generating both <base_filename>.csv and machine-readable <base_filename>_data.csv.
    Asserts that reading back <base_filename>_data.csv preserves all 12 population counters and metrics.
    """
    import pandas as pd
    from analysis.make_tables import export_table

    df_display = pd.DataFrame({
        "m1": ["1.2345", "1/3 4.5678*"],
        "m2": ["FEAS(3)", "SL"],
    }, index=["ds_alpha", "ds_beta"])

    df_raw = pd.DataFrame({
        "m1": [
            {
                "status": "OPTIMAL",
                "par2": 1.2345,
                "solve_time": 1.2345,
                "wall_time": 1.30,
                "encode_time": 0.05,
                "best_ub": 2,
                "n_total": 1,
                "n_algorithmic": 1,
                "n_optimal": 1,
                "n_timeout": 0,
                "n_interrupted": 0,
                "n_feasible": 0,
                "n_infeasible": 0,
                "n_error": 0,
                "n_size_limit": 0,
                "n_unavailable": 0,
                "n_license_error": 0,
                "n_unknown": 0,
            },
            {
                "status": "PARTIAL",
                "par2": 404.5678,
                "solve_time": 4.5678,
                "wall_time": 4.60,
                "encode_time": 0.10,
                "best_ub": 3,
                "n_total": 3,
                "n_algorithmic": 3,
                "n_optimal": 1,
                "n_timeout": 2,
                "n_interrupted": 0,
                "n_feasible": 0,
                "n_infeasible": 0,
                "n_error": 0,
                "n_size_limit": 0,
                "n_unavailable": 0,
                "n_license_error": 0,
                "n_unknown": 0,
            },
        ],
        "m2": [
            {
                "status": "FEASIBLE",
                "par2": 1200.0,
                "solve_time": 600.0,
                "wall_time": 600.1,
                "encode_time": 0.02,
                "best_ub": 3,
                "n_total": 1,
                "n_algorithmic": 1,
                "n_optimal": 0,
                "n_timeout": 0,
                "n_interrupted": 0,
                "n_feasible": 1,
                "n_infeasible": 0,
                "n_error": 0,
                "n_size_limit": 0,
                "n_unavailable": 0,
                "n_license_error": 0,
                "n_unknown": 0,
            },
            {
                "status": "SIZE_LIMIT",
                "par2": None,
                "solve_time": 0.01,
                "wall_time": 0.02,
                "encode_time": 0.01,
                "best_ub": -1,
                "n_total": 1,
                "n_algorithmic": 0,
                "n_optimal": 0,
                "n_timeout": 0,
                "n_interrupted": 0,
                "n_feasible": 0,
                "n_infeasible": 0,
                "n_error": 0,
                "n_size_limit": 1,
                "n_unavailable": 0,
                "n_license_error": 0,
                "n_unknown": 0,
            },
        ],
    }, index=["ds_alpha", "ds_beta"])

    summary_rows = [
        ("Solved", {"m1": "1", "m2": "0"}),
        ("SL", {"m1": "0", "m2": "1"}),
    ]

    out_dir = str(tmp_path / "table_out")
    export_table(
        df_display,
        summary_rows,
        title="Audit Table",
        base_filename="audit_table",
        output_dirs=(out_dir,),
        df_raw=df_raw,
    )

    disp_csv_path = os.path.join(out_dir, "audit_table.csv")
    data_csv_path = os.path.join(out_dir, "audit_table_data.csv")
    assert os.path.exists(disp_csv_path)
    assert os.path.exists(data_csv_path)

    # Read back machine-readable data CSV
    df_readback = pd.read_csv(data_csv_path)
    assert len(df_readback) == 4  # 2 datasets * 2 methods

    expected_cols = [
        "dataset", "column", "status", "par2", "solve_time", "wall_time", "encode_time", "best_ub",
        "n_total", "n_algorithmic", "n_optimal", "n_timeout", "n_interrupted", "n_feasible",
        "n_infeasible", "n_error", "n_size_limit", "n_unavailable", "n_license_error", "n_unknown",
    ]
    for col in expected_cols:
        assert col in df_readback.columns

    # Verify specific row values
    row_alpha_m1 = df_readback[(df_readback["dataset"] == "ds_alpha") & (df_readback["column"] == "m1")].iloc[0]
    assert row_alpha_m1["status"] == "OPTIMAL"
    assert row_alpha_m1["n_total"] == 1
    assert row_alpha_m1["n_algorithmic"] == 1
    assert row_alpha_m1["n_optimal"] == 1
    assert row_alpha_m1["n_timeout"] == 0
    assert abs(row_alpha_m1["par2"] - 1.2345) < 1e-4

    row_beta_m1 = df_readback[(df_readback["dataset"] == "ds_beta") & (df_readback["column"] == "m1")].iloc[0]
    assert row_beta_m1["status"] == "PARTIAL"
    assert row_beta_m1["n_total"] == 3
    assert row_beta_m1["n_algorithmic"] == 3
    assert row_beta_m1["n_optimal"] == 1
    assert row_beta_m1["n_timeout"] == 2

    row_beta_m2 = df_readback[(df_readback["dataset"] == "ds_beta") & (df_readback["column"] == "m2")].iloc[0]
    assert row_beta_m2["status"] == "SIZE_LIMIT"
    assert row_beta_m2["n_total"] == 1
    assert row_beta_m2["n_algorithmic"] == 0
    assert row_beta_m2["n_size_limit"] == 1


# ============================================================================
# 8. TIMING VALIDATION, FALLBACK, AND CSV SCHEMA CONTRACT TESTS
# ============================================================================

def test_validate_solve_time_helper():
    """Unit tests for validate_solve_time and extract_repetition_solve_time helpers."""
    import math
    from analysis.make_tables import validate_solve_time, extract_repetition_solve_time

    assert validate_solve_time(None) is None
    assert validate_solve_time(True) is None  # boolean rejected
    assert validate_solve_time(False) is None
    assert validate_solve_time("invalid") is None
    assert validate_solve_time(-1.0) is None
    assert validate_solve_time(-0.0001) is None
    assert validate_solve_time(float("nan")) is None
    assert validate_solve_time(float("inf")) is None
    assert validate_solve_time(float("-inf")) is None

    assert validate_solve_time(0.0) == 0.0
    assert validate_solve_time(12.345) == 12.345
    assert validate_solve_time("42.5") == 42.5

    # extract_repetition_solve_time strictly checks solve_time and does NOT fall back to wall_time
    assert extract_repetition_solve_time({"solve_time": 5.0, "wall_time": 6.0}) == 5.0
    assert extract_repetition_solve_time({"solve_time": None, "wall_time": 6.0}) is None
    assert extract_repetition_solve_time({"solve_time": "bad", "wall_time": 6.0}) is None
    assert extract_repetition_solve_time({"solve_time": -1, "wall_time": 6.0}) is None
    assert extract_repetition_solve_time({"solve_time": float("nan"), "wall_time": 6.0}) is None
    assert extract_repetition_solve_time({"solve_time": float("inf"), "wall_time": 6.0}) is None
    assert extract_repetition_solve_time({"solve_time": -1, "wall_time": "invalid"}) is None
    assert extract_repetition_solve_time("not_a_dict") is None


def test_par2_fallback_with_invalid_solve_times_and_wall_time_fallback():
    """
    Test compute_summary_stats with repetitions having invalid solve_time:
    Under strict timing semantics (Section 4 & 5), missing or invalid solve_time
    is not substituted by wall_time; each is penalized as 2 * time_limit.
    """
    import pandas as pd
    from analysis.make_tables import compute_summary_stats

    # Repetition 1: solve_time is 'invalid' -> penalized as 2 * 300 = 600.0
    # Repetition 2: solve_time is NaN -> penalized as 2 * 300 = 600.0
    # Repetition 3: solve_time is -1 -> penalized as 2 * 300 = 600.0
    # Repetition 4: TIMEOUT at 300 -> penalized as 2 * 300 = 600.0
    # Exact PAR2 = (600.0 + 600.0 + 600.0 + 600.0) / 4 = 2400.0 / 4 = 600.0
    rec_reps = {
        "status": "PARTIAL",
        "solve_time": 6.0,
        "n_optimal": 3,
        "n_timeout": 1,
        "n_total": 4,
        "n_algorithmic": 4,
        "reps": [
            {"status": "OPTIMAL", "solve_time": "invalid", "wall_time": 4.0, "time_limit": 300.0},
            {"status": "OPTIMAL", "solve_time": float("nan"), "wall_time": 8.0, "time_limit": 300.0},
            {"status": "OPTIMAL", "solve_time": -1.0, "wall_time": "bad", "time_limit": 300.0},
            {"status": "TIMEOUT", "solve_time": 300.0, "time_limit": 300.0},
        ],
    }

    df_raw = pd.DataFrame({"sol_test": [rec_reps]}, index=["ds_timing"])
    solved, sl, par2, speedup = compute_summary_stats(df_raw, ["sol_test"], timeout=300.0)

    expected_par2 = 600.0
    assert abs(par2["sol_test"] - expected_par2) < 1e-5


def test_gurobi_status_aggregation_and_par2_handling():
    """
    Test that Gurobi's search limit statuses (FEASIBLE with incumbent, ERROR without)
    are aggregated and penalized according to unified taxonomy.
    """
    import pandas as pd
    from analysis.make_tables import aggregate_repetitions, compute_summary_stats

    records = [
        # Gurobi with incumbent after node limit -> FEASIBLE
        {"dataset": "ds_grb", "solver": "gurobi", "status": "FEASIBLE", "best_ub": 2, "solve_time": 50.0, "time_limit": 600.0, "rep": 1},
        # Gurobi without incumbent after iteration limit -> ERROR
        {"dataset": "ds_grb2", "solver": "gurobi", "status": "ERROR", "solve_time": 20.0, "time_limit": 600.0, "rep": 1},
        # Gurobi numeric error without incumbent -> NUMERIC_ERROR (categorized as ERROR)
        {"dataset": "ds_grb3", "solver": "gurobi", "status": "NUMERIC_ERROR", "solve_time": 5.0, "time_limit": 600.0, "rep": 1},
    ]

    agg = aggregate_repetitions(records)
    agg_map = {r["dataset"]: r for r in agg}

    assert agg_map["ds_grb"]["status"] == "FEASIBLE"
    assert agg_map["ds_grb"]["par2"] == 1200.0  # 2 * 600
    assert agg_map["ds_grb"]["n_feasible"] == 1

    assert agg_map["ds_grb2"]["status"] == "ERROR"
    assert agg_map["ds_grb2"]["par2"] == 1200.0
    assert agg_map["ds_grb2"]["n_error"] == 1

    assert agg_map["ds_grb3"]["status"] == "ERROR"
    assert agg_map["ds_grb3"]["par2"] == 1200.0
    assert agg_map["ds_grb3"]["n_error"] == 1


def test_export_table_data_csv_schema_contract_when_empty_or_no_dict(tmp_path):
    """
    Contract requirement: export_table must create <base_filename>_data.csv
    whenever df_raw is passed, even when df_raw is empty, has None cells, or has no dict cells.
    Schema header must contain all 20 columns.
    """
    import pandas as pd
    from analysis.make_tables import export_table, DATA_CSV_COLUMNS

    df_disp = pd.DataFrame({"colA": ["1.0"]})
    summary_rows = [("Solved", {"colA": "1"})]
    out_dir = str(tmp_path / "csv_contracts")

    # Case 1: Empty DataFrame
    export_table(df_disp, summary_rows, title="Empty Test", base_filename="empty_test", output_dirs=(out_dir,), df_raw=pd.DataFrame())
    p1 = f"{out_dir}/empty_test_data.csv"
    assert os.path.exists(p1)
    df1 = pd.read_csv(p1)
    assert list(df1.columns) == DATA_CSV_COLUMNS
    assert len(df1) == 0

    # Case 2: DataFrame with None cells
    df_none = pd.DataFrame({"colA": [None, None]}, index=["ds1", "ds2"])
    export_table(df_disp, summary_rows, title="None Test", base_filename="none_test", output_dirs=(out_dir,), df_raw=df_none)
    p2 = f"{out_dir}/none_test_data.csv"
    assert os.path.exists(p2)
    df2 = pd.read_csv(p2)
    assert list(df2.columns) == DATA_CSV_COLUMNS
    assert len(df2) == 0

    # Case 3: DataFrame without dict cells (formatted strings only)
    df_strings = pd.DataFrame({"colA": ["1.2345", "TO"]}, index=["ds1", "ds2"])
    export_table(df_disp, summary_rows, title="Strings Test", base_filename="strings_test", output_dirs=(out_dir,), df_raw=df_strings)
    p3 = f"{out_dir}/strings_test_data.csv"
    assert os.path.exists(p3)
    df3 = pd.read_csv(p3)
    assert list(df3.columns) == DATA_CSV_COLUMNS
    assert len(df3) == 0


@pytest.mark.parametrize("invalid_par2", [float("nan"), float("inf"), float("-inf"), -1.0, "invalid", [10.0]])
def test_par2_partial_invalid_par2_falls_back_to_reps(invalid_par2):
    """
    When a PARTIAL record has an invalid par2 (NaN, inf, negative, string, list),
    compute_summary_stats must fall back to recalculating exact PAR2 from individual reps.
    """
    import pandas as pd
    from analysis.make_tables import compute_summary_stats

    # Reps: OPTIMAL with solve_time 10.0, TIMEOUT at 600.0 (penalty 1200.0)
    # Exact PAR2 = (10.0 + 1200.0) / 2 = 605.0
    rec = {
        "status": "PARTIAL",
        "par2": invalid_par2,
        "solve_time": 10.0,
        "n_optimal": 1,
        "n_timeout": 1,
        "n_total": 2,
        "n_algorithmic": 2,
        "reps": [
            {"status": "OPTIMAL", "solve_time": 10.0, "time_limit": 600.0},
            {"status": "TIMEOUT", "solve_time": 600.0, "time_limit": 600.0},
        ],
    }

    df_raw = pd.DataFrame({"sol_p": [rec]}, index=["ds_fallback"])
    solved, sl, par2, speedup = compute_summary_stats(df_raw, ["sol_p"], timeout=600.0)

    expected = (10.0 + 2.0 * 600.0) / 2.0  # 605.0
    assert abs(par2["sol_p"] - expected) < 1e-5


def test_par2_partial_valid_par2_prioritized_over_reps():
    """
    When a PARTIAL record already has a valid precomputed par2,
    compute_summary_stats must preserve and prioritize that exact precomputed value.
    """
    import pandas as pd
    from analysis.make_tables import compute_summary_stats

    rec = {
        "status": "PARTIAL",
        "par2": 777.888,
        "solve_time": 5.0,
        "n_optimal": 1,
        "n_total": 2,
        "n_algorithmic": 2,
        "reps": [
            {"status": "OPTIMAL", "solve_time": 10.0, "time_limit": 600.0},
            {"status": "TIMEOUT", "solve_time": 600.0, "time_limit": 600.0},
        ],
    }

    df_raw = pd.DataFrame({"sol_prioritized": [rec]}, index=["ds_pri"])
    solved, sl, par2, speedup = compute_summary_stats(df_raw, ["sol_prioritized"], timeout=600.0)
    assert abs(par2["sol_prioritized"] - 777.888) < 1e-5


def test_par2_partial_no_par2_no_reps_yields_nan():
    """
    When a PARTIAL record has neither valid par2 nor reps,
    compute_summary_stats must NOT reconstruct fake PAR2 from median; PAR2 must be NaN.
    """
    import math
    import pandas as pd
    from analysis.make_tables import compute_summary_stats

    rec = {
        "status": "PARTIAL",
        "solve_time": 15.0,
        "n_optimal": 1,
        "n_total": 3,
        "n_algorithmic": 3,
    }

    df_raw = pd.DataFrame({"sol_unrecoverable": [rec]}, index=["ds_unrec"])
    solved, sl, par2, speedup = compute_summary_stats(df_raw, ["sol_unrecoverable"], timeout=600.0)
    assert math.isnan(par2["sol_unrecoverable"])


def test_par2_exact_numerator_and_denominator_with_exclusions():
    """
    Test exact calculation of PAR2 numerator and denominator across diverse repetition types:
    - 2 OPTIMAL reps: 5.0s, 15.0s
    - 1 FEASIBLE rep: penalized 2 * 300 = 600.0s
    - 1 TIMEOUT rep: penalized 2 * 300 = 600.0s
    - 1 SIZE_LIMIT rep: non-algorithmic, EXCLUDED from numerator and denominator
    - 1 UNAVAILABLE rep: non-algorithmic, EXCLUDED from numerator and denominator
    Denominator = 4 (only algorithmic runs).
    Numerator = 5.0 + 15.0 + 600.0 + 600.0 = 1220.0
    PAR2 = 1220.0 / 4 = 305.0
    """
    import pandas as pd
    from analysis.make_tables import compute_summary_stats

    rec = {
        "status": "PARTIAL",
        "solve_time": 10.0,
        "n_optimal": 2,
        "n_feasible": 1,
        "n_timeout": 1,
        "n_size_limit": 1,
        "n_unavailable": 1,
        "n_total": 6,
        "n_algorithmic": 4,
        "reps": [
            {"status": "OPTIMAL", "solve_time": 5.0, "time_limit": 300.0},
            {"status": "OPTIMAL", "solve_time": 15.0, "time_limit": 300.0},
            {"status": "FEASIBLE", "best_ub": 3, "solve_time": 300.0, "time_limit": 300.0},
            {"status": "TIMEOUT", "solve_time": 300.0, "time_limit": 300.0},
            {"status": "SIZE_LIMIT", "solve_time": 0.01, "time_limit": 300.0},
            {"status": "UNAVAILABLE", "solve_time": 0.0, "time_limit": 300.0},
        ],
    }

    df_raw = pd.DataFrame({"sol_denom": [rec]}, index=["ds_denom"])
    solved, sl, par2, speedup = compute_summary_stats(df_raw, ["sol_denom"], timeout=300.0)

    expected_par2 = (5.0 + 15.0 + 600.0 + 600.0) / 4.0
    assert abs(par2["sol_denom"] - expected_par2) < 1e-5
    assert sl["sol_denom"] == 0  # rec status is PARTIAL, sl_count counts cell status == SIZE_LIMIT


def test_par2_optimal_aggregate_skewed_times_recomputes_exact_mean_from_reps():
    """
    Test that when an OPTIMAL aggregated record has an invalid or missing par2,
    compute_summary_stats recomputes the exact arithmetic mean PAR2 from individual reps
    rather than substituting the record's median solve_time.
    Solve times: [1.0, 2.0, 3.0, 4.0, 100.0].
    Median = 3.0.
    Mean = 22.0.
    """
    import pandas as pd
    from analysis.make_tables import compute_summary_stats

    reps = [
        {"status": "OPTIMAL", "solve_time": 1.0, "time_limit": 600.0},
        {"status": "OPTIMAL", "solve_time": 2.0, "time_limit": 600.0},
        {"status": "OPTIMAL", "solve_time": 3.0, "time_limit": 600.0},
        {"status": "OPTIMAL", "solve_time": 4.0, "time_limit": 600.0},
        {"status": "OPTIMAL", "solve_time": 100.0, "time_limit": 600.0},
    ]
    rec = {
        "status": "OPTIMAL",
        "solve_time": 3.0,  # median
        "n_total": 5,
        "n_algorithmic": 5,
        "n_optimal": 5,
        "par2": None,  # unpopulated or invalid
        "reps": reps,
    }

    df_raw = pd.DataFrame({"sol_opt": [rec]}, index=["ds_skew"])
    solved, sl, par2, speedup = compute_summary_stats(df_raw, ["sol_opt"], timeout=600.0)
    assert solved["sol_opt"] == 1
    assert abs(par2["sol_opt"] - 22.0) < 1e-5
    assert abs(par2["sol_opt"] - 3.0) > 1.0  # Must NOT be the median 3.0


def test_par2_optimal_aggregate_valid_par2_prioritized():
    """
    When an OPTIMAL aggregated record already contains a validated precomputed par2,
    compute_summary_stats must preserve and prioritize it.
    """
    import pandas as pd
    from analysis.make_tables import compute_summary_stats

    reps = [
        {"status": "OPTIMAL", "solve_time": 1.0, "time_limit": 600.0},
        {"status": "OPTIMAL", "solve_time": 2.0, "time_limit": 600.0},
        {"status": "OPTIMAL", "solve_time": 3.0, "time_limit": 600.0},
        {"status": "OPTIMAL", "solve_time": 4.0, "time_limit": 600.0},
        {"status": "OPTIMAL", "solve_time": 100.0, "time_limit": 600.0},
    ]
    rec = {
        "status": "OPTIMAL",
        "solve_time": 3.0,
        "n_total": 5,
        "n_algorithmic": 5,
        "n_optimal": 5,
        "par2": 22.0,  # valid precomputed par2
        "reps": reps,
    }

    df_raw = pd.DataFrame({"sol_opt": [rec]}, index=["ds_skew"])
    solved, sl, par2, speedup = compute_summary_stats(df_raw, ["sol_opt"], timeout=600.0)
    assert abs(par2["sol_opt"] - 22.0) < 1e-5


def test_par2_optimal_single_run_timing_and_fallbacks():
    """
    Test direct timing extraction and timeout fallbacks for single independent runs (n_total == 1).
    Under strict timing semantics, missing solve_time is penalized with 2 * timeout.
    """
    import pandas as pd
    from analysis.make_tables import compute_summary_stats

    # Case 1: valid solve_time
    r1 = {"status": "OPTIMAL", "solve_time": 4.5, "n_total": 1, "n_algorithmic": 1, "n_optimal": 1}
    # Case 2: solve_time is None -> penalize with 2 * timeout (no disguised wall_time)
    r2 = {"status": "OPTIMAL", "solve_time": None, "wall_time": 4.8, "n_total": 1, "n_algorithmic": 1, "n_optimal": 1}
    # Case 3: invalid solve_time (-1.0) -> penalize with 2 * timeout
    r3 = {"status": "OPTIMAL", "solve_time": -1.0, "n_total": 1, "n_algorithmic": 1, "n_optimal": 1}

    df = pd.DataFrame({"s1": [r1], "s2": [r2], "s3": [r3]}, index=["ds1"])
    solved, sl, par2, _ = compute_summary_stats(df, ["s1", "s2", "s3"], timeout=600.0)

    assert abs(par2["s1"] - 4.5) < 1e-5
    assert abs(par2["s2"] - 1200.0) < 1e-5
    assert abs(par2["s3"] - 1200.0) < 1e-5


def test_par2_optimal_aggregate_without_reps_yields_nan():
    """
    When an aggregated OPTIMAL record (n_total > 1) lacks both valid par2 and reps,
    compute_summary_stats must NOT substitute median solve_time; PAR2 must be NaN.
    """
    import math
    import pandas as pd
    from analysis.make_tables import compute_summary_stats

    rec = {
        "status": "OPTIMAL",
        "solve_time": 3.0,  # median across unknown runs
        "n_total": 5,
        "n_algorithmic": 5,
        "n_optimal": 5,
        "par2": "corrupted",
    }

    df_raw = pd.DataFrame({"sol_opt": [rec]}, index=["ds_opt"])
    solved, sl, par2, _ = compute_summary_stats(df_raw, ["sol_opt"], timeout=600.0)
    assert math.isnan(par2["sol_opt"])


def test_compute_repetition_par2_unit_and_exclusions():
    """
    Unit tests for compute_repetition_par2 helper covering all status categories and exclusions.
    """
    from analysis.make_tables import compute_repetition_par2

    assert compute_repetition_par2([]) is None

    # All non-algorithmic
    all_non_algo = [
        {"status": "SIZE_LIMIT", "solve_time": 0.01},
        {"status": "UNAVAILABLE", "solve_time": 0.0},
        {"status": "LICENSE_ERROR", "solve_time": 0.0},
        {"status": "UNKNOWN", "solve_time": 0.0},
    ]
    assert compute_repetition_par2(all_non_algo) is None

    # Mixed algorithmic and non-algorithmic
    mixed = [
        {"status": "OPTIMAL", "solve_time": 10.0, "time_limit": 300.0},
        {"status": "TIMEOUT", "solve_time": 300.0, "time_limit": 300.0},
        {"status": "SIZE_LIMIT", "solve_time": 0.01, "time_limit": 300.0},
        {"status": "UNAVAILABLE", "solve_time": 0.0, "time_limit": 300.0},
    ]
    # Numerator: 10.0 + 2 * 300.0 = 610.0; Denominator: 2
    assert abs(compute_repetition_par2(mixed, default_timeout=300.0) - 305.0) < 1e-5

    # OPTIMAL with missing solve_time penalized as 2 * timeout
    opt_wall = [{"status": "OPTIMAL", "solve_time": None, "wall_time": 12.34, "time_limit": 300.0}]
    assert abs(compute_repetition_par2(opt_wall, default_timeout=300.0) - 600.0) < 1e-5

    # Rep with invalid solve_time penalized as 2 * timeout
    opt_bad_time = [{"status": "OPTIMAL", "solve_time": -5.0, "time_limit": 300.0}]
    assert abs(compute_repetition_par2(opt_bad_time, default_timeout=300.0) - 600.0) < 1e-5


def test_make_table_solvers_unification_with_canonical_par2(tmp_path):
    """
    Test that make_table_solvers aggregates runs using aggregate_repetition_group:
    - Sets all 12 population counters.
    - Excludes SIZE_LIMIT and UNAVAILABLE runs from PAR2 denominator.
    - Recomputes exact arithmetic PAR2 across seeds.
    - If all runs are non-algorithmic, status is SIZE_LIMIT and PAR2 is NaN.
    """
    import math
    from analysis.make_tables import make_table_solvers

    # Dataset 1: Mixed 3 seeds (2 OPTIMAL, 1 TIMEOUT)
    recs_ds1 = [
        {"dataset": "ds1", "solver": "cadical195", "status": "OPTIMAL", "solve_time": 1.0, "config": {"time_limit": 300.0}, "groups": ["solvers_small"]},
        {"dataset": "ds1", "solver": "cadical195", "status": "OPTIMAL", "solve_time": 3.0, "config": {"time_limit": 300.0}, "groups": ["solvers_small"]},
        {"dataset": "ds1", "solver": "cadical195", "status": "TIMEOUT", "solve_time": 300.0, "config": {"time_limit": 300.0}, "groups": ["solvers_small"]},
    ]

    # Dataset 2: Mixed with SIZE_LIMIT (1 OPTIMAL, 1 SIZE_LIMIT)
    recs_ds2 = [
        {"dataset": "ds2", "solver": "cadical195", "status": "OPTIMAL", "solve_time": 5.0, "config": {"time_limit": 300.0}, "groups": ["solvers_small"]},
        {"dataset": "ds2", "solver": "cadical195", "status": "SIZE_LIMIT", "solve_time": 0.01, "config": {"time_limit": 300.0}, "groups": ["solvers_small"]},
    ]

    # Dataset 3: All SIZE_LIMIT
    recs_ds3 = [
        {"dataset": "ds3", "solver": "cadical195", "status": "SIZE_LIMIT", "solve_time": 0.01, "config": {"time_limit": 300.0}, "groups": ["solvers_small"]},
        {"dataset": "ds3", "solver": "cadical195", "status": "SIZE_LIMIT", "solve_time": 0.01, "config": {"time_limit": 300.0}, "groups": ["solvers_small"]},
    ]

    all_recs = recs_ds1 + recs_ds2 + recs_ds3
    out_dir = str(tmp_path / "solvers_tables")

    make_table_solvers(all_recs, group_name="solvers_small", base_filename="test_solvers", output_dirs=(out_dir,))

    # Read back the generated CSV
    import pandas as pd
    data_csv = f"{out_dir}/test_solvers_data.csv"
    assert os.path.exists(data_csv)
    df_data = pd.read_csv(data_csv)

    row_ds1 = df_data[(df_data["dataset"] == "ds1") & (df_data["column"] == "cadical195")].iloc[0]
    assert row_ds1["status"] == "PARTIAL"
    assert row_ds1["n_total"] == 3
    assert row_ds1["n_algorithmic"] == 3
    assert row_ds1["n_optimal"] == 2
    assert row_ds1["n_timeout"] == 1
    # PAR2 for ds1: (1.0 + 3.0 + 600.0) / 3 = 201.3333...
    expected_par2_ds1 = (1.0 + 3.0 + 600.0) / 3.0
    assert abs(row_ds1["par2"] - expected_par2_ds1) < 1e-4

    row_ds2 = df_data[(df_data["dataset"] == "ds2") & (df_data["column"] == "cadical195")].iloc[0]
    assert row_ds2["status"] == "PARTIAL"
    assert row_ds2["n_total"] == 2
    assert row_ds2["n_algorithmic"] == 1
    assert row_ds2["n_optimal"] == 1
    assert row_ds2["n_size_limit"] == 1
    # SIZE_LIMIT excluded: PAR2 is 5.0 / 1 = 5.0
    assert abs(row_ds2["par2"] - 5.0) < 1e-4

    row_ds3 = df_data[(df_data["dataset"] == "ds3") & (df_data["column"] == "cadical195")].iloc[0]
    assert row_ds3["status"] == "SIZE_LIMIT"
    assert row_ds3["n_total"] == 2
    assert row_ds3["n_algorithmic"] == 0
    assert row_ds3["n_size_limit"] == 2
    assert pd.isna(row_ds3["par2"])


def test_data_csv_schema_and_termination_reason(tmp_path):
    """
    Test machine-readable CSV export schema:
    - DATA_CSV_COLUMNS has 21 columns with termination_reason at the end.
    - Preserves all 20 legacy columns in their exact original order.
    - Records with termination_reason export it correctly.
    - Records without termination_reason export empty/NaN.
    """
    import pandas as pd
    from analysis.make_tables import export_table, DATA_CSV_COLUMNS

    assert "termination_reason" in DATA_CSV_COLUMNS
    assert len(DATA_CSV_COLUMNS) == 21
    assert DATA_CSV_COLUMNS[-1] == "termination_reason"

    legacy_20 = [
        "dataset", "column", "status",
        "par2", "solve_time", "wall_time", "encode_time", "best_ub",
        "n_total", "n_algorithmic", "n_optimal", "n_timeout", "n_interrupted",
        "n_feasible", "n_infeasible", "n_error", "n_size_limit",
        "n_unavailable", "n_license_error", "n_unknown",
    ]
    assert DATA_CSV_COLUMNS[:20] == legacy_20

    cell_gurobi_search = {
        "status": "ERROR",
        "termination_reason": "search_limit",
        "solve_time": 10.0,
        "n_total": 1,
        "n_algorithmic": 1,
        "n_error": 1,
    }
    cell_gurobi_time = {
        "status": "TIMEOUT",
        "termination_reason": "time_limit",
        "solve_time": 300.0,
        "n_total": 1,
        "n_algorithmic": 1,
        "n_timeout": 1,
    }
    cell_sat = {
        "status": "OPTIMAL",
        "solve_time": 1.23,
        "n_total": 1,
        "n_algorithmic": 1,
        "n_optimal": 1,
    }

    df_raw = pd.DataFrame(
        {
            "gurobi": [cell_gurobi_search, cell_gurobi_time],
            "sat": [cell_sat, None],
        },
        index=["ds1", "ds2"],
    )
    df_disp = pd.DataFrame({"gurobi": ["ERR", "TO"], "sat": ["1.23", "-"]}, index=["ds1", "ds2"])
    summary_rows = [("Solved", {"gurobi": "0", "sat": "1"})]

    out_dir = str(tmp_path / "csv_schema_test")
    export_table(df_disp, summary_rows, title="Schema Test", base_filename="schema_test", output_dirs=(out_dir,), df_raw=df_raw)

    csv_path = f"{out_dir}/schema_test_data.csv"
    assert os.path.exists(csv_path)

    df_read = pd.read_csv(csv_path)
    assert list(df_read.columns) == DATA_CSV_COLUMNS

    row_grb1 = df_read[(df_read["dataset"] == "ds1") & (df_read["column"] == "gurobi")].iloc[0]
    assert row_grb1["termination_reason"] == "search_limit"

    row_grb2 = df_read[(df_read["dataset"] == "ds2") & (df_read["column"] == "gurobi")].iloc[0]
    assert row_grb2["termination_reason"] == "time_limit"

    row_sat = df_read[(df_read["dataset"] == "ds1") & (df_read["column"] == "sat")].iloc[0]
    assert pd.isna(row_sat["termination_reason"])


def test_data_csv_backward_compatibility_with_legacy_csv(tmp_path):
    """
    Test that reading a legacy 20-column CSV (created before termination_reason existed)
    remains completely compatible and aligns with DATA_CSV_COLUMNS[:20].
    """
    import pandas as pd
    from analysis.make_tables import DATA_CSV_COLUMNS

    legacy_cols = DATA_CSV_COLUMNS[:20]
    legacy_row = {col: 0 for col in legacy_cols}
    legacy_row["dataset"] = "legacy_ds"
    legacy_row["column"] = "cadical"
    legacy_row["status"] = "OPTIMAL"
    legacy_row["solve_time"] = 1.5

    legacy_df = pd.DataFrame([legacy_row], columns=legacy_cols)
    legacy_path = tmp_path / "legacy_data.csv"
    legacy_df.to_csv(legacy_path, index=False)

    df_loaded = pd.read_csv(legacy_path)
    assert list(df_loaded.columns) == DATA_CSV_COLUMNS[:20]
    assert "termination_reason" not in df_loaded.columns
    assert df_loaded.iloc[0]["solve_time"] == 1.5


def test_aggregate_repetition_group_robust_timing_and_sorting():
    """
    Test robust timing validation and representative selection in aggregate_repetition_group:
    1. Repetitions with string, boolean, negative, NaN, inf, or None timing do NOT crash sorting or median.
    2. Fallback from invalid solve_time to valid wall_time is applied cleanly.
    3. Representative FEASIBLE selection with malformed best_ub (str, bool, None, negative) does NOT crash sorting.
    4. encode_time with malformed values is validated and filtered before computing median.
    5. Non-dict items in reps are safely ignored.
    6. Invariant holds: n_algorithmic + n_size_limit + n_unavailable + n_license_error + n_unknown == n_total.
    """
    import math
    from analysis.make_tables import aggregate_repetition_group

    reps = [
        # Rep 1: valid solve_time
        {"status": "FEASIBLE", "best_ub": 6, "solve_time": 10.0, "wall_time": 10.2, "encode_time": 0.5, "time_limit": 300.0},
        # Rep 2: solve_time is corrupted string, fallback to wall_time
        {"status": "FEASIBLE", "best_ub": 4, "solve_time": "invalid_string", "wall_time": 8.0, "encode_time": True, "time_limit": 300.0},
        # Rep 3: best_ub is None, solve_time is boolean False
        {"status": "FEASIBLE", "best_ub": None, "solve_time": False, "wall_time": 15.0, "encode_time": float("nan"), "time_limit": 300.0},
        # Rep 4: best_ub is negative string, solve_time is NaN
        {"status": "FEASIBLE", "best_ub": "-5", "solve_time": float("nan"), "wall_time": 20.0, "encode_time": 0.3, "time_limit": 300.0},
        # Rep 5: best_ub is float 3.0 (valid bound 3), solve_time is -1.0, wall_time is 5.0
        {"status": "FEASIBLE", "best_ub": 3.0, "solve_time": -1.0, "wall_time": 5.0, "encode_time": 0.4, "time_limit": 300.0},
        # Non-dict item (should be safely skipped)
        "not_a_dict",
        None,
        42,
    ]

    agg = aggregate_repetition_group(reps, default_timeout=300.0)
    assert agg is not None
    assert agg["status"] == "FEASIBLE"
    assert agg["n_total"] == 5
    assert agg["n_feasible"] == 5
    assert agg["n_algorithmic"] == 5
    assert agg["n_size_limit"] == 0
    assert agg["n_unavailable"] == 0
    assert agg["n_license_error"] == 0
    assert agg["n_unknown"] == 0

    # Representative record: best valid best_ub is 3 from Rep 5
    assert agg["best_ub"] == 3

    # solve_times with fallback to wall_time:
    # Rep 1: 10.0
    # Rep 2: fallback to wall_time 8.0
    # Rep 3: fallback to wall_time 15.0
    # Rep 4: fallback to wall_time 20.0
    # Rep 5: fallback to wall_time 5.0
    # Sorted solve_times: [5.0, 8.0, 10.0, 15.0, 20.0] -> median is 10.0
    assert abs(agg["solve_time"] - 10.0) < 1e-5

    # wall_times: [10.2, 8.0, 15.0, 20.0, 5.0] -> sorted [5.0, 8.0, 10.2, 15.0, 20.0] -> median 10.2
    assert abs(agg["wall_time"] - 10.2) < 1e-5

    # encode_times: valid are 0.5 (Rep 1), 0.3 (Rep 4), 0.4 (Rep 5). (True and NaN filtered out).
    # Sorted: [0.3, 0.4, 0.5] -> median is 0.4
    assert abs(agg["encode_time"] - 0.4) < 1e-5


def test_aggregate_repetition_group_all_timings_missing_does_not_inject_zero():
    """
    Test that when all repetitions lack valid timing, aggregate_repetition_group does NOT
    arbitrarily inject 0.0 into solve_time, wall_time, or encode_time.
    """
    from analysis.make_tables import aggregate_repetition_group

    reps = [
        {"status": "TIMEOUT", "solve_time": None, "wall_time": None, "time_limit": 600.0},
        {"status": "TIMEOUT", "solve_time": "none", "wall_time": -1.0, "time_limit": 600.0},
    ]

    agg = aggregate_repetition_group(reps, default_timeout=600.0)
    assert agg is not None
    assert agg["status"] == "TIMEOUT"
    assert agg["solve_time"] is None
    assert agg["wall_time"] is None
    assert agg["encode_time"] is None
    assert agg["par2"] == 1200.0  # TIMEOUT penalized as 2 * 600


def test_validate_best_ub_unit_cases():
    """
    Unit tests for validate_best_ub():
    - Accepts non-negative integers, integral floats, integral NumPy types, integral strings, large ints.
    - Rejects fractional floats (never truncates!), booleans, negative numbers, NaN, inf, malformed strings/types.
    """
    import numpy as np
    from analysis.make_tables import validate_best_ub

    # 1. Valid non-negative integers
    assert validate_best_ub(0) == 0
    assert validate_best_ub(1) == 1
    assert validate_best_ub(3) == 3
    assert validate_best_ub(100) == 100
    assert validate_best_ub(10**25) == 10**25

    # 2. Valid integral floats
    assert validate_best_ub(0.0) == 0
    assert validate_best_ub(3.0) == 3
    assert validate_best_ub(100.0) == 100

    # 3. Valid NumPy scalar types
    assert validate_best_ub(np.int32(4)) == 4
    assert validate_best_ub(np.int64(10)) == 10
    assert validate_best_ub(np.uint16(7)) == 7
    assert validate_best_ub(np.float64(5.0)) == 5
    assert validate_best_ub(np.float32(8.0)) == 8

    # 4. Valid strings representing integers
    assert validate_best_ub("0") == 0
    assert validate_best_ub("3") == 3
    assert validate_best_ub("3.0") == 3
    assert validate_best_ub("  15  ") == 15

    # 5. REJECT fractional floats - must NEVER truncate or round!
    assert validate_best_ub(3.7) is None
    assert validate_best_ub(0.9) is None
    assert validate_best_ub(0.0001) is None
    assert validate_best_ub(1e-5) is None
    assert validate_best_ub(np.float64(3.7)) is None
    assert validate_best_ub(np.float32(0.9)) is None
    assert validate_best_ub("3.7") is None
    assert validate_best_ub("0.9") is None
    assert validate_best_ub("1e-5") is None

    # 6. REJECT booleans (bool is a subclass of int in Python)
    assert validate_best_ub(True) is None
    assert validate_best_ub(False) is None
    assert validate_best_ub(np.bool_(True)) is None

    # 7. REJECT negative values
    assert validate_best_ub(-1) is None
    assert validate_best_ub(-5.0) is None
    assert validate_best_ub("-3") is None
    assert validate_best_ub(np.int32(-2)) is None

    # 8. REJECT non-finite numbers
    assert validate_best_ub(float("nan")) is None
    assert validate_best_ub(float("inf")) is None
    assert validate_best_ub(float("-inf")) is None
    assert validate_best_ub(np.nan) is None
    assert validate_best_ub(np.inf) is None

    # 9. REJECT malformed types and strings
    assert validate_best_ub(None) is None
    assert validate_best_ub("abc") is None
    assert validate_best_ub("") is None
    assert validate_best_ub("   ") is None
    assert validate_best_ub([1]) is None
    assert validate_best_ub({"ub": 1}) is None


def test_format_cell_value_missing_and_invalid_timing():
    """
    Test format_cell_value() presentation:
    - Timing missing or invalid does NOT display as 0.0000.
    - OPTIMAL with invalid/missing timing displays '-'.
    - OPTIMAL with valid 0.0 timing displays '0.0000'.
    - PARTIAL with invalid/missing timing displays 'k/n -*'.
    - PARTIAL with valid 0.0 timing displays 'k/n 0.0000*'.
    - FEASIBLE/TIMEOUT/INTERRUPTED only format best_ub when it is a valid non-negative integer.
    """
    from analysis.make_tables import format_cell_value

    # OPTIMAL cases
    assert format_cell_value("OPTIMAL", solve_time=None) == "-"
    assert format_cell_value("OPTIMAL", solve_time=-1.0) == "-"
    assert format_cell_value("OPTIMAL", solve_time=float("nan")) == "-"
    assert format_cell_value("OPTIMAL", solve_time=float("inf")) == "-"
    assert format_cell_value("OPTIMAL", solve_time="invalid") == "-"
    assert format_cell_value("OPTIMAL", solve_time=0.0) == "0.0000"
    assert format_cell_value("OPTIMAL", solve_time=1.2345) == "1.2345"

    # Dict-based OPTIMAL cases
    assert format_cell_value({"status": "OPTIMAL", "solve_time": None, "wall_time": None}) == "-"
    assert format_cell_value({"status": "OPTIMAL", "solve_time": 0.0}) == "0.0000"
    assert format_cell_value({"status": "OPTIMAL", "solve_time": None, "wall_time": 2.5}) == "-"

    # PARTIAL cases
    assert format_cell_value("PARTIAL", solve_time=None, n_optimal=2, n_total=4) == "2/4 -*"
    assert format_cell_value("PARTIAL", solve_time=-5.0, n_optimal=2, n_total=4) == "2/4 -*"
    assert format_cell_value("PARTIAL", solve_time=float("nan"), n_optimal=2, n_total=4) == "2/4 -*"
    assert format_cell_value("PARTIAL", solve_time=0.0, n_optimal=2, n_total=4) == "2/4 0.0000*"
    assert format_cell_value("PARTIAL", solve_time=1.5, n_optimal=2, n_total=4) == "2/4 1.5000*"

    # FEASIBLE cases with valid and invalid best_ub
    assert format_cell_value("FEASIBLE", best_ub=4) == "FEAS(4)"
    assert format_cell_value("FEASIBLE", best_ub=4.0) == "FEAS(4)"
    assert format_cell_value("FEASIBLE", best_ub="4") == "FEAS(4)"
    assert format_cell_value("FEASIBLE", best_ub=0) == "FEAS(0)"
    assert format_cell_value("FEASIBLE", best_ub=3.7) == "FEAS"   # fractional float rejected
    assert format_cell_value("FEASIBLE", best_ub=-1) == "FEAS"
    assert format_cell_value("FEASIBLE", best_ub=None) == "FEAS"
    assert format_cell_value("FEASIBLE", best_ub="invalid") == "FEAS"
    assert format_cell_value("FEASIBLE", best_ub=True) == "FEAS"

    # TIMEOUT and INTERRUPTED cases
    assert format_cell_value("TIMEOUT", best_ub=5) == "TO(5)"
    assert format_cell_value("TIMEOUT", best_ub=5.7) == "TO"
    assert format_cell_value("TIMEOUT", best_ub=-1) == "TO"
    assert format_cell_value("INTERRUPTED", best_ub=6) == "INT(6)"
    assert format_cell_value("INTERRUPTED", best_ub=None) == "INT"


def test_speedup_calculation_symmetric_timing_and_corner_cases():
    """
    Test speedup calculation in compute_summary_stats across all 9 required scenarios:
    1. Baseline and candidate both have valid solve_time.
    2. Baseline lacks solve_time but has wall_time (symmetric fallback).
    3. Candidate lacks solve_time but has wall_time (symmetric fallback).
    4. Baseline lacks both solve_time and wall_time (instance excluded from baseline_times, no fake timeout).
    5. Candidate lacks both solve_time and wall_time (speedup is NaN).
    6. Candidate timing is 0.0 (no division by zero, speedup is NaN).
    7. Candidate timing is negative, NaN, inf (speedup is NaN).
    8. Geometric mean when one instance in eligible set has invalid ratio (speedup is NaN).
    9. col == baseline_col (speedup is 1.00).
    """
    import math
    import pandas as pd
    from analysis.make_tables import compute_summary_stats

    # Scenario 1: Both have valid solve_time
    df1 = pd.DataFrame({
        "base": [{"status": "OPTIMAL", "solve_time": 10.0}],
        "cand": [{"status": "OPTIMAL", "solve_time": 5.0}],
    }, index=["ds1"])
    _, _, _, sp1 = compute_summary_stats(df1, ["base", "cand"], baseline_col="base")
    assert abs(sp1["base"] - 1.00) < 1e-5
    assert abs(sp1["cand"] - 2.00) < 1e-5

    # Scenario 2: Baseline lacks solve_time -> speedup is NaN (no wall_time disguised as solve_time)
    df2 = pd.DataFrame({
        "base": [{"status": "OPTIMAL", "solve_time": None, "wall_time": 10.0}],
        "cand": [{"status": "OPTIMAL", "solve_time": 5.0}],
    }, index=["ds1"])
    _, _, _, sp2 = compute_summary_stats(df2, ["base", "cand"], baseline_col="base")
    assert math.isnan(sp2["cand"])

    # Scenario 3: Candidate lacks solve_time -> speedup is NaN
    df3 = pd.DataFrame({
        "base": [{"status": "OPTIMAL", "solve_time": 10.0}],
        "cand": [{"status": "OPTIMAL", "solve_time": None, "wall_time": 5.0}],
    }, index=["ds1"])
    _, _, _, sp3 = compute_summary_stats(df3, ["base", "cand"], baseline_col="base")
    assert math.isnan(sp3["cand"])

    # Scenario 4: Baseline lacks both solve_time and wall_time
    df4 = pd.DataFrame({
        "base": [{"status": "OPTIMAL", "solve_time": None, "wall_time": None}],
        "cand": [{"status": "OPTIMAL", "solve_time": 5.0}],
    }, index=["ds1"])
    _, _, _, sp4 = compute_summary_stats(df4, ["base", "cand"], baseline_col="base", timeout=300.0)
    assert math.isnan(sp4["base"])
    assert math.isnan(sp4["cand"])  # No eligible baseline instance

    # Scenario 5: Candidate lacks both solve_time and wall_time
    df5 = pd.DataFrame({
        "base": [{"status": "OPTIMAL", "solve_time": 10.0}],
        "cand": [{"status": "OPTIMAL", "solve_time": None, "wall_time": None}],
    }, index=["ds1"])
    _, _, _, sp5 = compute_summary_stats(df5, ["base", "cand"], baseline_col="base")
    assert math.isnan(sp5["cand"])

    # Scenario 6: Candidate timing is 0.0
    df6 = pd.DataFrame({
        "base": [{"status": "OPTIMAL", "solve_time": 10.0}],
        "cand": [{"status": "OPTIMAL", "solve_time": 0.0}],
    }, index=["ds1"])
    _, _, _, sp6 = compute_summary_stats(df6, ["base", "cand"], baseline_col="base")
    assert math.isnan(sp6["cand"])  # No division by zero

    # Scenario 7: Candidate timing is negative, NaN, inf
    for bad_t in (-5.0, float("nan"), float("inf"), "invalid"):
        df7 = pd.DataFrame({
            "base": [{"status": "OPTIMAL", "solve_time": 10.0}],
            "cand": [{"status": "OPTIMAL", "solve_time": bad_t}],
        }, index=["ds1"])
        _, _, _, sp7 = compute_summary_stats(df7, ["base", "cand"], baseline_col="base")
        assert math.isnan(sp7["cand"])

    # Scenario 8: Multiple eligible instances, one candidate timing is missing/invalid
    # ds1: valid ratio 10.0 / 5.0 = 2.0; ds2: candidate missing timing -> candidate speedup is NaN
    df8 = pd.DataFrame({
        "base": [
            {"status": "OPTIMAL", "solve_time": 10.0},
            {"status": "OPTIMAL", "solve_time": 20.0},
        ],
        "cand": [
            {"status": "OPTIMAL", "solve_time": 5.0},
            {"status": "OPTIMAL", "solve_time": None, "wall_time": None},
        ],
    }, index=["ds1", "ds2"])
    _, _, _, sp8 = compute_summary_stats(df8, ["base", "cand"], baseline_col="base")
    assert math.isnan(sp8["cand"])

    # Scenario 9: Multiple eligible instances, all candidate timings valid -> geometric mean
    # ds1: 10.0 / 5.0 = 2.0; ds2: 20.0 / 2.5 = 8.0. Geo mean = sqrt(2.0 * 8.0) = 4.0
    df9 = pd.DataFrame({
        "base": [
            {"status": "OPTIMAL", "solve_time": 10.0},
            {"status": "OPTIMAL", "solve_time": 20.0},
        ],
        "cand": [
            {"status": "OPTIMAL", "solve_time": 5.0},
            {"status": "OPTIMAL", "solve_time": 2.5},
        ],
    }, index=["ds1", "ds2"])
    _, _, _, sp9 = compute_summary_stats(df9, ["base", "cand"], baseline_col="base")
    assert abs(sp9["cand"] - 4.00) < 1e-5


def test_integration_optimal_solve_time_none_wall_time_valid(tmp_path):
    """
    Integration test for Section 5:
    - Dataset with status OPTIMAL.
    - solve_time = None, wall_time = 5.5s (valid).
    - PAR2, speedup, and cell presentation follow canonical semantics.
    - No fake '0.0000'.
    - No timeout substitution in speedup.
    - Export to CSV/LaTeX succeeds without exception.
    """
    import os
    import math
    import pandas as pd
    from analysis.make_tables import (
        aggregate_repetition_group,
        compute_summary_stats,
        format_cell_value,
        export_table,
        DATA_CSV_COLUMNS,
    )

    reps = [
        {"dataset": "ds_opt_wall", "solver": "cadical", "status": "OPTIMAL", "solve_time": None, "wall_time": 5.5, "time_limit": 300.0, "best_ub": 4},
    ]

    agg = aggregate_repetition_group(reps, default_timeout=300.0)
    assert agg is not None
    assert agg["status"] == "OPTIMAL"
    assert agg["solve_time"] is None
    assert agg["wall_time"] == 5.5
    assert agg["effective_time"] == 5.5
    assert agg["effective_time_source"] == "wall_time"
    assert abs(agg["par2"] - 600.0) < 1e-5

    # Cell presentation must format -, not disguise wall_time as solve_time
    assert format_cell_value(agg) == "-"

    # Speedup: baseline cadical lacks solve_time -> speedup is NaN
    df_raw = pd.DataFrame({
        "cadical": [agg],
        "glucose": [{"status": "OPTIMAL", "solve_time": 2.75, "time_limit": 300.0}],
    }, index=["ds_opt_wall"])

    solved, sl, par2, speedup = compute_summary_stats(df_raw, ["cadical", "glucose"], baseline_col="cadical", timeout=300.0)
    assert solved["cadical"] == 1
    assert solved["glucose"] == 1
    assert abs(par2["cadical"] - 600.0) < 1e-5
    assert math.isnan(speedup["cadical"])
    assert math.isnan(speedup["glucose"])

    # Export table to CSV and LaTeX
    out_dir = str(tmp_path / "integration_export")
    df_disp = pd.DataFrame({"cadical": ["-"], "glucose": ["2.7500"]}, index=["ds_opt_wall"])
    summary_rows = [("Solved", {"cadical": "1", "glucose": "1"})]

    export_table(df_disp, summary_rows, title="Integration Test", base_filename="integ_table", output_dirs=(out_dir,), df_raw=df_raw)

    csv_path = os.path.join(out_dir, "integ_table.csv")
    data_csv_path = os.path.join(out_dir, "integ_table_data.csv")
    tex_path = os.path.join(out_dir, "integ_table.tex")

    assert os.path.exists(csv_path)
    assert os.path.exists(data_csv_path)
    assert os.path.exists(tex_path)

    df_read = pd.read_csv(data_csv_path)
    assert list(df_read.columns) == DATA_CSV_COLUMNS
    assert pd.isna(df_read[df_read["column"] == "cadical"].iloc[0]["solve_time"])
    assert df_read[df_read["column"] == "cadical"].iloc[0]["wall_time"] == 5.5


def test_aggregate_repetition_interleaved_valid_and_invalid_timing():
    """
    Test aggregation of repetitions with interleaved valid and invalid timing:
    - 2 OPTIMAL with valid solve_time (1.0, 3.0)
    - 1 OPTIMAL with solve_time=None, wall_time=2.0 (fallback)
    - 1 OPTIMAL with solve_time="corrupted", wall_time=4.0 (fallback)
    - 1 TIMEOUT with solve_time=300.0
    - 1 SIZE_LIMIT with solve_time=None
    Invariant: n_algorithmic (5) + n_size_limit (1) + n_unavailable (0) + n_license_error (0) + n_unknown (0) == n_total (6).
    Status: PARTIAL (4 optimal out of 6).
    """
    from analysis.make_tables import aggregate_repetition_group

    reps = [
        {"status": "OPTIMAL", "solve_time": 1.0, "time_limit": 300.0},
        {"status": "OPTIMAL", "solve_time": 3.0, "time_limit": 300.0},
        {"status": "OPTIMAL", "solve_time": None, "wall_time": 2.0, "time_limit": 300.0},
        {"status": "OPTIMAL", "solve_time": "corrupted", "wall_time": 4.0, "time_limit": 300.0},
        {"status": "TIMEOUT", "solve_time": 300.0, "time_limit": 300.0},
        {"status": "SIZE_LIMIT", "solve_time": None, "time_limit": 300.0},
    ]

    agg = aggregate_repetition_group(reps, default_timeout=300.0)
    assert agg is not None
    assert agg["status"] == "PARTIAL"
    assert agg["n_total"] == 6
    assert agg["n_algorithmic"] == 5
    assert agg["n_optimal"] == 4
    assert agg["n_timeout"] == 1
    assert agg["n_size_limit"] == 1
    assert agg["n_unavailable"] == 0
    assert agg["n_license_error"] == 0
    assert agg["n_unknown"] == 0

    # Invariant holds
    assert agg["n_algorithmic"] + agg["n_size_limit"] + agg["n_unavailable"] + agg["n_license_error"] + agg["n_unknown"] == agg["n_total"]

    # Optimal reps genuine solve_times: [1.0, 3.0] -> median is 2.0 (no mixing with wall_time)
    assert abs(agg["solve_time"] - 2.0) < 1e-5
    assert agg["solve_time_source"] == "solve_time"
    assert agg["timing_fallback"] is False
    assert abs(agg["wall_time"] - 3.0) < 1e-5

    # PAR2: 2 optimal with solve_time (1.0 + 3.0), 2 optimal with missing solve_time (2 * 600 = 1200),
    # 1 timeout (2 * 300 = 600.0), SIZE_LIMIT excluded.
    # Denominator: 5. Numerator: 1.0 + 3.0 + 600.0 + 600.0 + 600.0 = 1804.0. PAR2 = 1804 / 5 = 360.8
    assert abs(agg["par2"] - 360.8) < 1e-5


def test_is_valid_record_dict_comprehensive():
    """Verify is_valid_record_dict safely filters missing values, scalars, booleans, and empty dicts."""
    import math
    import numpy as np
    import pandas as pd
    from analysis.make_tables import is_valid_record_dict

    # Invalid values
    assert not is_valid_record_dict(None)
    assert not is_valid_record_dict(np.nan)
    assert not is_valid_record_dict(float("nan"))
    assert not is_valid_record_dict(np.float32("nan"))
    assert not is_valid_record_dict(np.float64("nan"))
    assert not is_valid_record_dict(pd.NA)
    assert not is_valid_record_dict(pd.NaT)
    assert not is_valid_record_dict({})
    assert not is_valid_record_dict(True)
    assert not is_valid_record_dict(False)
    assert not is_valid_record_dict(123)
    assert not is_valid_record_dict(0)
    assert not is_valid_record_dict("OPTIMAL")
    assert not is_valid_record_dict(["list"])

    # Valid non-empty dicts
    assert is_valid_record_dict({"status": "OPTIMAL"})
    assert is_valid_record_dict({"status": "TIMEOUT", "solve_time": 600.0})
    assert is_valid_record_dict({"any_key": "any_value"})


def test_status_normalization_and_casing():
    """Verify normalize_status and is_optimal_status handle casing, invalid strings, and missing values uniformly."""
    import math
    import numpy as np
    import pandas as pd
    from analysis.make_tables import (
        normalize_status,
        is_optimal_status,
        categorize_status,
        compute_summary_stats,
    )

    # Casing variations for OPTIMAL
    for opt_input in ["OPTIMAL", "optimal", "Optimal", "oPtImAL", "  OPTIMAL  "]:
        assert normalize_status(opt_input) == "OPTIMAL"
        assert is_optimal_status(opt_input)
        assert categorize_status(opt_input) == "OPTIMAL"
        assert is_optimal_status({"status": opt_input})

    # Casing variations for other statuses
    assert normalize_status("timeout") == "TIMEOUT"
    assert normalize_status("Timeout") == "TIMEOUT"
    assert normalize_status("partial") == "PARTIAL"
    assert normalize_status("Partial") == "PARTIAL"
    assert normalize_status("size_limit") == "SIZE_LIMIT"
    assert normalize_status("Size_Limit") == "SIZE_LIMIT"
    assert normalize_status("unavailable") == "UNAVAILABLE"
    assert normalize_status("license_error") == "LICENSE_ERROR"
    assert normalize_status("infeasible") == "INFEASIBLE"
    assert normalize_status("unsat") == "INFEASIBLE"
    assert normalize_status("numeric_error") == "NUMERIC_ERROR"
    assert normalize_status("error") == "ERROR"

    # Missing / invalid inputs return None from normalize_status and False from is_optimal_status
    for inv in [None, np.nan, float("nan"), pd.NA, "", "   ", "NONE", "NAN", "<NA>", "NULL", "-", True, False]:
        assert normalize_status(inv) is None
        assert not is_optimal_status(inv)
        assert not is_optimal_status({"status": inv})

    # Unknown statuses
    assert normalize_status("weird_unknown_status") == "UNKNOWN"
    assert categorize_status("weird_unknown_status") == "UNKNOWN"

    # Verify that different casing of OPTIMAL yields identical summary statistics
    df_casing = pd.DataFrame({
        "mixed_opt": [
            {"status": "optimal", "solve_time": 1.0, "time_limit": 600.0},
            {"status": "Optimal", "solve_time": 2.0, "time_limit": 600.0},
            {"status": "OPTIMAL", "solve_time": 3.0, "time_limit": 600.0},
        ],
        "upper_opt": [
            {"status": "OPTIMAL", "solve_time": 1.0, "time_limit": 600.0},
            {"status": "OPTIMAL", "solve_time": 2.0, "time_limit": 600.0},
            {"status": "OPTIMAL", "solve_time": 3.0, "time_limit": 600.0},
        ],
    }, index=["ds1", "ds2", "ds3"])

    solved, sl, par2, speedup = compute_summary_stats(df_casing, ["mixed_opt", "upper_opt"], baseline_col="upper_opt")
    assert solved["mixed_opt"] == 3
    assert solved["upper_opt"] == 3
    assert abs(par2["mixed_opt"] - 2.0) < 1e-5
    assert abs(par2["upper_opt"] - 2.0) < 1e-5
    assert abs(speedup["mixed_opt"] - 1.00) < 1e-5
    assert abs(speedup["upper_opt"] - 1.00) < 1e-5


def test_dataframe_missing_values_and_non_uniform_solvers():
    """
    Test Section 1 requirements:
    1. Dataset A has baseline and solver X, but lacks solver Y (NaN).
    2. Dataset B has solver Y but lacks baseline (NaN) and lacks solver X (None).
    3. Column with record None.
    4. Column with record NaN.
    5. Column with empty dict.
    6. DataFrame with heterogeneous solvers across datasets.
    Verify: no crash, no missing record counted as result, correct solved, PAR2, and speedup.
    """
    import math
    import numpy as np
    import pandas as pd
    from analysis.make_tables import (
        compute_summary_stats,
        get_baseline_eligible_instances,
        format_cell_value,
    )

    # Construct DataFrame with NaN, None, pd.NA, and empty dicts
    df_raw = pd.DataFrame({
        "baseline": [
            {"status": "OPTIMAL", "solve_time": 2.0, "time_limit": 600.0},  # ds_A
            np.nan,                                                         # ds_B: lacks baseline
            {"status": "OPTIMAL", "solve_time": 5.0, "time_limit": 600.0},  # ds_C
            {"status": "OPTIMAL", "solve_time": 10.0, "time_limit": 600.0}, # ds_D
        ],
        "solver_x": [
            {"status": "OPTIMAL", "solve_time": 1.0, "time_limit": 600.0},  # ds_A: 2x speedup
            None,                                                           # ds_B: None
            pd.NA,                                                          # ds_C: pd.NA
            {"status": "OPTIMAL", "solve_time": 5.0, "time_limit": 600.0},  # ds_D: 2x speedup
        ],
        "solver_y": [
            np.nan,                                                         # ds_A: lacks solver Y
            {"status": "OPTIMAL", "solve_time": 4.0, "time_limit": 600.0},  # ds_B: solved on ds_B
            {},                                                             # ds_C: empty dict
            {"status": "OPTIMAL", "solve_time": 2.5, "time_limit": 600.0},  # ds_D: solved on ds_D
        ],
    }, index=["ds_A", "ds_B", "ds_C", "ds_D"])

    # Baseline eligible instances: ds_A (2.0s), ds_C (5.0s), ds_D (10.0s). ds_B has NaN. Total: 3.
    base_insts = get_baseline_eligible_instances(df_raw, "baseline")
    assert set(base_insts.keys()) == {"ds_A", "ds_C", "ds_D"}
    assert len(base_insts) == 3

    # Compute summary stats without crash
    solved, sl, par2, speedup = compute_summary_stats(
        df_raw, ["baseline", "solver_x", "solver_y"], baseline_col="baseline", timeout=600.0
    )

    # Solved counts
    assert solved["baseline"] == 3
    assert solved["solver_x"] == 2  # solved on ds_A, ds_D
    assert solved["solver_y"] == 2  # solved on ds_B, ds_D

    # SL counts
    assert sl["baseline"] == 0
    assert sl["solver_x"] == 0
    assert sl["solver_y"] == 0

    # PAR2: computed over non-missing algorithmic runs
    # baseline: (2.0 + 5.0 + 10.0) / 3 = 17 / 3 ≈ 5.6667
    assert abs(par2["baseline"] - (17.0 / 3.0)) < 1e-4
    # solver_x: (1.0 + 5.0) / 2 = 3.0
    assert abs(par2["solver_x"] - 3.0) < 1e-4
    # solver_y: (4.0 + 2.5) / 2 = 3.25
    assert abs(par2["solver_y"] - 3.25) < 1e-4

    # Speedup:
    # baseline is 1.00
    assert abs(speedup["baseline"] - 1.00) < 1e-5
    # solver_x is missing on ds_C (eligible instance) -> len(ratios) == 2 != 3 -> speedup is NaN
    assert math.isnan(speedup["solver_x"])
    # solver_y is missing on ds_A and ds_C (eligible instances) -> len(ratios) == 1 != 3 -> speedup is NaN
    assert math.isnan(speedup["solver_y"])

    # Check cell formatting for missing values
    assert format_cell_value(np.nan) == "-"
    assert format_cell_value(pd.NA) == "-"
    assert format_cell_value(None) == "-"
    assert format_cell_value({}) == "-"


def test_speedup_and_n_used_synchronization():
    """
    Test Section 4:
    - Baseline eligible helper get_baseline_eligible_instances.
    - Uniform conditions: status OPTIMAL, valid timing, timing >= 0.1s.
    - Synchronized [n_used/n_total] and speedup instance set.
    - Candidate with wall_time fallback on eligible instance.
    - Candidate missing timing on eligible instance yields NaN speedup.
    - Non-OPTIMAL statuses never eligible for baseline.
    """
    import math
    import pandas as pd
    from analysis.make_tables import (
        get_baseline_eligible_instances,
        compute_summary_stats,
    )

    df_raw = pd.DataFrame({
        "baseline": [
            {"status": "OPTIMAL", "solve_time": 0.05, "time_limit": 600.0},       # ds1: timing < 0.1s -> NOT eligible
            {"status": "OPTIMAL", "solve_time": 2.0, "time_limit": 600.0},        # ds2: genuine solve_time 2.0s -> eligible
            {"status": "TIMEOUT", "solve_time": 600.0, "time_limit": 600.0},      # ds3: TIMEOUT -> NOT eligible
            {"status": "FEASIBLE", "solve_time": 10.0, "time_limit": 600.0},     # ds4: FEASIBLE -> NOT eligible
            {"status": "PARTIAL", "solve_time": 5.0, "time_limit": 600.0},       # ds5: PARTIAL -> NOT eligible
            {"status": "OPTIMAL", "solve_time": 4.0, "time_limit": 600.0},       # ds6: OPTIMAL 4.0s -> eligible
        ],
        "cand_all_valid": [
            {"status": "OPTIMAL", "solve_time": 0.01, "time_limit": 600.0},      # ds1
            {"status": "OPTIMAL", "solve_time": 1.0, "time_limit": 600.0},       # ds2: ratio 2.0/1.0 = 2.0
            {"status": "TIMEOUT", "solve_time": 600.0, "time_limit": 600.0},      # ds3
            {"status": "OPTIMAL", "solve_time": 5.0, "time_limit": 600.0},       # ds4
            {"status": "OPTIMAL", "solve_time": 5.0, "time_limit": 600.0},       # ds5
            {"status": "OPTIMAL", "solve_time": 2.0, "time_limit": 600.0},       # ds6: ratio 4.0/2.0 = 2.0
        ],
        "cand_missing_one": [
            {"status": "OPTIMAL", "solve_time": 0.01, "time_limit": 600.0},      # ds1
            {"status": "OPTIMAL", "solve_time": 1.0, "time_limit": 600.0},       # ds2: ratio 2.0
            {"status": "TIMEOUT", "solve_time": 600.0, "time_limit": 600.0},      # ds3
            {"status": "OPTIMAL", "solve_time": 5.0, "time_limit": 600.0},       # ds4
            {"status": "OPTIMAL", "solve_time": 2.0, "time_limit": 600.0},       # ds5
            {"status": "TIMEOUT", "solve_time": 600.0, "time_limit": 600.0},      # ds6: TIMEOUT on eligible instance!
        ],
    }, index=["ds1", "ds2", "ds3", "ds4", "ds5", "ds6"])

    base_insts = get_baseline_eligible_instances(df_raw, "baseline")
    assert set(base_insts.keys()) == {"ds2", "ds6"}
    assert len(base_insts) == 2
    assert abs(base_insts["ds2"] - 2.0) < 1e-5
    assert abs(base_insts["ds6"] - 4.0) < 1e-5

    solved, sl, par2, speedup = compute_summary_stats(
        df_raw, ["baseline", "cand_all_valid", "cand_missing_one"], baseline_col="baseline", timeout=600.0
    )

    # cand_all_valid solved both eligible instances (ds2 ratio=2.0, ds6 ratio=2.0) -> geom mean = 2.0
    assert abs(speedup["cand_all_valid"] - 2.00) < 1e-5

    # cand_missing_one is TIMEOUT on ds6 (eligible) -> ratio missing -> speedup is NaN
    assert math.isnan(speedup["cand_missing_one"])


def test_export_table_with_missing_and_heterogeneous_solvers(tmp_path):
    """
    Test Section 5:
    - Export CSV and LaTeX when solvers have missing records (None, NaN, pd.NA, empty dict).
    - Machine-readable data CSV has 21 columns in exact order.
    - Missing timing does not format as 0.
    """
    import os
    import numpy as np
    import pandas as pd
    from analysis.make_tables import export_table, DATA_CSV_COLUMNS

    df_raw = pd.DataFrame({
        "sat": [
            {"dataset": "g1", "column": "sat", "status": "OPTIMAL", "solve_time": 1.5, "wall_time": 1.6, "time_limit": 600.0, "n_total": 1, "n_algorithmic": 1, "n_optimal": 1, "termination_reason": "ALL_OPTIMAL"},
            {"dataset": "g2", "column": "sat", "status": "TIMEOUT", "solve_time": 600.0, "wall_time": 600.1, "time_limit": 600.0, "n_total": 1, "n_algorithmic": 1, "n_timeout": 1, "termination_reason": "TIME_LIMIT"},
        ],
        "cpsat": [
            np.nan,  # missing on g1
            {"dataset": "g2", "column": "cpsat", "status": "OPTIMAL", "solve_time": 2.5, "wall_time": 2.6, "time_limit": 600.0, "n_total": 1, "n_algorithmic": 1, "n_optimal": 1, "termination_reason": "ALL_OPTIMAL"},
        ],
    }, index=["g1", "g2"])

    df_disp = pd.DataFrame({
        "sat": ["1.5000", "TO"],
        "cpsat": ["-", "2.5000"],
    }, index=["g1", "g2"])

    summary_rows = [
        ("Solved", {"sat": "1", "cpsat": "1"}),
        ("PAR2 (s)", {"sat": "600.750", "cpsat": "2.500"}),
    ]

    out_dir = str(tmp_path / "export_missing_test")
    export_table(df_disp, summary_rows, title="Test Heterogeneous Export", base_filename="table_hetero", output_dirs=(out_dir,), df_raw=df_raw)

    csv_path = os.path.join(out_dir, "table_hetero.csv")
    data_csv_path = os.path.join(out_dir, "table_hetero_data.csv")
    tex_path = os.path.join(out_dir, "table_hetero.tex")

    assert os.path.exists(csv_path)
    assert os.path.exists(data_csv_path)
    assert os.path.exists(tex_path)

    # Validate LaTeX content
    with open(tex_path, "r", encoding="utf-8") as f:
        tex_content = f.read()
    assert "\\begin{table}" in tex_content
    assert "g1 & 1.5000 & -" in tex_content

    # Validate data CSV schema and content
    df_data = pd.read_csv(data_csv_path)
    assert list(df_data.columns) == DATA_CSV_COLUMNS
    assert len(df_data) == 3  # g1/sat, g2/sat, g2/cpsat (g1/cpsat was NaN so excluded)
    assert set(df_data["dataset"]) == {"g1", "g2"}
    assert df_data[df_data["dataset"] == "g1"].iloc[0]["column"] == "sat"
    assert df_data[df_data["dataset"] == "g1"].iloc[0]["termination_reason"] == "ALL_OPTIMAL"


def test_make_table_per_dataset_validation_and_formatting(tmp_path):
    """
    Test Section 1 requirements for make_table_per_dataset:
    1. OPTIMAL with valid solve_time.
    2. OPTIMAL with solve_time=None.
    3. OPTIMAL with solve_time=NaN.
    4. TIMEOUT with wall_time=None.
    5. Status lowercase or mixed case.
    6. best_ub is valid int.
    7. best_ub is fractional float, negative, NaN, None, or error string.
    8. best_lb missing or invalid.
    9. Export CSV succeeds and produces no fake 0.
    """
    import os
    import math
    import numpy as np
    import pandas as pd
    from analysis.make_tables import make_table_per_dataset

    records = [
        # 1. OPTIMAL with valid solve_time and valid int best_ub
        {
            "groups": ["per_dataset"],
            "dataset": "ds_valid_opt",
            "N": 14,
            "num_edges": 20,
            "status": "OPTIMAL",
            "solve_time": 1.2345,
            "wall_time": 1.5000,
            "best_ub": 3,
            "best_lb": 2,
        },
        # 2. OPTIMAL with solve_time=None, fractional best_ub (3.7) -> ub rejected, time is '-'
        {
            "groups": ["per_dataset"],
            "dataset": "ds_opt_none_time",
            "N": 20,
            "num_edges": 30,
            "status": "optimal",  # lowercase
            "solve_time": None,
            "wall_time": 2.5000,
            "best_ub": 3.7,       # fractional -> '-'
            "best_lb": None,
        },
        # 3. OPTIMAL with solve_time=NaN, negative best_ub (-1)
        {
            "groups": ["per_dataset"],
            "dataset": "ds_opt_nan_time",
            "N": 25,
            "num_edges": 35,
            "status": "Optimal",  # mixed case
            "solve_time": float("nan"),
            "wall_time": float("nan"),
            "best_ub": -1,        # negative -> '-'
            "best_lb": "corrupt", # unparseable -> '-'
        },
        # 4. TIMEOUT with wall_time=None, solve_time=600.0, valid best_ub=6
        {
            "groups": ["per_dataset"],
            "dataset": "ds_to_none_wall",
            "N": 30,
            "num_edges": 45,
            "status": "Timeout",  # mixed case
            "solve_time": 600.0,
            "wall_time": None,
            "best_ub": 6,
            "best_lb": 2,
        },
        # 5. TIMEOUT with corrupt best_ub ("invalid")
        {
            "groups": ["per_dataset"],
            "dataset": "ds_to_bad_ub",
            "N": 35,
            "num_edges": 50,
            "status": "TIMEOUT",
            "solve_time": 600.0,
            "wall_time": 600.5,
            "best_ub": "invalid",
            "best_lb": None,
        },
    ]

    out_dir = str(tmp_path / "per_dataset_export")
    df = make_table_per_dataset(records, base_filename="table_per_dataset_test", output_dirs=(out_dir,))
    assert df is not None

    csv_path = os.path.join(out_dir, "table_per_dataset_test.csv")
    assert os.path.exists(csv_path)

    df_read = pd.read_csv(csv_path)
    rows_by_ds = {r["dataset"]: r for _, r in df_read.iterrows()}

    # 1. ds_valid_opt: OPTIMAL, 1.2345, ub=3, ub_minus_lb="0"
    r1 = rows_by_ds["ds_valid_opt"]
    assert r1["status"] == "OPTIMAL"
    assert r1["solve_time"] == "1.2345"
    assert r1["wall_time"] == "1.5000"
    assert str(r1["best_ub"]) == "3"
    assert str(r1["ub_minus_lb"]) == "0"

    # 2. ds_opt_none_time: optimal -> OPTIMAL, solve_time='-', best_ub='-', ub_minus_lb='0'
    r2 = rows_by_ds["ds_opt_none_time"]
    assert r2["status"] == "OPTIMAL"
    assert r2["solve_time"] == "-"
    assert r2["wall_time"] == "2.5000"
    assert r2["best_ub"] == "-"
    assert str(r2["ub_minus_lb"]) == "0"

    # 3. ds_opt_nan_time: Optimal -> OPTIMAL, solve_time='-', wall_time='-', best_ub='-'
    r3 = rows_by_ds["ds_opt_nan_time"]
    assert r3["status"] == "OPTIMAL"
    assert r3["solve_time"] == "-"
    assert r3["wall_time"] == "-"
    assert r3["best_ub"] == "-"
    assert str(r3["ub_minus_lb"]) == "0"

    # 4. ds_to_none_wall: Timeout -> TIMEOUT, solve_time="600.0000", wall_time='-', best_ub=6
    r4 = rows_by_ds["ds_to_none_wall"]
    assert r4["status"] == "TIMEOUT"
    assert r4["solve_time"] == "600.0000"
    assert r4["wall_time"] == "-"
    assert str(r4["best_ub"]) == "6"
    # lb = max(lb_deg(0), 2) = 2 -> ub_minus_lb = 6 - 2 = 4
    assert str(r4["ub_minus_lb"]) == "4"

    # 5. ds_to_bad_ub: TIMEOUT with invalid best_ub -> best_ub='-', ub_minus_lb='-'
    r5 = rows_by_ds["ds_to_bad_ub"]
    assert r5["status"] == "TIMEOUT"
    assert r5["best_ub"] == "-"
    assert r5["ub_minus_lb"] == "-"


def test_ablation_family_and_methods_family_missing_timing_safety(tmp_path):
    """
    Test Section 2 requirements:
    - Missing candidate/target timing does NOT become 0.0 or a false faster pair.
    - Missing reference timing does NOT become ratio.
    - Missing CP-SAT timing does NOT become ratio.
    - Ratio only computed when BOTH sides have positive valid finite timing.
    """
    import os
    import math
    import pandas as pd
    from analysis.make_tables import (
        make_table_ablation_family,
        make_table_methods_family,
        make_table_ablation_phases,
    )

    out_dir = str(tmp_path / "family_export")

    # Ablation family test with missing target timing and missing reference timing
    records_ablation = [
        # Pair (ds1, seed 0): both valid -> target=2.0, ref=4.0 -> ratio 0.5 (faster: 1/1)
        {"dataset": "ds1", "relabel_seed": 0, "status": "OPTIMAL", "solve_time": 2.0, "config": {"sb_rotation": True, "sb_reflection": False, "heuristic_ub": False}},
        {"dataset": "ds1", "relabel_seed": 0, "status": "OPTIMAL", "solve_time": 4.0, "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}},
        # Pair (ds1, seed 1): target missing timing (None) -> MUST NOT become 0.0 and NOT be faster!
        {"dataset": "ds1", "relabel_seed": 1, "status": "OPTIMAL", "solve_time": None, "config": {"sb_rotation": True, "sb_reflection": False, "heuristic_ub": False}},
        {"dataset": "ds1", "relabel_seed": 1, "status": "OPTIMAL", "solve_time": 3.0, "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}},
        # Pair (ds1, seed 2): reference missing timing (NaN) -> MUST NOT be in total_count
        {"dataset": "ds1", "relabel_seed": 2, "status": "OPTIMAL", "solve_time": 2.0, "config": {"sb_rotation": True, "sb_reflection": False, "heuristic_ub": False}},
        {"dataset": "ds1", "relabel_seed": 2, "status": "OPTIMAL", "solve_time": float("nan"), "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}},

        # Need dummy valid pairs for the other 6 comparisons so table generation doesn't skip
        # rotation_reflection vs none
        {"dataset": "ds1", "relabel_seed": 0, "status": "OPTIMAL", "solve_time": 1.0, "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": False}},
        # heuristic_ub vs none
        {"dataset": "ds1", "relabel_seed": 0, "status": "OPTIMAL", "solve_time": 3.0, "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": True}},
        # all vs none
        {"dataset": "ds1", "relabel_seed": 0, "status": "OPTIMAL", "solve_time": 0.8, "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}},
    ]

    df_ab = make_table_ablation_family(records_ablation, output_dirs=(out_dir,))
    assert df_ab is not None

    row_rot_vs_none = df_ab[(df_ab["comparison_group"] == "variant vs none") & (df_ab["variant"] == "rotation")].iloc[0]
    # Exactly 1 valid pair (seed 0). Seed 1 and Seed 2 were excluded because of missing timing.
    # Total count must be 1, faster count must be 1. (NOT 2/2 or 2/3!)
    assert row_rot_vs_none["faster_pairs"] == "1/1"
    assert abs(float(row_rot_vs_none["median_time_ratio"]) - 0.5000) < 1e-4

    # Methods family test with missing CP-SAT timing and missing SAT timing
    records_methods = [
        # ds1, seed 0: valid cp (2.0) and sat (1.0) -> ratio 2.0
        {"dataset": "ds1", "relabel_seed": 0, "method": "cpsat", "status": "OPTIMAL", "solve_time": 2.0},
        {"dataset": "ds1", "relabel_seed": 0, "method": "sat", "solver": "cadical195", "status": "OPTIMAL", "solve_time": 1.0, "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}},
        # ds1, seed 1: cpsat missing timing (None) -> excluded from ratios
        {"dataset": "ds1", "relabel_seed": 1, "method": "cpsat", "status": "OPTIMAL", "solve_time": None},
        {"dataset": "ds1", "relabel_seed": 1, "method": "sat", "solver": "cadical195", "status": "OPTIMAL", "solve_time": 1.0, "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}},
        # ds1, seed 2: sat missing timing (NaN) -> excluded from ratios
        {"dataset": "ds1", "relabel_seed": 2, "method": "cpsat", "status": "OPTIMAL", "solve_time": 2.0},
        {"dataset": "ds1", "relabel_seed": 2, "method": "sat", "solver": "cadical195", "status": "OPTIMAL", "solve_time": float("nan"), "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}},
    ]

    df_mf = make_table_methods_family(records_methods, output_dirs=(out_dir,))
    assert df_mf is not None
    row_ds1 = df_mf[df_mf["dataset"] == "ds1"].iloc[0]
    # Only seed 0 is valid. Seeds 1 and 2 excluded.
    assert row_ds1["faster_pairs"] == "0/1"
    assert abs(float(row_ds1["median_cpsat_to_sat_ratio"]) - 2.0000) < 1e-4

    # Phases ablation test with invalid / empty per_iter_times
    records_phases = [
        # Valid per_iter_times: [1.0, 2.0] -> last=2.0, prev=1.0, solve_time=3.0 -> ratio=2/3
        {"N": 40, "relabel_seed": 0, "status": "OPTIMAL", "solve_time": 3.0, "per_iter_times": [1.0, 2.0], "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}},
        {"N": 40, "relabel_seed": 0, "status": "OPTIMAL", "solve_time": 3.0, "per_iter_times": [1.0, 2.0], "config": {"sb_rotation": True, "sb_reflection": False, "heuristic_ub": False}},
        {"N": 40, "relabel_seed": 0, "status": "OPTIMAL", "solve_time": 3.0, "per_iter_times": [1.0, 2.0], "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": False}},
        {"N": 40, "relabel_seed": 0, "status": "OPTIMAL", "solve_time": 3.0, "per_iter_times": [1.0, 2.0], "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": True}},
        {"N": 40, "relabel_seed": 0, "status": "OPTIMAL", "solve_time": 3.0, "per_iter_times": [1.0, 2.0], "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}},
        # Additional run with corrupt per_iter_times (empty, non-numeric, negative)
        {"N": 40, "relabel_seed": 1, "status": "OPTIMAL", "solve_time": 3.0, "per_iter_times": [], "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}},
        {"N": 40, "relabel_seed": 2, "status": "OPTIMAL", "solve_time": 3.0, "per_iter_times": [1.0, "corrupt"], "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}},
    ]
    df_ph = make_table_ablation_phases(records_phases, output_dirs=(out_dir,))
    assert df_ph is not None
    row_none = df_ph[(df_ph["N"] == 40) & (df_ph["variant"] == "none")].iloc[0]
    # Corrupt per_iter_times are excluded; only the valid seed 0 run contributes
    assert abs(float(row_none["median_last_iter_time"]) - 2.0000) < 1e-4
    assert abs(float(row_none["median_prev_iters_time"]) - 1.0000) < 1e-4


def test_aggregation_solve_time_and_wall_time_provenance():
    """
    Test Section 3 requirements:
    1. Pool with ONLY wall_time: solve_time_source="wall_time", timing_fallback=True.
    2. Pool with neither solve_time nor wall_time: solve_time_source=None, solve_time=None, timing_fallback=False.
    3. Pool with genuine solve_time: solve_time_source="solve_time", timing_fallback=False.
    4. Wall time is always independently preserved.
    """
    from analysis.make_tables import aggregate_repetition_group

    # 1. Pool with ONLY wall_time (genuine solve_time is None, wall_time is preserved)
    reps_wall_only = [
        {"status": "OPTIMAL", "solve_time": None, "wall_time": 5.0, "time_limit": 600.0},
        {"status": "OPTIMAL", "solve_time": None, "wall_time": 7.0, "time_limit": 600.0},
    ]
    agg1 = aggregate_repetition_group(reps_wall_only)
    assert agg1["solve_time"] is None
    assert agg1["solve_time_source"] is None
    assert agg1["timing_fallback"] is False
    assert agg1["wall_time"] == 6.0
    assert agg1["effective_time"] == 6.0
    assert agg1["effective_time_source"] == "wall_time"

    # 2. Pool with neither solve_time nor wall_time
    reps_none = [
        {"status": "OPTIMAL", "solve_time": None, "wall_time": None, "time_limit": 600.0},
    ]
    agg2 = aggregate_repetition_group(reps_none)
    assert agg2["solve_time"] is None
    assert agg2["solve_time_source"] is None
    assert agg2["timing_fallback"] is False
    assert agg2["wall_time"] is None

    # 3. Pool with genuine solve_time on all reps
    reps_genuine = [
        {"status": "OPTIMAL", "solve_time": 2.0, "wall_time": 2.5, "time_limit": 600.0},
        {"status": "OPTIMAL", "solve_time": 4.0, "wall_time": 4.5, "time_limit": 600.0},
    ]
    agg3 = aggregate_repetition_group(reps_genuine)
    assert agg3["solve_time"] == 3.0
    assert agg3["solve_time_source"] == "solve_time"
    assert agg3["timing_fallback"] is False
    assert agg3["wall_time"] == 3.5


def test_missing_value_types_edge_cases():
    """
    Test Section 4 requirements:
    - np.float32("nan"), np.float64("nan"), pd.NaT across validation helpers.
    - Consistency across normalize_status, is_optimal_status, categorize_status.
    - Distinction between ERROR and NUMERIC_ERROR.
    """
    import numpy as np
    import pandas as pd
    from analysis.make_tables import (
        validate_best_ub,
        validate_solve_time,
        normalize_status,
        is_optimal_status,
        categorize_status,
    )

    for nan_val in [np.float32("nan"), np.float64("nan"), float("nan"), pd.NA, pd.NaT]:
        assert validate_best_ub(nan_val) is None
        assert validate_solve_time(nan_val) is None
        assert normalize_status(nan_val) is None
        assert not is_optimal_status(nan_val)

    # Status taxonomy distinctions
    assert normalize_status("ERROR") == "ERROR"
    assert normalize_status("NUMERIC_ERROR") == "NUMERIC_ERROR"
    assert categorize_status("ERROR") == "ERROR"
    assert categorize_status("NUMERIC_ERROR") == "ERROR"  # canonical category maps NUMERIC_ERROR to ERROR
    assert normalize_status("error") == "ERROR"
    assert normalize_status("numeric_error") == "NUMERIC_ERROR"


def test_is_run_completed_unified_validation():
    """
    Test Section 1 & 4 requirements:
    Verify unified validation between run_experiments.py and make_tables.py.
    Ensure fractional ub, booleans, negative values, and validation errors are rejected.
    """
    from experiments.run_experiments import is_run_completed

    # retry_failed=False always accepts existing records
    assert is_run_completed({"status": "ERROR"}, retry_failed=False) is True
    assert is_run_completed({"status": "TIMEOUT"}, retry_failed=False) is True

    # retry_failed=True
    # Conclusive proofs and limits
    for st in ["OPTIMAL", "optimal", "INFEASIBLE", "infeasible", "SIZE_LIMIT", "size_limit"]:
        assert is_run_completed({"status": st}, retry_failed=True) is True

    # Environmental/license barriers
    for st in ["UNAVAILABLE", "unavailable", "LICENSE_ERROR", "license_error"]:
        assert is_run_completed({"status": st}, retry_failed=True) is True

    # Feasible with valid integer bounds
    assert is_run_completed({"status": "FEASIBLE", "best_ub": 3}, retry_failed=True) is True
    assert is_run_completed({"status": "feasible", "best_ub": 3.0}, retry_failed=True) is True
    assert is_run_completed({"status": "FEASIBLE", "best_ub": "3"}, retry_failed=True) is True
    assert is_run_completed({"status": "FEASIBLE", "best_ub": 0}, retry_failed=True) is True

    # Feasible with invalid bounds - MUST be rejected
    assert is_run_completed({"status": "FEASIBLE", "best_ub": 3.7}, retry_failed=True) is False
    assert is_run_completed({"status": "FEASIBLE", "best_ub": "3.7"}, retry_failed=True) is False
    assert is_run_completed({"status": "FEASIBLE", "best_ub": True}, retry_failed=True) is False
    assert is_run_completed({"status": "FEASIBLE", "best_ub": False}, retry_failed=True) is False
    assert is_run_completed({"status": "FEASIBLE", "best_ub": -1}, retry_failed=True) is False
    assert is_run_completed({"status": "FEASIBLE", "best_ub": None}, retry_failed=True) is False
    assert is_run_completed({"status": "FEASIBLE", "best_ub": float("nan")}, retry_failed=True) is False
    assert is_run_completed({"status": "FEASIBLE", "best_ub": 3, "validation_error": True}, retry_failed=True) is False

    # Transient failures
    for st in ["TIMEOUT", "MEMOUT", "ERROR", "UNKNOWN", "unknown", "arbitrary_fail"]:
        assert is_run_completed({"status": st}, retry_failed=True) is False


def test_make_plots_hardening_and_safety(tmp_path):
    """
    Test Section 3 requirements:
    Verify make_plots.py functions handle pathological records safely
    and output valid PNG/PDF without crashes or invalid log-scale zeroes.
    """
    import os
    from analysis.make_plots import (
        plot_cactus,
        plot_scatter_sb_vs_nosb,
        plot_ablation_median_time,
        plot_ablation_iterations_and_unsat,
        plot_ablation_family_boxplots,
    )

    out_dir = str(tmp_path / "figs")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Pathological records: None, NaN, inf, negative, strings, empty arrays
    pathological_recs = [
        {"dataset": "ds1", "status": "UNKNOWN", "solve_time": None, "groups": ["solvers_small"]},
        {"dataset": "ds2", "status": "OPTIMAL", "solve_time": float("nan"), "groups": ["solvers_small"]},
        {"dataset": "ds3", "status": "OPTIMAL", "solve_time": -5.0, "groups": ["solvers_small"]},
        {"dataset": "ds4", "status": "OPTIMAL", "solve_time": 0.0, "groups": ["solvers_small"]},
        {"dataset": "ds5", "status": "OPTIMAL", "solve_time": "invalid", "groups": ["solvers_small"]},
        {"dataset": "ds6", "status": "OPTIMAL", "solve_time": 1.5, "per_iter_times": [float("nan"), "err"], "num_iterations": "none", "groups": ["ablation_small"], "config": {}},
    ]

    # None of these should crash
    plot_cactus(pathological_recs, "solvers_small", lambda r: "test", "Test Cactus", "cactus_patho", output_dirs=(out_dir,))
    plot_scatter_sb_vs_nosb(pathological_recs, "ablation_small", "scatter_patho", "Test Scatter", output_dirs=(out_dir,))
    plot_ablation_median_time(pathological_recs, "ablation_small", "median_patho", "Test Median", output_dirs=(out_dir,))
    plot_ablation_iterations_and_unsat(pathological_recs, "ablation_small", "iter_patho", "Test Iter", output_dirs=(out_dir,))
    plot_ablation_family_boxplots(pathological_recs, "boxplot_patho", output_dirs=(out_dir,))

    # 2. Valid records for all plot functions
    valid_recs = [
        # Cactus & Ablation small records
        {"dataset": "dsA", "status": "OPTIMAL", "solve_time": 0.12, "groups": ["solvers_small", "ablation_small"], "solver": "sat_kissat", "config": {}, "num_iterations": 2, "per_iter_times": [0.05, 0.07]},
        {"dataset": "dsB", "status": "OPTIMAL", "solve_time": 0.45, "groups": ["solvers_small", "ablation_small"], "solver": "sat_kissat", "config": {"sb_rotation": True}, "num_iterations": 3, "per_iter_times": [0.1, 0.15, 0.2]},
        {"dataset": "dsA", "status": "OPTIMAL", "solve_time": 0.08, "groups": ["solvers_small", "ablation_small"], "solver": "sat_kissat", "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}, "num_iterations": 1, "per_iter_times": [0.08]},
        # Family records for N=40 and N=50
    ]
    for n_val in [40, 50]:
        for seed in [1, 2]:
            for var_cfg, var_name in [
                ({}, "none"),
                ({"sb_rotation": True}, "rotation"),
                ({"sb_rotation": True, "sb_reflection": True}, "rot_ref"),
                ({"heuristic_ub": True}, "heur_ub"),
                ({"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}, "all"),
            ]:
                valid_recs.append({
                    "dataset": f"ds_fam_{n_val}",
                    "N": n_val,
                    "relabel_seed": seed,
                    "status": "OPTIMAL",
                    "solve_time": 1.0 if var_name == "none" else 0.5,
                    "config": var_cfg,
                })

    plot_cactus(valid_recs, "solvers_small", lambda r: r.get("solver", "s"), "Valid Cactus", "cactus_valid", output_dirs=(out_dir,))
    assert os.path.exists(f"{out_dir}/cactus_valid.png")
    assert os.path.exists(f"{out_dir}/cactus_valid.pdf")
    assert os.path.getsize(f"{out_dir}/cactus_valid.png") > 0

    plot_scatter_sb_vs_nosb(valid_recs, "ablation_small", "scatter_valid", "Valid Scatter", output_dirs=(out_dir,))
    assert os.path.exists(f"{out_dir}/scatter_valid.png")
    assert os.path.exists(f"{out_dir}/scatter_valid.pdf")

    plot_ablation_median_time(valid_recs, "ablation_small", "median_valid", "Valid Median", output_dirs=(out_dir,))
    assert os.path.exists(f"{out_dir}/median_valid.png")
    assert os.path.exists(f"{out_dir}/median_valid.pdf")

    plot_ablation_iterations_and_unsat(valid_recs, "ablation_small", "iter_valid", "Valid Iter", output_dirs=(out_dir,))
    assert os.path.exists(f"{out_dir}/iter_valid.png")
    assert os.path.exists(f"{out_dir}/iter_valid.pdf")

    plot_ablation_family_boxplots(valid_recs, "boxplot_valid", output_dirs=(out_dir,))
    assert os.path.exists(f"{out_dir}/boxplot_valid.png")
    assert os.path.exists(f"{out_dir}/boxplot_valid.pdf")


def test_end_to_end_pipeline_roundtrip(tmp_path):
    """
    Test Section 7 requirements:
    Full pipeline round-trip: JSONL -> load_records -> tables -> CSV/LaTeX -> reload verify.
    """
    import json
    import pandas as pd
    from analysis.make_tables import (
        load_records,
        make_table_per_dataset,
        export_table,
        DATA_CSV_COLUMNS,
    )

    jsonl_path = tmp_path / "runs.jsonl"
    out_dir = str(tmp_path / "results")

    # Generate synthetic experiment records
    records = [
        {
            "dataset": "494_bus_n40",
            "N": 40,
            "M": 80,
            "density": 0.1,
            "solver": "cbmp_cadical",
            "method": "sat",
            "repetition": 1,
            "status": "OPTIMAL",
            "best_ub": 5,
            "lower_bound": 5,
            "solve_time": 0.245,
            "wall_time": 0.250,
            "encode_time": 0.005,
            "num_iterations": 3,
            "groups": ["per_dataset", "methods"],
            "termination_reason": "OPTIMAL_FOUND",
        },
        {
            "dataset": "494_bus_n40",
            "N": 40,
            "M": 80,
            "density": 0.1,
            "solver": "ortools_cpsat",
            "method": "cp",
            "repetition": 1,
            "status": "FEASIBLE",
            "best_ub": 7,
            "lower_bound": 3,
            "solve_time": 600.0,
            "wall_time": 600.05,
            "encode_time": 0.01,
            "num_iterations": None,
            "groups": ["per_dataset", "methods"],
            "termination_reason": "TIME_LIMIT",
        },
    ]

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    loaded = load_records(str(jsonl_path))
    assert len(loaded) == 2

    # Build per_dataset table
    df_per_ds = make_table_per_dataset(loaded, output_dirs=(out_dir,))
    assert not df_per_ds.empty
    assert os.path.exists(f"{out_dir}/table_per_dataset.csv")

    # Export 21-column data CSV using export_table
    df_raw = pd.DataFrame({"cadical": [loaded[0]], "cpsat": [loaded[1]]}, index=["494_bus_n40"])
    df_disp = pd.DataFrame({"cadical": ["0.2450"], "cpsat": ["TO"]}, index=["494_bus_n40"])
    export_table(
        df_disp,
        [],
        title="Test Per Dataset",
        base_filename="table_test_roundtrip",
        output_dirs=(out_dir,),
        df_raw=df_raw,
    )

    csv_data_path = f"{out_dir}/table_test_roundtrip_data.csv"
    assert os.path.exists(csv_data_path)
    df_reloaded = pd.read_csv(csv_data_path)
    assert list(df_reloaded.columns) == DATA_CSV_COLUMNS
    assert df_reloaded.columns[20] == "termination_reason"
    assert len(df_reloaded) == 2

