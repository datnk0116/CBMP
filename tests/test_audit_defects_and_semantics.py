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




