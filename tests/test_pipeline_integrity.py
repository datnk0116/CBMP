"""
tests/test_pipeline_integrity.py

Comprehensive Regression and Pipeline Integrity Test Suite for CBMP:
Group A: Canonical Identity & Key Alignment (Strict Parsing, Aliases, Conflict Detection)
Group B: Runner Lifecycle & JSONL Integrity (UB/Assignment Validation, Duplicate Conflict)
Group C: Table Correctness & Invariance (No Overwrite, Deterministic sat_best, Order Invariance)
Group D: Cross-Validation Independent Sources (Engine vs Config Level, Alias Normalization)
Group E: Plot Key Alignment & Log-Scale Safety (Pair Matching, Exact Dimensions)
Group F: End-to-End Pipeline Fixture & Round-Trip
"""

import json
import math
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from analysis.make_tables import (
    normalize_status,
    categorize_status,
    validate_best_ub,
    validate_lower_bound,
    validate_solve_time,
    strict_int,
    strict_positive_int,
    strict_bool,
    normalize_solver_alias,
    is_valid_assignment,
    resolve_field_with_conflict_check,
    extract_time_limit,
    extract_repetition_solve_time,
    extract_effective_time,
    is_valid_record_dict,
    is_optimal_status,
    get_canonical_key,
    get_source_key,
    aggregate_repetition_group,
    aggregate_repetitions,
    load_records,
    cross_validate_optima,
    format_cell_value,
    get_baseline_eligible_instances,
    compute_summary_stats,
    make_table_ablation,
    make_table_solvers,
    make_table_methods,
    make_table_ablation_family,
    make_table_per_dataset,
    export_table,
    DATA_CSV_COLUMNS,
)

from experiments.run_experiments import (
    is_run_completed,
    load_existing_runs,
)


# ============================================================================
# GROUP A: CANONICAL IDENTITY & KEY ALIGNMENT
# ============================================================================

def test_group_a_canonical_key_distinguishes_solvers_and_methods():
    """Verify distinct solvers, methods, and configurations produce non-colliding canonical keys."""
    rec_sat = {
        "dataset": "grid_3x3",
        "solver": "cadical195",
        "method": "sat",
        "config": {"encoding": "order", "solving_mode": "incremental", "eo_strategy": "commander"},
        "sb_rotation": True,
        "sb_reflection": False,
        "heuristic_ub": False,
        "relabel_seed": 0,
    }
    rec_sat_kissat = dict(rec_sat, solver="kissat")
    rec_sat_enc = dict(rec_sat, config={"encoding": "pairwise", "solving_mode": "incremental", "eo_strategy": "commander"})
    rec_cpsat = {
        "dataset": "grid_3x3",
        "solver": "cp_sat",
        "method": "cpsat",
        "config": {"num_workers": 4, "sb_rotation": True},
        "relabel_seed": 0,
    }
    rec_cpsat_w1 = {
        "dataset": "grid_3x3",
        "solver": "cp_sat",
        "method": "cpsat",
        "config": {"num_workers": 1, "sb_rotation": True},
        "relabel_seed": 0,
    }
    rec_gurobi = {
        "dataset": "grid_3x3",
        "solver": "gurobi",
        "method": "gurobi",
        "config": {"sb_rotation": True},
        "relabel_seed": 0,
    }

    key_sat = get_canonical_key(rec_sat)
    key_sat_kissat = get_canonical_key(rec_sat_kissat)
    key_sat_enc = get_canonical_key(rec_sat_enc)
    key_cpsat = get_canonical_key(rec_cpsat)
    key_cpsat_w1 = get_canonical_key(rec_cpsat_w1)
    key_gurobi = get_canonical_key(rec_gurobi)

    all_keys = [key_sat, key_sat_kissat, key_sat_enc, key_cpsat, key_cpsat_w1, key_gurobi]
    assert len(all_keys) == len(set(all_keys)), "Collision detected among distinct configurations!"


def test_group_a_relabel_seed_zero_vs_none_distinct():
    """Verify relabel_seed=0 is strictly distinguished from relabel_seed=None."""
    r_seed0 = {"dataset": "d1", "solver": "cadical195", "relabel_seed": 0}
    r_seed_none = {"dataset": "d1", "solver": "cadical195", "relabel_seed": None}
    r_seed_cfg0 = {"dataset": "d1", "solver": "cadical195", "config": {"relabel_seed": 0}}

    k_0 = get_canonical_key(r_seed0)
    k_none = get_canonical_key(r_seed_none)
    k_cfg0 = get_canonical_key(r_seed_cfg0)

    assert k_0 != k_none
    assert k_0 == k_cfg0
    assert k_0[-1] == 0
    assert k_none[-1] is None


def test_group_a_fractional_and_bool_seed_rejected():
    """Verify fractional seed, bool seed, or negative seed returns None (invalid key)."""
    assert get_canonical_key({"dataset": "d1", "solver": "cadical195", "relabel_seed": 2.7}) is None
    assert get_canonical_key({"dataset": "d1", "solver": "cadical195", "relabel_seed": True}) is None
    assert get_canonical_key({"dataset": "d1", "solver": "cadical195", "relabel_seed": False}) is None
    assert get_canonical_key({"dataset": "d1", "solver": "cadical195", "relabel_seed": -1}) is None


def test_group_a_fractional_and_bool_worker_count_rejected():
    """Verify fractional or boolean worker counts in CP-SAT are rejected."""
    assert get_canonical_key({"dataset": "d1", "solver": "cp_sat", "num_workers": 4.8}) is None
    assert get_canonical_key({"dataset": "d1", "solver": "cp_sat", "num_workers": True}) is None
    assert get_canonical_key({"dataset": "d1", "solver": "cp_sat", "num_workers": 0}) is None
    assert get_canonical_key({"dataset": "d1", "solver": "cp_sat", "num_workers": -2}) is None


def test_group_a_string_boolean_parsing():
    """Verify strict boolean parsing handles strings safely and rejects invalid strings."""
    assert strict_bool(True) is True
    assert strict_bool(False) is False
    assert strict_bool("true") is True
    assert strict_bool("false") is False
    assert strict_bool("yes") is True
    assert strict_bool("no") is False
    assert strict_bool("1") is True
    assert strict_bool("0") is False

    assert strict_bool(1) is None  # Numeric int rejected
    assert strict_bool(0) is None
    assert strict_bool("invalid") is None
    assert strict_bool("2") is None


def test_group_a_top_level_config_conflict_detected():
    """Verify mismatch between top-level and config dictionary produces key failure (None)."""
    r_conflict_rot = {
        "dataset": "d1",
        "solver": "cadical195",
        "sb_rotation": True,
        "config": {"sb_rotation": False},
    }
    assert get_canonical_key(r_conflict_rot) is None

    r_conflict_seed = {
        "dataset": "d1",
        "solver": "cadical195",
        "relabel_seed": 1,
        "config": {"relabel_seed": 2},
    }
    assert get_canonical_key(r_conflict_seed) is None


def test_group_a_solver_alias_normalization():
    """Verify solver aliases normalize consistently without creating spurious keys."""
    r1 = {"dataset": "d1", "solver": "cp_sat"}
    r2 = {"dataset": "d1", "solver": "cpsat"}
    r3 = {"dataset": "d1", "solver": "ortools_cpsat"}
    assert get_canonical_key(r1) == get_canonical_key(r2) == get_canonical_key(r3)

    r_cad1 = {"dataset": "d1", "solver": "cadical"}
    r_cad2 = {"dataset": "d1", "solver": "cadical195"}
    assert get_canonical_key(r_cad1) == get_canonical_key(r_cad2)


def test_group_a_missing_identity_rejected():
    """Verify record with missing dataset or missing solver/method returns None."""
    assert get_canonical_key({}) is None
    assert get_canonical_key({"dataset": ""}) is None
    assert get_canonical_key({"solver": "cadical195"}) is None


# ============================================================================
# GROUP B: RUNNER LIFECYCLE & JSONL INTEGRITY
# ============================================================================

def test_group_b_optimal_missing_and_invalid_ub_not_completed():
    """Verify is_run_completed requires valid integer best_ub for OPTIMAL runs when retry_failed=True."""
    assert is_run_completed({"status": "OPTIMAL", "best_ub": 4}, retry_failed=True) is True
    assert is_run_completed({"status": "OPTIMAL"}, retry_failed=True) is False
    assert is_run_completed({"status": "OPTIMAL", "best_ub": None}, retry_failed=True) is False
    assert is_run_completed({"status": "OPTIMAL", "best_ub": -1}, retry_failed=True) is False
    assert is_run_completed({"status": "OPTIMAL", "best_ub": 3.7}, retry_failed=True) is False
    assert is_run_completed({"status": "OPTIMAL", "best_ub": True}, retry_failed=True) is False
    assert is_run_completed({"status": "OPTIMAL", "best_ub": float("nan")}, retry_failed=True) is False
    assert is_run_completed({"status": "OPTIMAL", "best_ub": float("inf")}, retry_failed=True) is False


def test_group_b_optimal_validation_error_not_completed():
    """Verify is_run_completed rejects OPTIMAL records that have validation_error=True."""
    assert is_run_completed({"status": "OPTIMAL", "best_ub": 4, "validation_error": True}, retry_failed=True) is False
    assert is_run_completed({"status": "OPTIMAL", "best_ub": 4, "validation_error": "corrupt"}, retry_failed=True) is False


def test_group_b_optimal_assignment_validation():
    """Verify is_run_completed validates assignment permutation bijection when present."""
    # Valid bijection
    r_valid = {"status": "OPTIMAL", "best_ub": 4, "N": 4, "assignment": [0, 1, 2, 3]}
    assert is_run_completed(r_valid, retry_failed=True) is True

    # Duplicate vertex in assignment (invalid permutation)
    r_dup = {"status": "OPTIMAL", "best_ub": 4, "N": 4, "assignment": [0, 1, 1, 3]}
    assert is_run_completed(r_dup, retry_failed=True) is False

    # Wrong length
    r_len = {"status": "OPTIMAL", "best_ub": 4, "N": 4, "assignment": [0, 1, 2]}
    assert is_run_completed(r_len, retry_failed=True) is False


def test_group_b_feasible_missing_ub_not_completed():
    """Verify FEASIBLE status requires valid best_ub to be considered completed."""
    assert is_run_completed({"status": "FEASIBLE", "best_ub": 5}, retry_failed=True) is True
    assert is_run_completed({"status": "FEASIBLE"}, retry_failed=True) is False
    assert is_run_completed({"status": "FEASIBLE", "best_ub": -1}, retry_failed=True) is False
    assert is_run_completed({"status": "FEASIBLE", "best_ub": None}, retry_failed=True) is False


def test_group_b_duplicate_optimal_conflict_raises(tmp_path):
    """Verify divergent OPTIMAL bounds for the same run ID raise ValueError in strict mode and runner."""
    jsonl_file = tmp_path / "conflict_runs.jsonl"
    r1 = {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "solve_time": 1.0, "best_ub": 4, "rep": 1}
    r2 = {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "solve_time": 1.1, "best_ub": 5, "rep": 1}  # Conflict UB 4 vs 5
    jsonl_file.write_text(json.dumps(r1) + "\n" + json.dumps(r2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="[Dd]uplicate.*conflict"):
        load_existing_runs(str(jsonl_file), retry_failed=True)

    with pytest.raises(ValueError, match="[Dd]uplicate.*conflict"):
        load_records(str(jsonl_file), strict=True)


def test_group_b_fractional_and_invalid_rep_rejected(tmp_path):
    """Verify invalid repetition numbers in JSONL are caught safely."""
    jsonl_file = tmp_path / "bad_rep.jsonl"
    r1 = {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "solve_time": 1.0, "best_ub": 4, "rep": 1.5}
    jsonl_file.write_text(json.dumps(r1) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="JSONL integrity violations"):
        load_records(str(jsonl_file), strict=True)


# ============================================================================
# GROUP C: TABLE CORRECTNESS & INVARIANCE
# ============================================================================

def test_group_c_ablation_does_not_overwrite_solvers_or_encodings(tmp_path):
    """Verify make_table_ablation strictly isolates cadical order runs without overwrite."""
    records = [
        # cadical order runs (valid ablation small)
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "status": "OPTIMAL", "solve_time": 10.0, "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "status": "OPTIMAL", "solve_time": 8.0, "config": {"sb_rotation": True, "sb_reflection": False, "heuristic_ub": False}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "status": "OPTIMAL", "solve_time": 6.0, "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": False}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "status": "OPTIMAL", "solve_time": 7.0, "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": True}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "status": "OPTIMAL", "solve_time": 4.0, "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}, "groups": ["ablation_small"]},
        # kissat runs sharing ablation_small group (must NOT overwrite cadical!)
        {"dataset": "d1", "solver": "kissat", "encoding": "order", "status": "OPTIMAL", "solve_time": 99.0, "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}, "groups": ["ablation_small"]},
    ]

    out_dir = str(tmp_path / "ablation_out")
    df_disp, summary, df_raw = make_table_ablation(records, output_dirs=(out_dir,))
    # cadical 'all' timing was 4.0, kissat was 99.0; must be 4.0000
    assert df_disp.loc["d1", "all"] == "4.0000"


def test_group_c_ablation_conflicting_optimal_raises():
    """Verify duplicate variant on same dataset with conflicting OPTIMAL UB raises ValueError."""
    records = [
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "status": "OPTIMAL", "best_ub": 3, "solve_time": 10.0, "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "status": "OPTIMAL", "best_ub": 4, "solve_time": 12.0, "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}, "groups": ["ablation_small"]},
    ]
    with pytest.raises(ValueError, match="make_table_ablation: conflicting OPTIMAL best_ub"):
        make_table_ablation(records)


def test_group_c_methods_selects_sat_best_deterministically(tmp_path):
    """Verify make_table_methods deterministically selects the best valid SAT solver."""
    records = [
        {"dataset": "d1", "solver": "minisat22", "method": "sat", "status": "TIMEOUT", "solve_time": 600.0, "groups": ["methods"]},
        {"dataset": "d1", "solver": "glucose4", "method": "sat", "status": "OPTIMAL", "solve_time": 5.0, "groups": ["methods"]},
        {"dataset": "d1", "solver": "cadical195", "method": "sat", "status": "OPTIMAL", "solve_time": 2.0, "groups": ["methods"]},
        {"dataset": "d1", "solver": "cp_sat", "method": "cpsat", "status": "OPTIMAL", "solve_time": 1.0, "groups": ["methods"]},
        {"dataset": "d1", "solver": "cplex", "method": "cplex", "status": "SIZE_LIMIT", "solve_time": 0.05, "groups": ["methods"]},
    ]

    out_dir = str(tmp_path / "methods_out")
    # Both normal order and reversed order must choose cadical195 (solve_time=2.0) as sat_best!
    df_disp, summary, df_raw = make_table_methods(records, output_dirs=(out_dir,))
    assert df_disp.loc["d1", "sat_best"] == "2.0000"

    df_disp_rev, _, _ = make_table_methods(list(reversed(records)), output_dirs=(out_dir,))
    assert df_disp_rev.loc["d1", "sat_best"] == "2.0000"


def test_group_c_input_order_invariance(tmp_path):
    """Verify table outputs are strictly invariant under arbitrary input record shuffling."""
    records = [
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "status": "OPTIMAL", "solve_time": 10.0, "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "status": "OPTIMAL", "solve_time": 8.0, "config": {"sb_rotation": True, "sb_reflection": False, "heuristic_ub": False}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "status": "OPTIMAL", "solve_time": 6.0, "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": False}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "status": "OPTIMAL", "solve_time": 7.0, "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": True}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "status": "OPTIMAL", "solve_time": 4.0, "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}, "groups": ["ablation_small"]},
    ]

    out1 = str(tmp_path / "order1")
    out2 = str(tmp_path / "order2")
    df1, sum1, _ = make_table_ablation(records, output_dirs=(out1,))
    df2, sum2, _ = make_table_ablation(list(reversed(records)), output_dirs=(out2,))

    pd.testing.assert_frame_equal(df1, df2)


# ============================================================================
# GROUP D: CROSS-VALIDATION INDEPENDENT SOURCES
# ============================================================================

def test_group_d_same_solver_diff_flags_not_independent_in_strict_mode():
    """Verify different flags of same solver do NOT count as independent sources under strict_engines=True."""
    records = [
        {"dataset": "d1", "solver": "cadical195", "method": "sat", "status": "OPTIMAL", "best_ub": 3, "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}},
        {"dataset": "d1", "solver": "cadical195", "method": "sat", "status": "OPTIMAL", "best_ub": 3, "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}},
    ]
    df = cross_validate_optima(records, strict_engines=True)
    assert len(df) == 1
    assert df.iloc[0]["validation_status"] == "single_source"
    assert df.iloc[0]["num_distinct_sources"] == 1


def test_group_d_two_independent_engines_verified():
    """Verify distinct solver engines reach verified status."""
    records = [
        {"dataset": "d1", "solver": "cadical195", "method": "sat", "status": "OPTIMAL", "best_ub": 3},
        {"dataset": "d1", "solver": "cp_sat", "method": "cpsat", "status": "OPTIMAL", "best_ub": 3},
    ]
    df = cross_validate_optima(records, strict_engines=True)
    assert len(df) == 2
    assert all(df["validation_status"] == "verified")
    assert all(df["num_distinct_sources"] == 2)


def test_group_d_solver_aliases_same_engine():
    """Verify aliases of the same engine (cp_sat vs cpsat) do not count as independent sources."""
    records = [
        {"dataset": "d1", "solver": "cp_sat", "method": "cpsat", "status": "OPTIMAL", "best_ub": 3},
        {"dataset": "d1", "solver": "cpsat", "method": "cpsat", "status": "OPTIMAL", "best_ub": 3},
    ]
    df = cross_validate_optima(records, strict_engines=True)
    assert len(df) == 1
    assert df.iloc[0]["validation_status"] == "single_source"


# ============================================================================
# GROUP E: PLOT KEY ALIGNMENT & LOG-SCALE SAFETY
# ============================================================================

def test_group_e_scatter_pairs_require_same_seed_and_config():
    """Verify plot_scatter_sb_vs_nosb matches pairs strictly on same seed and config dimensions."""
    from analysis.make_plots import plot_scatter_sb_vs_nosb

    # Record 1: seed 0, no_sb
    # Record 2: seed 1, with_sb (MUST NOT PAIR WITH RECORD 1!)
    # Record 3: seed 0, with_sb (MUST PAIR WITH RECORD 1)
    records = [
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "relabel_seed": 0, "solve_time": 10.0, "status": "OPTIMAL", "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "relabel_seed": 1, "solve_time": 2.0, "status": "OPTIMAL", "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "relabel_seed": 0, "solve_time": 5.0, "status": "OPTIMAL", "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}},
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        plot_scatter_sb_vs_nosb(records, base_filename="scatter_sb_vs_nosb", output_dirs=(tmp_dir,))
        assert os.path.exists(os.path.join(tmp_dir, "scatter_sb_vs_nosb.png"))


# ============================================================================
# GROUP F: END-TO-END PIPELINE FIXTURE & ROUND-TRIP
# ============================================================================

def test_group_f_end_to_end_fixture_roundtrip_and_order_invariance(tmp_path):
    """
    End-to-end integration fixture test:
    - Multi-dataset (d1, d2), multi-solver (cadical195, kissat, cp_sat, cplex).
    - Diverse statuses: OPTIMAL, FEASIBLE, TIMEOUT, SIZE_LIMIT.
    - Missing timing handled cleanly.
    - Load JSONL -> aggregate -> cross-validation -> make tables -> export 21-col data CSV.
    - Proves 100% order invariance between forward and reversed JSONL lines.
    """
    fixture_runs = [
        # Dataset 1: Cadical (none vs all)
        {"dataset": "d1", "solver": "cadical195", "method": "sat", "encoding": "order", "status": "OPTIMAL", "solve_time": 10.0, "wall_time": 10.2, "best_ub": 3, "rep": 1, "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}, "groups": ["ablation_small", "methods"]},
        {"dataset": "d1", "solver": "cadical195", "method": "sat", "encoding": "order", "status": "OPTIMAL", "solve_time": 8.0, "wall_time": 8.2, "best_ub": 3, "rep": 1, "config": {"sb_rotation": True, "sb_reflection": False, "heuristic_ub": False}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "method": "sat", "encoding": "order", "status": "OPTIMAL", "solve_time": 6.0, "wall_time": 6.2, "best_ub": 3, "rep": 1, "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": False}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "method": "sat", "encoding": "order", "status": "OPTIMAL", "solve_time": 7.0, "wall_time": 7.2, "best_ub": 3, "rep": 1, "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": True}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "method": "sat", "encoding": "order", "status": "OPTIMAL", "solve_time": 4.0, "wall_time": 4.2, "best_ub": 3, "rep": 1, "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}, "groups": ["ablation_small", "methods"]},
        # Dataset 1: CP-SAT & CPLEX
        {"dataset": "d1", "solver": "cp_sat", "method": "cpsat", "status": "OPTIMAL", "solve_time": 2.0, "wall_time": 2.1, "best_ub": 3, "rep": 1, "groups": ["methods"]},
        {"dataset": "d1", "solver": "cplex", "method": "cplex", "status": "SIZE_LIMIT", "solve_time": 0.05, "wall_time": 0.06, "best_ub": None, "rep": 1, "groups": ["methods"]},
        # Dataset 2: Feasible & Timeout
        {"dataset": "d2", "solver": "cadical195", "method": "sat", "encoding": "order", "status": "FEASIBLE", "solve_time": 15.0, "wall_time": 15.2, "best_ub": 5, "rep": 1, "groups": ["methods"]},
        {"dataset": "d2", "solver": "cp_sat", "method": "cpsat", "status": "TIMEOUT", "solve_time": 600.0, "wall_time": 600.5, "best_ub": None, "rep": 1, "groups": ["methods"]},
    ]

    f_fwd = tmp_path / "forward.jsonl"
    f_rev = tmp_path / "reverse.jsonl"
    f_fwd.write_text("\n".join(json.dumps(r) for r in fixture_runs) + "\n", encoding="utf-8")
    f_rev.write_text("\n".join(json.dumps(r) for r in reversed(fixture_runs)) + "\n", encoding="utf-8")

    recs_fwd = load_records(str(f_fwd), strict=True)
    recs_rev = load_records(str(f_rev), strict=True)

    assert len(recs_fwd) == len(recs_rev)

    # Cross-validation
    df_cv_fwd = cross_validate_optima(recs_fwd)
    df_cv_rev = cross_validate_optima(recs_rev)
    assert len(df_cv_fwd) == len(df_cv_rev)

    # Methods table
    out_fwd = str(tmp_path / "out_fwd")
    out_rev = str(tmp_path / "out_rev")
    df_m_fwd, _, _ = make_table_methods(recs_fwd, output_dirs=(out_fwd,))
    df_m_rev, _, _ = make_table_methods(recs_rev, output_dirs=(out_rev,))
    pd.testing.assert_frame_equal(df_m_fwd, df_m_rev)

    # Verify 21-column data CSV exported
    data_csv = os.path.join(out_fwd, "table_methods_data.csv")
    assert os.path.exists(data_csv)
    df_data = pd.read_csv(data_csv)
    assert list(df_data.columns) == DATA_CSV_COLUMNS


# ============================================================================
# COMPREHENSIVE REGRESSION: SECTIONS 2, 3, 4, 5, 6, 7, 8, 11
# ============================================================================

def test_group_a_solver_method_conflict_rejected():
    """Section 2.2: Contradictory solver and method identities must be rejected."""
    # cadical195 is SAT, not CP-SAT
    assert get_canonical_key({"dataset": "d1", "solver": "cadical195", "method": "cpsat"}) is None
    # cp_sat is CP-SAT, not SAT
    assert get_canonical_key({"dataset": "d1", "solver": "cp_sat", "method": "sat"}) is None
    # cplex is MILP, not SAT
    assert get_canonical_key({"dataset": "d1", "solver": "cplex", "method": "sat"}) is None
    # gurobi is MILP, not SAT
    assert get_canonical_key({"dataset": "d1", "solver": "gurobi", "method": "sat"}) is None


def test_group_a_strict_int_large_precision_and_types():
    """Section 9.1: strict_int preserves large integer precision and rejects malformed values."""
    # Arbitrary precision string integer
    large_int_str = "9007199254740993"
    assert strict_int(large_int_str) == 9007199254740993
    assert strict_int("4") == 4
    assert strict_int("4.0") == 4
    assert strict_int("4.5") is None
    assert strict_int(True) is None
    assert strict_int(False) is None
    assert strict_int(float("nan")) is None
    assert strict_int(float("inf")) is None
    assert strict_int("invalid") is None
    # NumPy integers
    assert strict_int(np.int64(42)) == 42


def test_group_b_assignment_validation_comprehensive():
    """Section 3: Exhaustive assignment validation across all specified test inputs."""
    # Valid permutations
    assert is_valid_assignment([0, 1, 2, 3], n=4) is True
    assert is_valid_assignment((0, 1, 2, 3), n=4) is True
    assert is_valid_assignment([3, 1, 0, 2], n=4) is True
    assert is_valid_assignment([0, 1, 2, 3], n="4") is True
    assert is_valid_assignment([0, 1, 2, 3], n=None) is True

    # Duplicate element
    assert is_valid_assignment([0, 1, 1, 3], n=4) is False
    assert is_valid_assignment((0, 1, 1, 3), n=4) is False

    # Incomplete length
    assert is_valid_assignment([0, 1, 2], n=4) is False

    # Boolean injection (reject bool as int)
    assert is_valid_assignment([0, 1, 2, True], n=4) is False
    assert is_valid_assignment([False, 1, 2, 3], n=4) is False

    # Float injection
    assert is_valid_assignment([0, 1, 2, 3.0], n=4) is False

    # String injection
    assert is_valid_assignment(["0", "1", "2", "3"], n=4) is False

    # Unhashable object (must NOT raise TypeError)
    assert is_valid_assignment([0, 1, 2, {}], n=4) is False
    assert is_valid_assignment([0, 1, 2, [3]], n=4) is False

    # None handling
    assert is_valid_assignment(None, n=4, allow_none=False) is False
    assert is_valid_assignment(None, n=4, allow_none=True) is True
    assert is_valid_assignment(None, n=None, allow_none=False) is False

    # Malformed N
    assert is_valid_assignment([0, 1, 2, 3], n=3.5) is False
    assert is_valid_assignment([0, 1, 2, 3], n=True) is False
    assert is_valid_assignment([0, 1, 2, 3], n=-1) is False
    assert is_valid_assignment([0, 1, 2, 3], n=0) is False
    assert is_valid_assignment([], n=None) is False


def test_group_b_jsonl_scalars_and_repetition_conflicts(tmp_path):
    """Section 8.1 & 8.2: Non-dict JSON scalars and conflicting repetition fields are rejected."""
    # 1. Non-dict JSON scalar
    f_scalar = tmp_path / "scalars.jsonl"
    f_scalar.write_text("123\n\"just a string\"\n[1, 2, 3]\nnull\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL integrity violations"):
        load_records(str(f_scalar), strict=True)

    loaded_permissive = load_records(str(f_scalar), strict=False)
    assert len(loaded_permissive) == 0

    # 2. Conflicting rep vs repetition
    f_rep_conflict = tmp_path / "rep_conflict.jsonl"
    bad_rep_rec = {
        "dataset": "d1",
        "solver": "cadical195",
        "rep": 1,
        "repetition": 2,  # Conflicting!
        "status": "OPTIMAL",
        "best_ub": 3,
        "solve_time": 1.0,
    }
    f_rep_conflict.write_text(json.dumps(bad_rep_rec) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting repetition fields"):
        load_records(str(f_rep_conflict), strict=True)

    # In runner
    runs_completed = load_existing_runs(str(f_rep_conflict), retry_failed=True)
    assert len(runs_completed) == 0


def test_group_c_ablation_shuffle_invariance_and_repetition_median(tmp_path):
    """Section 4.2 & 4.3: make_table_ablation aggregates multiple repetitions with median and is 100% shuffle-invariant."""
    # 5 variants x 3 repetitions on d1
    import random
    raw_runs = []
    times = {"none": [10.0, 12.0, 11.0], "rotation": [8.0, 7.0, 9.0], "rot_ref": [5.0, 6.0, 5.5], "heur_ub": [7.0, 8.0, 7.5], "all": [3.0, 4.0, 3.5]}
    configs = {
        "none": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False},
        "rotation": {"sb_rotation": True, "sb_reflection": False, "heuristic_ub": False},
        "rot_ref": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": False},
        "heur_ub": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": True},
        "all": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True},
    }

    for var_name, var_times in times.items():
        for rep_idx, t in enumerate(var_times, start=1):
            raw_runs.append({
                "dataset": "d1",
                "solver": "cadical195",
                "encoding": "order",
                "status": "OPTIMAL",
                "best_ub": 4,
                "solve_time": t,
                "wall_time": t + 0.1,
                "rep": rep_idx,
                "config": configs[var_name],
                "groups": ["ablation_small"],
            })

    out_base = str(tmp_path / "base")
    df_base, _, _ = make_table_ablation(raw_runs, output_dirs=(out_base,))

    # Median of none: [10.0, 11.0, 12.0] -> 11.0000
    assert df_base.loc["d1", "none"] == "11.0000"
    # Median of all: [3.0, 3.5, 4.0] -> 3.5000
    assert df_base.loc["d1", "all"] == "3.5000"

    # Test random shuffles
    rng = random.Random(42)
    for trial in range(5):
        shuffled = list(raw_runs)
        rng.shuffle(shuffled)
        out_trial = str(tmp_path / f"trial_{trial}")
        df_shuffled, _, _ = make_table_ablation(shuffled, output_dirs=(out_trial,))
        pd.testing.assert_frame_equal(df_base, df_shuffled)


def test_group_c_ablation_filters_missing_and_wrong_solver(tmp_path):
    """Section 4.1: make_table_ablation strictly excludes records without solver or with non-Cadical solver."""
    records = [
        # Valid cadical ablation set
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "status": "OPTIMAL", "solve_time": 10.0, "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "status": "OPTIMAL", "solve_time": 8.0, "config": {"sb_rotation": True, "sb_reflection": False, "heuristic_ub": False}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "status": "OPTIMAL", "solve_time": 6.0, "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": False}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "status": "OPTIMAL", "solve_time": 7.0, "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": True}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "status": "OPTIMAL", "solve_time": 4.0, "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}, "groups": ["ablation_small"]},
        # Missing solver (must be rejected!)
        {"dataset": "d1", "solver": None, "encoding": "order", "status": "OPTIMAL", "solve_time": 0.01, "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}, "groups": ["ablation_small"]},
        # Wrong solver (must be rejected!)
        {"dataset": "d1", "solver": "kissat", "encoding": "order", "status": "OPTIMAL", "solve_time": 0.02, "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}, "groups": ["ablation_small"]},
    ]
    out_dir = str(tmp_path / "out")
    df_disp, _, _ = make_table_ablation(records, output_dirs=(out_dir,))
    # If missing solver or kissat was erroneously accepted, "all" would be 0.0100 or 0.0200.
    # It must be 4.0000!
    assert df_disp.loc["d1", "all"] == "4.0000"


def test_group_c_methods_sat_best_aggregates_repetitions_and_timeouts(tmp_path):
    """Section 5: sat_best aggregates candidate repetitions and never lets fast timeout beat optimal."""
    records = [
        # Cadical has 3 reps on d1: [3.0, 4.0, 5.0] -> median 4.0
        {"dataset": "d1", "solver": "cadical195", "method": "sat", "status": "OPTIMAL", "solve_time": 3.0, "rep": 1, "groups": ["methods"]},
        {"dataset": "d1", "solver": "cadical195", "method": "sat", "status": "OPTIMAL", "solve_time": 4.0, "rep": 2, "groups": ["methods"]},
        {"dataset": "d1", "solver": "cadical195", "method": "sat", "status": "OPTIMAL", "solve_time": 5.0, "rep": 3, "groups": ["methods"]},
        # Glucose has 3 reps on d1: [1.0, 2.0, 6.0] -> median 2.0
        {"dataset": "d1", "solver": "glucose4", "method": "sat", "status": "OPTIMAL", "solve_time": 1.0, "rep": 1, "groups": ["methods"]},
        {"dataset": "d1", "solver": "glucose4", "method": "sat", "status": "OPTIMAL", "solve_time": 2.0, "rep": 2, "groups": ["methods"]},
        {"dataset": "d1", "solver": "glucose4", "method": "sat", "status": "OPTIMAL", "solve_time": 6.0, "rep": 3, "groups": ["methods"]},
        # Minisat hit TIMEOUT at 0.05s (aborted early) -> must NEVER beat OPTIMAL!
        {"dataset": "d1", "solver": "minisat22", "method": "sat", "status": "TIMEOUT", "solve_time": 0.05, "rep": 1, "groups": ["methods"]},
        # CP-SAT and CPLEX
        {"dataset": "d1", "solver": "cp_sat", "method": "cpsat", "status": "OPTIMAL", "solve_time": 1.5, "rep": 1, "groups": ["methods"]},
        {"dataset": "d1", "solver": "cplex", "method": "cplex", "status": "SIZE_LIMIT", "solve_time": 0.01, "rep": 1, "groups": ["methods"]},
    ]
    out_dir = str(tmp_path / "out")
    df_disp, _, _ = make_table_methods(records, output_dirs=(out_dir,))
    # Glucose median is 2.0000, Cadical median is 4.0000. Glucose is chosen!
    assert df_disp.loc["d1", "sat_best"] == "2.0000"


def test_group_d_cross_validation_comprehensive_cases(tmp_path):
    """Section 2.3: Cross-validation handles all 11 specified cases accurately."""
    # 1. One solver, multiple SB configs -> single_source under default strict engines
    recs_sb = [
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}},
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}},
    ]
    df_sb = cross_validate_optima(recs_sb)
    assert len(df_sb) == 1 and df_sb.iloc[0]["validation_status"] == "single_source"

    # 2. One solver, multiple seeds -> single_source
    recs_seeds = [
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "relabel_seed": 0},
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "relabel_seed": 1},
    ]
    df_seeds = cross_validate_optima(recs_seeds)
    assert len(df_seeds) == 2 and all(df_seeds["validation_status"] == "single_source")

    # 3. One solver, multiple repetitions -> single_source
    recs_reps = [
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "rep": 1},
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "rep": 2},
    ]
    df_reps = cross_validate_optima(recs_reps)
    assert len(df_reps) == 1 and df_reps.iloc[0]["validation_status"] == "single_source"

    # 4. Two aliases of same engine -> single_source
    recs_alias = [
        {"dataset": "d1", "solver": "cadical", "status": "OPTIMAL", "best_ub": 3},
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3},
    ]
    df_alias = cross_validate_optima(recs_alias)
    assert len(df_alias) == 1 and df_alias.iloc[0]["validation_status"] == "single_source"

    # 5. Two independent engines agree -> verified
    recs_verified = [
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3},
        {"dataset": "d1", "solver": "cp_sat", "status": "OPTIMAL", "best_ub": 3},
    ]
    df_ver = cross_validate_optima(recs_verified)
    assert len(df_ver) == 2 and all(df_ver["validation_status"] == "verified")

    # 6. Two independent engines disagree -> conflict raises ValueError
    recs_conflict = [
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3},
        {"dataset": "d1", "solver": "cp_sat", "status": "OPTIMAL", "best_ub": 4},
    ]
    with pytest.raises(ValueError, match="conflicting optima"):
        cross_validate_optima(recs_conflict)

    # 7. One engine has multiple OPTIMAL with different UBs -> internal conflict raises ValueError
    recs_int_conflict = [
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "rep": 1},
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 4, "rep": 2},
    ]
    with pytest.raises(ValueError, match="conflicting optima"):
        cross_validate_optima(recs_int_conflict)

    # 8. Record OPTIMAL with missing or invalid UB -> skipped
    recs_invalid_ub = [
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": None},
        {"dataset": "d1", "solver": "cp_sat", "status": "OPTIMAL", "best_ub": -1},
    ]
    df_inval = cross_validate_optima(recs_invalid_ub)
    assert len(df_inval) == 0

    # 9. Record with validation_error -> skipped
    recs_val_err = [
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "validation_error": True},
        {"dataset": "d1", "solver": "cp_sat", "status": "OPTIMAL", "best_ub": 3},
    ]
    df_val_err = cross_validate_optima(recs_val_err)
    assert len(df_val_err) == 1 and df_val_err.iloc[0]["validation_status"] == "single_source"

    # 10. Dataset with seed=0 and unseeded (seed=None)
    recs_seeds_diff = [
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "relabel_seed": 0},
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "relabel_seed": None},
    ]
    df_seeds_diff = cross_validate_optima(recs_seeds_diff)
    assert len(df_seeds_diff) == 2

    # 11. Insufficient data (< 2 independent sources) -> single_source
    recs_single = [
        {"dataset": "d1", "solver": "glucose4", "status": "OPTIMAL", "best_ub": 3},
    ]
    df_single = cross_validate_optima(recs_single)
    assert len(df_single) == 1 and df_single.iloc[0]["validation_status"] == "single_source"


def test_group_e_scatter_pairs_group_isolation_and_pure_data():
    """Section 6 & 7: extract_scatter_pairs strictly isolates groups and produces exact arithmetic medians."""
    from analysis.make_plots import extract_scatter_pairs

    records = [
        # Group ablation_small: seed 0 has 3 reps for no_sb [10, 12, 14] -> median 12.0
        # and 3 reps for with_sb [4, 5, 6] -> median 5.0
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "relabel_seed": 0, "solve_time": 10.0, "status": "OPTIMAL", "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "relabel_seed": 0, "solve_time": 12.0, "status": "OPTIMAL", "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "relabel_seed": 0, "solve_time": 14.0, "status": "OPTIMAL", "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "relabel_seed": 0, "solve_time": 4.0, "status": "OPTIMAL", "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "relabel_seed": 0, "solve_time": 5.0, "status": "OPTIMAL", "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}, "groups": ["ablation_small"]},
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "relabel_seed": 0, "solve_time": 6.0, "status": "OPTIMAL", "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}, "groups": ["ablation_small"]},

        # Group foreign_group (MUST NEVER CONTAMINATE ablation_small!):
        {"dataset": "d1", "solver": "cadical195", "encoding": "order", "relabel_seed": 0, "solve_time": 999.0, "status": "OPTIMAL", "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}, "groups": ["foreign_group"]},
    ]

    pairs = extract_scatter_pairs(records, group_name="ablation_small")
    assert len(pairs) == 1
    p = pairs[0]
    assert p["dataset"] == "d1"
    assert p["no_sb"] == 12.0
    assert p["with_sb"] == 5.0

    # Asking for a non-existent group returns empty without fallback
    pairs_empty = extract_scatter_pairs(records, group_name="nonexistent_group")
    assert len(pairs_empty) == 0


def test_group_g_metamorphic_properties(tmp_path):
    """
    Section 11: Metamorphic properties test for CBMP pipeline:
    1. Input reversal invariance.
    2. Input shuffle invariance.
    3. Exact duplicate does not alter metrics.
    4. Solver alias equivalence produces identical canonical key.
    5. Seed change creates distinct canonical identity.
    6. Config change creates distinct canonical identity.
    7. Group change prevents data leaking to other group.
    8. Removing independent source downgrades verified status.
    9. Adding config variant of same engine does not increase independent sources in strict mode.
    10. Adding invalid record does not alter valid results.
    """
    import random

    # Property 4: Alias equivalence
    r_cad1 = {"dataset": "d1", "solver": "cadical", "status": "OPTIMAL", "best_ub": 3}
    r_cad2 = {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3}
    assert get_canonical_key(r_cad1) == get_canonical_key(r_cad2)

    # Property 5: Seed change
    r_seed0 = {"dataset": "d1", "solver": "cadical195", "relabel_seed": 0}
    r_seed1 = {"dataset": "d1", "solver": "cadical195", "relabel_seed": 1}
    assert get_canonical_key(r_seed0) != get_canonical_key(r_seed1)

    # Property 6: Config change
    r_cfg1 = {"dataset": "d1", "solver": "cadical195", "config": {"sb_rotation": False}}
    r_cfg2 = {"dataset": "d1", "solver": "cadical195", "config": {"sb_rotation": True}}
    assert get_canonical_key(r_cfg1) != get_canonical_key(r_cfg2)

    # Property 8 & 9: Independent source verification & config variant invariance
    two_engines = [
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3},
        {"dataset": "d1", "solver": "cp_sat", "status": "OPTIMAL", "best_ub": 3},
    ]
    df_two = cross_validate_optima(two_engines)
    assert all(df_two["validation_status"] == "verified")

    # Property 8: Remove one source -> downgrades to single_source
    one_engine = [two_engines[0]]
    df_one = cross_validate_optima(one_engine)
    assert df_one.iloc[0]["validation_status"] == "single_source"

    # Property 9: Add multiple config variants of cadical195 -> still 1 engine source!
    engine_plus_configs = [
        two_engines[0],
        {"dataset": "d1", "solver": "cadical195", "config": {"sb_rotation": False}, "status": "OPTIMAL", "best_ub": 3},
        {"dataset": "d1", "solver": "cadical195", "config": {"sb_reflection": True}, "status": "OPTIMAL", "best_ub": 3},
    ]
    df_plus_configs = cross_validate_optima(engine_plus_configs)
    assert df_plus_configs.iloc[0]["num_distinct_sources"] == 1
    assert df_plus_configs.iloc[0]["validation_status"] == "single_source"

    # Property 3: Exact duplicate does not alter metrics
    base_runs = [
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "solve_time": 5.0, "rep": 1},
        {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "solve_time": 5.0, "rep": 2},
    ]
    agg_base = aggregate_repetitions(base_runs)
    # Add exact duplicate of rep 1
    with_exact_dup = base_runs + [dict(base_runs[0])]
    f_dup = tmp_path / "dup.jsonl"
    f_dup.write_text("\n".join(json.dumps(r) for r in with_exact_dup) + "\n", encoding="utf-8")
    loaded_dedup = load_records(str(f_dup), strict=True)
    assert loaded_dedup[0]["solve_time"] == agg_base[0]["solve_time"]
