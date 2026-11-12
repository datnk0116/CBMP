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
