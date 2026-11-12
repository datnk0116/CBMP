"""
tests/test_pipeline_integrity.py

Comprehensive Regression and Pipeline Integrity Test Suite for CBMP:
Group A: Canonical Identity & Key Alignment
Group B: JSONL Integrity & Safe Error Reporting
Group C: Validation & Bounds Semantics
Group D: Repetition Aggregation & Exact PAR2
Group E: Independent Cross-Validation
Group F: Table Generation & Metric Presentation
Group G: Plot Key Alignment & Log-Scale Safety
Group H: Experiment Runner Lifecycle (Resume, Retry, Deduplication, Process Crash)
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


def test_group_a_missing_solver_method_preservation():
    """Verify that when solver is empty or missing, method distinction prevents collision."""
    r_sat = {"dataset": "d1", "method": "sat", "solver": ""}
    r_cpsat = {"dataset": "d1", "method": "cpsat", "solver": ""}
    assert get_canonical_key(r_sat) != get_canonical_key(r_cpsat)


# ============================================================================
# GROUP B: JSONL INTEGRITY & SAFE ERROR REPORTING
# ============================================================================

def test_group_b_load_records_syntax_errors_and_non_objects(tmp_path):
    """Verify load_records handles syntax errors, primitives, and non-objects safely."""
    jsonl_file = tmp_path / "corrupt.jsonl"
    lines = [
        "",  # empty line
        "   ",  # whitespace
        '{"_type": "metadata", "env": "mac"}',  # metadata record
        "NOT A JSON LINE {{{",  # syntax error
        '"just a string"',  # valid json, but string
        "42",  # valid json, but number
        "[1, 2, 3]",  # valid json, but list
        "null",  # valid json, but null
        '{"solver": "cadical195"}',  # missing dataset
        '{"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "solve_time": 1.23, "best_ub": 4, "rep": 1}',  # valid record
    ]
    jsonl_file.write_text("\n".join(lines), encoding="utf-8")

    # Non-strict mode: skips invalid lines without raising
    records = load_records(str(jsonl_file), strict=False)
    assert len(records) == 1
    assert records[0]["dataset"] == "d1"
    assert records[0]["best_ub"] == 4

    # Strict mode: raises ValueError on syntax errors
    with pytest.raises(ValueError, match="JSONL integrity violations"):
        load_records(str(jsonl_file), strict=True)


def test_group_b_duplicate_run_deduplication(tmp_path):
    """Verify deduplication of identical (canonical_key, rep) runs with warning on conflict."""
    jsonl_file = tmp_path / "dups.jsonl"
    r1 = {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "solve_time": 1.0, "best_ub": 4, "rep": 1}
    r2 = {"dataset": "d1", "solver": "cadical195", "status": "OPTIMAL", "solve_time": 1.1, "best_ub": 4, "rep": 1}  # duplicate same rep
    lines = [json.dumps(r1), json.dumps(r2)]
    jsonl_file.write_text("\n".join(lines), encoding="utf-8")

    records = load_records(str(jsonl_file), strict=False)
    assert len(records) == 1
    assert records[0]["n_total"] == 1


# ============================================================================
# GROUP C: VALIDATION & BOUNDS SEMANTICS
# ============================================================================

def test_group_c_validate_best_ub_and_lower_bound():
    """Verify validate_best_ub and validate_lower_bound reject booleans, negative, fractional, and invalid types."""
    # Valid non-negative integers
    assert validate_best_ub(0) == 0
    assert validate_best_ub(5) == 5
    assert validate_best_ub(5.0) == 5
    assert validate_best_ub("5") == 5
    assert validate_best_ub(np.int64(10)) == 10

    assert validate_lower_bound(0) == 0
    assert validate_lower_bound(5) == 5
    assert validate_lower_bound(5.0) == 5

    # Invalid values must return None
    for invalid in [
        True, False,  # Booleans
        -1, -5.0, "-1",  # Negatives
        3.7, 0.9, "3.7",  # Non-integers
        float("nan"), float("inf"), float("-inf"),  # Non-finite
        None, pd.NA, pd.NaT,  # Missing
        [], {}, "invalid_string",  # Malformed
    ]:
        assert validate_best_ub(invalid) is None
        assert validate_lower_bound(invalid) is None


def test_group_c_validate_solve_time_strict():
    """Verify validate_solve_time rejects booleans, negatives, and non-finite numbers."""
    assert validate_solve_time(0.0) == 0.0
    assert validate_solve_time(12.34) == 12.34
    assert validate_solve_time("12.34") == 12.34
    assert validate_solve_time(np.float64(5.5)) == 5.5

    for bad in [True, False, -0.01, -5.0, float("nan"), float("inf"), float("-inf"), None, pd.NA, "bad"]:
        assert validate_solve_time(bad) is None


def test_group_c_bound_conflict_triggers_validation_error():
    """Verify minimization bound conflict (best_ub < lower_bound) is flagged as validation_error and status ERROR."""
    rec = {
        "dataset": "ds_conflict",
        "solver": "cadical",
        "status": "OPTIMAL",
        "solve_time": 1.0,
        "best_ub": 2,
        "lower_bound": 5,  # UB 2 < LB 5 is impossible in minimization
        "rep": 1,
    }
    agg = aggregate_repetition_group([rec])
    assert agg["status"] == "ERROR"
    assert agg["validation_error"] is True
    assert agg["best_ub"] == -1


# ============================================================================
# GROUP D: REPETITION AGGREGATION & EXACT PAR2
# ============================================================================

def test_group_d_population_counters_sum_to_n_total():
    """Verify all 12 status population counters are present and sum exactly to n_total."""
    reps = [
        {"dataset": "ds1", "solver": "cadical", "status": "OPTIMAL", "solve_time": 1.0, "best_ub": 4, "time_limit": 300.0, "rep": 1},
        {"dataset": "ds1", "solver": "cadical", "status": "TIMEOUT", "solve_time": 300.0, "best_ub": 5, "time_limit": 300.0, "rep": 2},
        {"dataset": "ds1", "solver": "cadical", "status": "FEASIBLE", "solve_time": 10.0, "best_ub": 6, "time_limit": 300.0, "rep": 3},
        {"dataset": "ds1", "solver": "cadical", "status": "INFEASIBLE", "time_limit": 300.0, "rep": 4},
        {"dataset": "ds1", "solver": "cadical", "status": "SIZE_LIMIT", "time_limit": 300.0, "rep": 5},
        {"dataset": "ds1", "solver": "cadical", "status": "UNAVAILABLE", "time_limit": 300.0, "rep": 6},
        {"dataset": "ds1", "solver": "cadical", "status": "LICENSE_ERROR", "time_limit": 300.0, "rep": 7},
        {"dataset": "ds1", "solver": "cadical", "status": "ERROR", "time_limit": 300.0, "rep": 8},
        {"dataset": "ds1", "solver": "cadical", "status": "MEMOUT", "time_limit": 300.0, "rep": 9},
        {"dataset": "ds1", "solver": "cadical", "status": "INTERRUPTED", "time_limit": 300.0, "rep": 10},
    ]
    agg = aggregate_repetition_group(reps, default_timeout=300.0)
    assert agg["n_total"] == 10
    assert agg["n_algorithmic"] == 7  # OPTIMAL, TIMEOUT, FEASIBLE, INFEASIBLE, ERROR, MEMOUT, INTERRUPTED
    status_sum = (
        agg["n_optimal"] + agg["n_timeout"] + agg["n_feasible"] + agg["n_infeasible"] +
        agg["n_size_limit"] + agg["n_unavailable"] + agg["n_license_error"] +
        agg["n_error"] + agg["n_interrupted"] + agg["n_unknown"]
    )
    assert status_sum == agg["n_total"]


def test_group_d_pure_solve_time_median_and_missing_timing_penalty():
    """Verify median solve_time is computed strictly over genuine solve_times without wall_time."""
    reps = [
        {"dataset": "ds_timing", "solver": "cadical", "status": "OPTIMAL", "solve_time": 10.0, "wall_time": 12.0, "time_limit": 300.0, "best_ub": 3, "rep": 1},
        {"dataset": "ds_timing", "solver": "cadical", "status": "OPTIMAL", "solve_time": None, "wall_time": 20.0, "time_limit": 300.0, "best_ub": 3, "rep": 2},
        {"dataset": "ds_timing", "solver": "cadical", "status": "OPTIMAL", "solve_time": 30.0, "wall_time": 32.0, "time_limit": 300.0, "best_ub": 3, "rep": 3},
    ]
    agg = aggregate_repetition_group(reps, default_timeout=300.0)
    # Genuine solve_times: [10.0, 30.0] -> median is 20.0 (not mixed with wall_time)
    assert abs(agg["solve_time"] - 20.0) < 1e-5
    assert agg["solve_time_source"] == "solve_time"

    # In PAR2, rep 1 gets 10.0, rep 2 (missing solve_time) gets 2 * 300 = 600.0 penalty, rep 3 gets 30.0
    # Expected PAR2: (10.0 + 600.0 + 30.0) / 3 = 640.0 / 3 = 213.3333...
    assert abs(agg["par2"] - (640.0 / 3.0)) < 1e-5


# ============================================================================
# GROUP E: INDEPENDENT CROSS-VALIDATION
# ============================================================================

def test_group_e_independent_cross_validation_logic():
    """Verify cross-validation requires >= 2 independent sources for verified status."""
    # 1. Single source only -> single_source
    single_source_recs = [
        {"dataset": "ds_indep", "solver": "cadical195", "method": "sat", "status": "OPTIMAL", "best_ub": 5, "relabel_seed": 0, "rep": 1},
        {"dataset": "ds_indep", "solver": "cadical195", "method": "sat", "status": "OPTIMAL", "best_ub": 5, "relabel_seed": 0, "rep": 2},
    ]
    df_single = cross_validate_optima(single_source_recs)
    assert df_single.iloc[0]["validation_status"] == "single_source"

    # 2. Two distinct independent sources with identical UB -> verified
    two_sources_recs = [
        {"dataset": "ds_indep", "solver": "cadical195", "method": "sat", "status": "OPTIMAL", "best_ub": 5, "relabel_seed": 0, "rep": 1},
        {"dataset": "ds_indep", "solver": "kissat", "method": "sat", "status": "OPTIMAL", "best_ub": 5, "relabel_seed": 0, "rep": 1},
    ]
    df_verified = cross_validate_optima(two_sources_recs)
    assert df_verified.iloc[0]["validation_status"] == "verified"


def test_group_e_conflicting_optima_raises_error():
    """Verify conflicting optima between independent sources raise ValueError."""
    conflict_recs = [
        {"dataset": "ds_conf", "solver": "cadical195", "method": "sat", "status": "OPTIMAL", "best_ub": 5, "relabel_seed": 0},
        {"dataset": "ds_conf", "solver": "cp_sat", "method": "cpsat", "status": "OPTIMAL", "best_ub": 4, "relabel_seed": 0},
    ]
    with pytest.raises(ValueError, match="[Cc]onflicting optima"):
        cross_validate_optima(conflict_recs)


def test_group_e_incumbent_violation_raises_error():
    """Verify non-OPTIMAL incumbent with UB < proven optimum raises ValueError."""
    violation_recs = [
        {"dataset": "ds_opt", "solver": "cadical195", "method": "sat", "status": "OPTIMAL", "best_ub": 5, "relabel_seed": 0},
        {"dataset": "ds_opt", "solver": "kissat", "method": "sat", "status": "FEASIBLE", "best_ub": 3, "relabel_seed": 0},  # UB 3 < proven opt 5
    ]
    with pytest.raises(ValueError, match="Invalid incumbent"):
        cross_validate_optima(violation_recs)


# ============================================================================
# GROUP F: TABLE GENERATION & METRIC PRESENTATION
# ============================================================================

def test_group_f_format_cell_value_missing_timing_displays_dash():
    """Verify missing timing on OPTIMAL displays '-' rather than 0.0000."""
    assert format_cell_value("OPTIMAL", solve_time=None) == "-"
    assert format_cell_value({"status": "OPTIMAL", "solve_time": None}) == "-"
    assert format_cell_value({"status": "OPTIMAL", "solve_time": 1.234}) == "1.2340"


def test_group_f_speedup_safety_on_missing_or_nonpositive_timing():
    """Verify speedup is NaN when timing is missing, zero, or negative."""
    df_raw = pd.DataFrame({
        "base": [{"status": "OPTIMAL", "solve_time": 10.0}],
        "cand_missing": [{"status": "OPTIMAL", "solve_time": None}],
        "cand_zero": [{"status": "OPTIMAL", "solve_time": 0.0}],
    }, index=["ds1"])

    solved, sl, par2, speedup = compute_summary_stats(
        df_raw, ["base", "cand_missing", "cand_zero"], baseline_col="base"
    )
    assert abs(speedup["base"] - 1.0) < 1e-5
    assert math.isnan(speedup["cand_missing"])
    assert math.isnan(speedup["cand_zero"])


def test_group_f_machine_readable_data_csv_21_columns(tmp_path):
    """Verify export_table generates data CSV with exact 21 columns in canonical order."""
    out_dir = str(tmp_path / "csv_test")
    df_raw = pd.DataFrame({
        "solver_a": [{
            "dataset": "ds1",
            "solver": "cadical",
            "status": "OPTIMAL",
            "solve_time": 2.5,
            "wall_time": 2.6,
            "encode_time": 0.1,
            "best_ub": 4,
            "par2": 2.5,
            "n_total": 1,
            "n_algorithmic": 1,
            "n_optimal": 1,
            "n_timeout": 0,
            "n_feasible": 0,
            "n_infeasible": 0,
            "n_size_limit": 0,
            "n_unavailable": 0,
            "n_license_error": 0,
            "n_error": 0,
            "n_memout": 0,
            "n_interrupted": 0,
            "n_unknown": 0,
        }]
    }, index=["ds1"])

    df_disp = pd.DataFrame({"solver_a": ["2.5000"]}, index=["ds1"])
    summary_rows = [("Solved", {"solver_a": "1"})]

    export_table(df_disp, summary_rows, title="Test Table", base_filename="test_tab", output_dirs=(out_dir,), df_raw=df_raw)

    data_csv_path = os.path.join(out_dir, "test_tab_data.csv")
    assert os.path.exists(data_csv_path)
    df_loaded = pd.read_csv(data_csv_path)
    assert list(df_loaded.columns) == DATA_CSV_COLUMNS


# ============================================================================
# GROUP G: PLOT KEY ALIGNMENT & LOG-SCALE SAFETY
# ============================================================================

def test_group_g_ablation_family_boxplots_base_key_separation(tmp_path):
    """Verify plot_ablation_family_boxplots does not overwrite pairs across solvers/encodings."""
    from analysis.make_plots import plot_ablation_family_boxplots

    records = []
    for n_val in [40, 50]:
        for solver in ["cadical", "kissat"]:
            ds = f"ds_{n_val}"
            configs = [
                ("none", {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False}, 10.0),
                ("rotation", {"sb_rotation": True, "sb_reflection": False, "heuristic_ub": False}, 8.0),
                ("rot_ref", {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": False}, 6.0),
                ("heur_ub", {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": True}, 7.0),
                ("all", {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True}, 4.0),
            ]
            for var_name, cfg, t in configs:
                records.append({
                    "dataset": ds,
                    "N": n_val,
                    "solver": solver,
                    "encoding": "order",
                    "relabel_seed": 0,
                    "status": "OPTIMAL",
                    "solve_time": t,
                    "config": cfg,
                    "groups": ["ablation_sb"],
                })

    plot_ablation_family_boxplots(records, base_filename="boxplot_test", output_dirs=(str(tmp_path),))
    assert os.path.exists(os.path.join(str(tmp_path), "boxplot_test.png"))
    assert os.path.exists(os.path.join(str(tmp_path), "boxplot_test.pdf"))


# ============================================================================
# GROUP H: EXPERIMENT RUNNER LIFECYCLE
# ============================================================================

def test_group_h_is_run_completed_lifecycle():
    """Verify is_run_completed correctly distinguishes retryable failures from conclusive proofs."""
    # Under retry_failed=False, all existing records are accepted
    assert is_run_completed({"status": "ERROR"}, retry_failed=False) is True
    assert is_run_completed({"status": "TIMEOUT"}, retry_failed=False) is True

    # Under retry_failed=True:
    # 1. OPTIMAL requires valid integer best_ub and no validation_error
    assert is_run_completed({"status": "OPTIMAL", "best_ub": 4}, retry_failed=True) is True
    assert is_run_completed({"status": "OPTIMAL", "best_ub": -1}, retry_failed=True) is False
    assert is_run_completed({"status": "OPTIMAL", "best_ub": 3.7}, retry_failed=True) is False
    assert is_run_completed({"status": "OPTIMAL", "best_ub": 4, "validation_error": True}, retry_failed=True) is False

    # 2. INFEASIBLE and SIZE_LIMIT are conclusive
    assert is_run_completed({"status": "INFEASIBLE"}, retry_failed=True) is True
    assert is_run_completed({"status": "SIZE_LIMIT"}, retry_failed=True) is True

    # 3. UNAVAILABLE and LICENSE_ERROR are permanent environment barriers
    assert is_run_completed({"status": "UNAVAILABLE"}, retry_failed=True) is True
    assert is_run_completed({"status": "LICENSE_ERROR"}, retry_failed=True) is True

    # 4. FEASIBLE preserves valid incumbent
    assert is_run_completed({"status": "FEASIBLE", "best_ub": 4}, retry_failed=True) is True
    assert is_run_completed({"status": "FEASIBLE", "best_ub": -1}, retry_failed=True) is False

    # 5. Transient failures must be retried
    assert is_run_completed({"status": "TIMEOUT"}, retry_failed=True) is False
    assert is_run_completed({"status": "ERROR"}, retry_failed=True) is False
    assert is_run_completed({"status": "MEMOUT"}, retry_failed=True) is False
    assert is_run_completed({"status": "INTERRUPTED"}, retry_failed=True) is False


def test_group_h_load_existing_runs_preserves_earlier_optimal(tmp_path):
    """Verify load_existing_runs preserves earlier completed OPTIMAL run when a retried crash appears later."""
    jsonl_file = tmp_path / "runs.jsonl"
    r_optimal = {
        "dataset": "ds1", "solver": "cadical195", "status": "OPTIMAL", "solve_time": 2.5, "best_ub": 4, "rep": 1
    }
    r_retry_fail = {
        "dataset": "ds1", "solver": "cadical195", "status": "ERROR", "solve_time": None, "best_ub": None, "rep": 1
    }
    lines = [json.dumps(r_optimal), json.dumps(r_retry_fail)]
    jsonl_file.write_text("\n".join(lines), encoding="utf-8")

    loaded = load_existing_runs(str(jsonl_file), retry_failed=True)
    key = get_canonical_key(r_optimal)
    assert (key, 1) in loaded
    assert loaded[(key, 1)]["status"] == "OPTIMAL"
    assert loaded[(key, 1)]["best_ub"] == 4
