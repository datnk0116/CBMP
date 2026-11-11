import pytest
import os
import sys
import json
import yaml
import numpy as np
from pathlib import Path

from experiments.run_experiments import (
    build_experiment_plan,
    get_canonical_key,
    format_task_flags,
    filter_datasets_for_group,
)
from analysis.make_tables import (
    aggregate_repetitions,
    cross_validate_optima,
    format_cell_value,
)

def test_filter_datasets_max_n():
    all_datasets = [
        {"name": "d14", "N": 14, "M": 20},
        {"name": "d30", "N": 30, "M": 39},
        {"name": "d100", "N": 100, "M": 150},
        {"name": "d200", "N": 200, "M": 300},
        {"name": "d250", "N": 250, "M": 400},
        {"name": "d494", "N": 494, "M": 580},
    ]

    defaults = {
        "filters": {
            "min_N": 1,
            "max_N": None,
        }
    }

    # 1. Group with max_n = 200 and datasets: all
    grp_cfg_1 = {"datasets": "all", "max_n": 200}
    res_1 = filter_datasets_for_group(all_datasets, grp_cfg_1, defaults)
    names_1 = [d["name"] for d in res_1]
    assert names_1 == ["d14", "d30", "d100", "d200"]
    assert "d250" not in names_1
    assert "d494" not in names_1

    # 2. Group with max_n = 100
    grp_cfg_2 = {"max_n": 100}
    res_2 = filter_datasets_for_group(all_datasets, grp_cfg_2, defaults)
    names_2 = [d["name"] for d in res_2]
    assert names_2 == ["d14", "d30", "d100"]

    # 3. Group with explicit dataset list and max_n = 200
    grp_cfg_3 = {"datasets": ["d14", "d30", "d494"], "max_n": 200}
    res_3 = filter_datasets_for_group(all_datasets, grp_cfg_3, defaults)
    names_3 = [d["name"] for d in res_3]
    # d494 is in the list but N=494 > 200, so it must be filtered out
    assert names_3 == ["d14", "d30"]

    # 4. Fallback to default max_N when group specifies neither datasets nor max_n
    defaults_with_max = {"filters": {"min_N": 1, "max_N": 30}}
    grp_cfg_4 = {}
    res_4 = filter_datasets_for_group(all_datasets, grp_cfg_4, defaults_with_max)
    names_4 = [d["name"] for d in res_4]
    assert names_4 == ["d14", "d30"]

    # 5. CLI override cli_max_n takes precedence over group max_n
    grp_cfg_5 = {"max_n": 200}
    res_5 = filter_datasets_for_group(all_datasets, grp_cfg_5, defaults, cli_max_n=30)
    names_5 = [d["name"] for d in res_5]
    assert names_5 == ["d14", "d30"]

def test_build_experiment_plan_deduplication():
    dummy_config = {
        "defaults": {
            "timeout": 100,
            "repetitions": 1,
        },
        "groups": {
            "per_dataset": {
                "datasets": "all",
                "solver": "cadical195",
                "sb_rotation": True,
                "sb_reflection": True,
                "heuristic_ub": True,
                "timeout": 100,
            },
            "solvers": {
                "datasets": "all",
                "sb_rotation": True,
                "sb_reflection": True,
                "heuristic_ub": True,
                "timeout": 100,
                "solvers": ["cadical195", "glucose4"],
            },
            "methods": {
                "datasets": "all",
                "timeout": 100,
                "methods": [
                    {
                        "name": "sat_best",
                        "type": "sat",
                        "solver": "cadical195",
                        "sb_rotation": True,
                        "sb_reflection": True,
                        "heuristic_ub": True,
                    },
                    {
                        "name": "cpsat",
                        "type": "cpsat",
                        "num_workers": 1,
                    },
                ],
            },
            "ablation": {
                "datasets": "all",
                "repetitions": 3,
                "solver": "cadical195",
                "timeout": 100,
                "variants": [
                    {
                        "name": "none",
                        "sb_rotation": False,
                        "sb_reflection": False,
                        "heuristic_ub": False,
                    },
                    {
                        "name": "all",
                        "sb_rotation": True,
                        "sb_reflection": True,
                        "heuristic_ub": True,
                    },
                ],
            },
        },
    }

    datasets = [{"name": "test_graph", "N": 10, "M": 15}]

    # Without deduplication: 1 (per_dataset) + 2 (solvers) + 2 (methods) + 2 (ablation) = 7 configs.
    # With deduplication:
    # 1. cadical195 (rot=True, ref=True, heur=True) is shared across ALL 4 groups!
    #    repetitions for this task = max(1, 1, 1, 3) = 3
    # 2. glucose4 (rot=True, ref=True, heur=True) [solvers, rep=1]
    # 3. cpsat [methods, rep=1]
    # 4. cadical195 (rot=False, ref=False, heur=False) [ablation, rep=3]
    # Total unique tasks = 4
    plan, group_stats = build_experiment_plan(dummy_config, all_datasets=datasets, selected_group="all")
    assert len(plan) == 4

    # Find the shared task
    shared_task = next(
        t for t in plan
        if t["solver"] == "cadical195" and t["flags"] == (True, True, True)
    )
    assert set(shared_task["groups"]) == {"per_dataset", "solvers", "methods", "ablation"}
    assert shared_task["repetitions"] == 3

    # Check group_stats
    assert group_stats["per_dataset"]["runs_count"] == 1
    assert group_stats["solvers"]["runs_count"] == 2
    assert group_stats["methods"]["runs_count"] == 2
    assert group_stats["ablation"]["runs_count"] == 6  # 2 variants x 3 reps

def test_aggregate_repetitions():
    records = [
        {
            "dataset": "g1",
            "solver": "cadical195",
            "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True},
            "status": "OPTIMAL",
            "best_ub": 3,
            "solve_time": 1.0,
            "wall_time": 1.2,
            "encode_time": 0.1,
            "num_iterations": 3,
            "groups": ["ablation"],
            "rep": 1,
        },
        {
            "dataset": "g1",
            "solver": "cadical195",
            "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True},
            "status": "OPTIMAL",
            "best_ub": 3,
            "solve_time": 2.0,
            "wall_time": 2.3,
            "encode_time": 0.2,
            "num_iterations": 3,
            "groups": ["solvers"],
            "rep": 2,
        },
        {
            "dataset": "g1",
            "solver": "cadical195",
            "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True},
            "status": "OPTIMAL",
            "best_ub": 3,
            "solve_time": 1.5,
            "wall_time": 1.7,
            "encode_time": 0.15,
            "num_iterations": 3,
            "groups": ["per_dataset"],
            "rep": 3,
        },
    ]

    agg = aggregate_repetitions(records)
    assert len(agg) == 1
    r = agg[0]
    assert r["solve_time"] == pytest.approx(1.5)
    assert r["wall_time"] == pytest.approx(1.7)
    assert r["encode_time"] == pytest.approx(0.15)
    assert set(r["groups"]) == {"ablation", "solvers", "per_dataset"}
    assert r["status"] == "OPTIMAL"
    assert r["best_ub"] == 3

def test_cross_validation_two_solvers_same_opt(tmp_path):
    # (a) Two solvers with same opt -> verified
    out_dir = str(tmp_path / "tables")
    matching_records = [
        {"dataset": "g1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "groups": ["solvers"]},
        {"dataset": "g1", "solver": "glucose4", "status": "OPTIMAL", "best_ub": 3, "groups": ["solvers"]},
    ]
    df = cross_validate_optima(matching_records, output_dir=out_dir)
    assert len(df) == 2
    assert all(df["validation_status"] == "verified")
    assert all(df["best_ub"] == 3)
    assert os.path.isfile(os.path.join(out_dir, "cross_validation.csv"))

def test_cross_validation_conflict_error(tmp_path):
    # (b) Conflicting optima -> error
    out_dir = str(tmp_path / "tables")
    conflicting_records = [
        {"dataset": "g1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "groups": ["solvers"]},
        {"dataset": "g1", "solver": "glucose4", "status": "OPTIMAL", "best_ub": 4, "groups": ["solvers"]},
    ]
    with pytest.raises(ValueError, match="conflicting optima"):
        cross_validate_optima(conflicting_records, output_dir=out_dir)
    assert os.path.isfile(os.path.join(out_dir, "cross_validation.csv"))
    import pandas as pd
    df = pd.read_csv(os.path.join(out_dir, "cross_validation.csv"))
    assert all(df["validation_status"] == "conflict")

def test_cross_validation_single_source_5_reps(tmp_path, capsys):
    # (c) Only 1 key repeated 5 reps -> single_source and warning printed
    out_dir = str(tmp_path / "tables")
    records = [
        {
            "dataset": "g1",
            "solver": "cadical195",
            "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True},
            "status": "OPTIMAL",
            "best_ub": 3,
            "rep": i,
            "groups": ["solvers"],
        }
        for i in range(1, 6)
    ]
    df = cross_validate_optima(records, output_dir=out_dir)
    assert len(df) == 1
    assert df.iloc[0]["validation_status"] == "single_source"
    assert df.iloc[0]["num_distinct_sources"] == 1
    out = capsys.readouterr().out
    assert "[WARNING]" in out and "single_source" in out
    assert "All optima match identically" not in out
    assert os.path.isfile(os.path.join(out_dir, "cross_validation.csv"))

def test_cross_validation_incumbent_smaller_than_opt_error():
    # (d) Incumbent < proven opt -> error
    records = [
        {"dataset": "g1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "groups": ["solvers"]},
        {"dataset": "g1", "solver": "glucose4", "status": "TIMEOUT", "best_ub": 2, "groups": ["solvers"]},
    ]
    with pytest.raises(ValueError, match="Invalid incumbent"):
        cross_validate_optima(records)

def test_cross_validation_none_vs_all_two_keys(tmp_path):
    # (e) none vs all with same cadical195 counted as 2 distinct keys
    out_dir = str(tmp_path / "tables")
    records = [
        {
            "dataset": "g1",
            "solver": "cadical195",
            "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False},
            "status": "OPTIMAL",
            "best_ub": 3,
        },
        {
            "dataset": "g1",
            "solver": "cadical195",
            "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True},
            "status": "OPTIMAL",
            "best_ub": 3,
        },
    ]
    df = cross_validate_optima(records, output_dir=out_dir)
    assert len(df) == 2
    assert df.iloc[0]["num_distinct_sources"] == 2
    assert all(df["validation_status"] == "verified")

def test_cross_validation_cli_exit_code(tmp_path):
    import subprocess
    import json

    out_dir = str(tmp_path / "tables")

    # Matching jsonl -> exit code 0
    match_file = tmp_path / "match.jsonl"
    with open(match_file, "w") as f:
        f.write(json.dumps({"dataset": "g1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "groups": ["solvers"]}) + "\n")
        f.write(json.dumps({"dataset": "g1", "solver": "glucose4", "status": "OPTIMAL", "best_ub": 3, "groups": ["solvers"]}) + "\n")
    res_ok = subprocess.run([sys.executable, "analysis/make_tables.py", str(match_file), out_dir], capture_output=True, text=True)
    assert res_ok.returncode == 0
    assert os.path.isfile(os.path.join(out_dir, "cross_validation.csv"))

    # Conflicting jsonl -> exit code != 0
    conflict_file = tmp_path / "conflict.jsonl"
    with open(conflict_file, "w") as f:
        f.write(json.dumps({"dataset": "g1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "groups": ["solvers"]}) + "\n")
        f.write(json.dumps({"dataset": "g1", "solver": "glucose4", "status": "OPTIMAL", "best_ub": 4, "groups": ["solvers"]}) + "\n")
    res_err = subprocess.run([sys.executable, "analysis/make_tables.py", str(conflict_file), out_dir], capture_output=True, text=True)
    assert res_err.returncode != 0
    assert "conflicting optima" in res_err.stderr.lower() or "conflicting optima" in res_err.stdout.lower()

def test_canonical_key_consistency():
    rec_sat = {
        "dataset": "bus14",
        "solver": "cadical195",
        "config": {"sb_rotation": True, "sb_reflection": False, "heuristic_ub": True},
    }
    key_sat = get_canonical_key(rec_sat)
    assert key_sat == ("bus14", "sat", "cadical195", (True, False, True), None)

    rec_sat_seed0 = {
        "dataset": "bus14",
        "solver": "cadical195",
        "relabel_seed": 0,
        "config": {"sb_rotation": True, "sb_reflection": False, "heuristic_ub": True},
    }
    key_sat_seed0 = get_canonical_key(rec_sat_seed0)
    assert key_sat_seed0 == ("bus14", "sat", "cadical195", (True, False, True), 0)
    assert key_sat != key_sat_seed0

    rec_cpsat = {
        "dataset": "bus14",
        "solver": "cp_sat",
        "config": {"num_workers": 1},
    }
    key_cpsat = get_canonical_key(rec_cpsat)
    assert key_cpsat == ("bus14", "cpsat", "cp_sat", (1,), None)

    rec_cplex = {
        "dataset": "bus14",
        "solver": "cplex",
        "config": {},
    }
    key_cplex = get_canonical_key(rec_cplex)
    assert key_cplex == ("bus14", "cplex", "cplex", (), None)

def test_main_optimal_exit(tmp_path):
    from main import main
    out_file = tmp_path / "main_run.json"
    stats = main(
        dataset="30_bus",
        sb_rotation=True,
        sb_reflection=True,
        use_heuristic_initial_ub=True,
        timeout=10.0,
        json_out=str(out_file),
    )
    assert stats["status"] == "OPTIMAL"
    assert stats["best_ub"] == 4
    assert out_file.exists()

def test_aggregate_repetitions_mixed_optimal_timeout():
    """
    Test 4 repetitions: OPTIMAL 10s, OPTIMAL 20s, TIMEOUT 30s, TIMEOUT 600s.
    Status must be PARTIAL (n_optimal=2, n_total=4).
    Median time must be computed only over OPTIMAL reps: median(10s, 20s) = 15s.
    PAR2: each non-optimal rep gets 2T. (10 + 20 + 2*600 + 2*600) / 4 = 607.5s.
    Table display must be 'k/n' and median time of optimal reps with '*': '2/4 15.0000*'.
    """
    records = [
        {
            "dataset": "g1",
            "solver": "cadical195",
            "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True, "time_limit": 600.0},
            "status": "OPTIMAL",
            "best_ub": 3,
            "solve_time": 10.0,
            "rep": 1,
        },
        {
            "dataset": "g1",
            "solver": "cadical195",
            "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True, "time_limit": 600.0},
            "status": "OPTIMAL",
            "best_ub": 3,
            "solve_time": 20.0,
            "rep": 2,
        },
        {
            "dataset": "g1",
            "solver": "cadical195",
            "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True, "time_limit": 600.0},
            "status": "TIMEOUT",
            "best_ub": 4,
            "solve_time": 30.0,
            "rep": 3,
        },
        {
            "dataset": "g1",
            "solver": "cadical195",
            "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True, "time_limit": 600.0},
            "status": "TIMEOUT",
            "best_ub": 4,
            "solve_time": 600.0,
            "rep": 4,
        },
    ]

    agg = aggregate_repetitions(records)
    assert len(agg) == 1
    r = agg[0]
    assert r["status"] == "PARTIAL"
    assert r["n_optimal"] == 2
    assert r["n_total"] == 4
    assert r.get("mixed") is True or r.get("is_mixed") is True
    # Median of OPTIMAL reps only: (10.0 + 20.0) / 2 = 15.0
    assert r["solve_time"] == pytest.approx(15.0)
    assert r["best_ub"] == 3
    # PAR2: (10 + 20 + 1200 + 1200) / 4 = 607.5
    assert r["par2"] == pytest.approx(607.5)

    # Format cell value check
    assert format_cell_value(r) == "2/4 15.0000*"
    assert format_cell_value("PARTIAL", 15.0, 3, n_optimal=2, n_total=4) == "2/4 15.0000*"

def test_compute_summary_stats_size_limit_excluded_from_par2():
    """
    Verify that SIZE_LIMIT is not added to PAR2 as timeout,
    and is reported separately in sl_counts.
    """
    import pandas as pd
    from analysis.make_tables import compute_summary_stats

    df_raw = pd.DataFrame({
        "cplex": {
            "d1": {"status": "OPTIMAL", "solve_time": 2.0, "config": {"time_limit": 10.0}},
            "d2": {"status": "SIZE_LIMIT", "solve_time": 0.0, "config": {"time_limit": 10.0}},
            "d3": {"status": "TIMEOUT", "solve_time": 10.0, "config": {"time_limit": 10.0}},
        }
    })

    solved, sl, par2, spdup = compute_summary_stats(df_raw, ["cplex"], timeout=10.0)
    assert solved["cplex"] == 1
    assert sl["cplex"] == 1
    # PAR2 has d1 (2.0s) and d3 (2*10.0 = 20.0s), d2 (SIZE_LIMIT) is excluded
    # Mean of [2.0, 20.0] = 11.0s (not including 2*10 for d2 which would make it 14.0s)
    assert par2["cplex"] == pytest.approx(11.0)

def test_resume_rejects_git_commit_mismatch(tmp_path):
    from experiments.run_experiments import run_experiments

    runs_file = tmp_path / "runs.jsonl"
    fake_record = {
        "dataset": "14_bus",
        "N": 14,
        "num_edges": 20,
        "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False},
        "solver": "cadical195",
        "status": "OPTIMAL",
        "best_ub": 3,
        "solve_time": 0.01,
        "git_commit": "0000000000000000000000000000000000000000",
        "git_dirty": False,
        "groups": ["ablation"],
        "rep": 1,
    }
    runs_file.write_text(json.dumps(fake_record) + "\n")

    # Resume without force_resume must raise ValueError due to commit mismatch
    with pytest.raises(ValueError, match="Git commit mismatch"):
        run_experiments(
            output_dir=str(tmp_path),
            group="ablation",
            dataset_filter="14_bus",
            resume=True,
            force_resume=False,
            allow_dirty=True,
            timeout=5.0,
        )

    # With force_resume=True, it should proceed (and skip the existing optimal task)
    run_experiments(
        output_dir=str(tmp_path),
        group="ablation",
        dataset_filter="14_bus",
        resume=True,
        force_resume=True,
        allow_dirty=True,
        timeout=5.0,
    )

def test_resume_and_retry_failed_behavior(tmp_path):
    from experiments.run_experiments import load_existing_runs, get_canonical_key

    runs_file = tmp_path / "runs.jsonl"
    opt_rec = {
        "dataset": "14_bus",
        "N": 14,
        "num_edges": 20,
        "config": {"sb_rotation": False, "sb_reflection": False, "heuristic_ub": False},
        "solver": "cadical195",
        "status": "OPTIMAL",
        "best_ub": 3,
        "solve_time": 0.01,
        "groups": ["ablation"],
        "rep": 1,
    }
    timeout_rec = {
        "dataset": "14_bus",
        "N": 14,
        "num_edges": 20,
        "config": {"sb_rotation": True, "sb_reflection": False, "heuristic_ub": False},
        "solver": "cadical195",
        "status": "TIMEOUT",
        "best_ub": -1,
        "solve_time": 600.0,
        "groups": ["ablation"],
        "rep": 1,
    }
    sl_rec = {
        "dataset": "494_bus",
        "N": 494,
        "num_edges": 580,
        "config": {},
        "solver": "cplex",
        "status": "SIZE_LIMIT",
        "best_ub": -1,
        "solve_time": 0.1,
        "groups": ["methods"],
        "rep": 1,
    }
    with open(runs_file, "w") as f:
        f.write(json.dumps(opt_rec) + "\n")
        f.write(json.dumps(timeout_rec) + "\n")
        f.write(json.dumps(sl_rec) + "\n")

    # Behavior 1: Default resume (retry_failed=False) retains all records (all skipped)
    loaded_default = load_existing_runs(str(runs_file), retry_failed=False)
    assert len(loaded_default) == 3
    assert (get_canonical_key(opt_rec), 1) in loaded_default
    assert (get_canonical_key(timeout_rec), 1) in loaded_default
    assert (get_canonical_key(sl_rec), 1) in loaded_default

    # Behavior 2: With retry_failed=True, TIMEOUT is omitted so it will be retried
    loaded_retry = load_existing_runs(str(runs_file), retry_failed=True)
    assert len(loaded_retry) == 2
    assert (get_canonical_key(opt_rec), 1) in loaded_retry
    assert (get_canonical_key(sl_rec), 1) in loaded_retry
    assert (get_canonical_key(timeout_rec), 1) not in loaded_retry

def test_runner_rejects_dirty_without_allow_dirty(monkeypatch, tmp_path):
    from experiments import run_experiments as runner

    # Mock dirty repository status
    monkeypatch.setattr(runner, "get_git_commit_info", lambda: ("testcommit123", True))

    with pytest.raises(ValueError, match="Working directory has uncommitted changes"):
        runner.run_experiments(
            output_dir=str(tmp_path),
            group="ablation",
            dataset_filter="14_bus",
            allow_dirty=False,
            timeout=5.0,
        )

    # When allow_dirty=True, it should proceed
    runner.run_experiments(
        output_dir=str(tmp_path),
        group="ablation",
        dataset_filter="14_bus",
        allow_dirty=True,
        timeout=5.0,
    )

def test_metadata_records_relative_config_path(tmp_path):
    from experiments import run_experiments as runner

    runner.run_experiments(
        output_dir=str(tmp_path),
        group="ablation",
        dataset_filter="14_bus",
        allow_dirty=True,
        record_metadata=True,
        timeout=5.0,
    )

    env_path = tmp_path / "env.json"
    assert env_path.exists()
    env_data = json.loads(env_path.read_text())
    assert "config_path" in env_data
    # Must be relative (not absolute starting with / or C:\)
    assert not os.path.isabs(env_data["config_path"])
    assert env_data["config_path"] == "experiments/configs.yaml"

    runs_file = tmp_path / "runs.jsonl"
    assert runs_file.exists()
    first_line = json.loads(runs_file.read_text().splitlines()[0])
    assert first_line.get("_type") == "metadata"
    assert "config_path" in first_line
    assert not os.path.isabs(first_line["config_path"])
    assert first_line["config_path"] == "experiments/configs.yaml"
