import pytest
import os
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

def test_cross_validation_pass_and_fail():
    matching_records = [
        {"dataset": "g1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "groups": ["solvers"]},
        {"dataset": "g1", "solver": "glucose4", "status": "OPTIMAL", "best_ub": 3, "groups": ["solvers"]},
        {"dataset": "g2", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 5, "groups": ["solvers"]},
    ]
    cross_validate_optima(matching_records)

    conflicting_records = [
        {"dataset": "g1", "solver": "cadical195", "status": "OPTIMAL", "best_ub": 3, "groups": ["solvers"]},
        {"dataset": "g1", "solver": "glucose4", "status": "OPTIMAL", "best_ub": 4, "groups": ["solvers"]},
    ]
    with pytest.raises(ValueError, match="conflicting optima"):
        cross_validate_optima(conflicting_records)

def test_canonical_key_consistency():
    rec_sat = {
        "dataset": "bus14",
        "solver": "cadical195",
        "config": {"sb_rotation": True, "sb_reflection": False, "heuristic_ub": True},
    }
    key_sat = get_canonical_key(rec_sat)
    assert key_sat == ("bus14", "sat", "cadical195", (True, False, True))

    rec_cpsat = {
        "dataset": "bus14",
        "solver": "cp_sat",
        "config": {"num_workers": 1},
    }
    key_cpsat = get_canonical_key(rec_cpsat)
    assert key_cpsat == ("bus14", "cpsat", "cp_sat", (1,))

    rec_cplex = {
        "dataset": "bus14",
        "solver": "cplex",
        "config": {},
    }
    key_cplex = get_canonical_key(rec_cplex)
    assert key_cplex == ("bus14", "cplex", "cplex", ())
