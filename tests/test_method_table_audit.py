import pytest

from analysis.make_tables import EXACT_SOLVER_ALIASES, make_table_methods, normalize_solver_alias


@pytest.mark.parametrize("alias,canonical", sorted(EXACT_SOLVER_ALIASES.items()))
def test_all_exact_solver_aliases_normalize_without_collapsing_engine_families(alias, canonical):
    assert normalize_solver_alias(alias) == canonical


def test_methods_table_keeps_five_exact_engines_separate_and_selects_best_sat_encoding(tmp_path):
    records = [
        {
            "dataset": "graph_n20", "N": 20, "num_edges": 25,
            "solver": "cadical195", "method": "sat",
            "status": "OPTIMAL", "best_ub": 5, "solve_time": 2.0,
            "config": {"encoding": "order", "solving_mode": "incremental", "eo_strategy": "commander",
                       "sb_rotation": True, "sb_reflection": True, "heuristic_ub": True},
            "groups": ["encodings"],
        },
        {
            "dataset": "graph_n20", "N": 20, "num_edges": 25,
            "solver": "cadical195", "method": "sat",
            "status": "OPTIMAL", "best_ub": 5, "solve_time": 1.0,
            "config": {"encoding": "pairwise", "solving_mode": "incremental", "eo_strategy": "commander",
                       "sb_rotation": True, "sb_reflection": True, "heuristic_ub": True},
            "groups": ["encodings"],
        },
        {
            "dataset": "graph_n20", "N": 20, "num_edges": 25,
            "solver": "ortools", "status": "OPTIMAL", "best_ub": 5, "solve_time": 3.0,
            "groups": ["exact_solvers"],
        },
        {
            "dataset": "graph_n20", "N": 20, "num_edges": 25,
            "solver": "ibm_cplex", "status": "SIZE_LIMIT", "best_ub": None, "solve_time": 0.01,
            "groups": ["exact_solvers"],
        },
        {
            "dataset": "graph_n20", "N": 20, "num_edges": 25,
            "solver": "docplex_cp", "status": "UNAVAILABLE", "best_ub": None, "solve_time": None,
            "groups": ["exact_solvers"],
        },
        {
            "dataset": "graph_n20", "N": 20, "num_edges": 25,
            "solver": "grb", "status": "LICENSE_ERROR", "best_ub": None, "solve_time": None,
            "groups": ["exact_solvers"],
        },
    ]
    display, summary, raw = make_table_methods(records, output_dirs=(str(tmp_path),))
    row = display.loc["graph_n20"]
    assert row["sat_best"] == "1.0000"
    assert row["cpsat"] == "3.0000"
    assert row["cplex"] == "SL"
    assert row["cplex_cp"] == "UNAV"
    assert row["gurobi"] == "LIC"

    raw_row = raw.loc["graph_n20"]
    assert raw_row["sat_best"]["config"]["encoding"] == "pairwise"
    assert raw_row["cpsat"]["status"] == "OPTIMAL"
    assert raw_row["cplex"]["status"] == "SIZE_LIMIT"
    assert raw_row["cplex_cp"]["status"] == "UNAVAILABLE"
    assert raw_row["gurobi"]["status"] == "LICENSE_ERROR"

    summary_map = {name: values for name, values in summary}
    assert summary_map["PAR2 runs (n)"]["cplex"] == "0"
    assert summary_map["PAR2 runs (n)"]["cplex_cp"] == "0"
    assert summary_map["PAR2 runs (n)"]["gurobi"] == "0"
    assert summary_map["PAR2 runs (n)"]["sat_best"] == "1"
    assert summary_map["PAR2 runs (n)"]["cpsat"] == "1"


def test_size_limit_unavailable_and_license_error_are_not_relabelled_as_timeout(tmp_path):
    records = [
        {"dataset": "graph_n20", "N": 20, "num_edges": 25,
         "solver": "cplex", "status": "SIZE_LIMIT", "groups": ["exact_solvers"]},
        {"dataset": "graph_n20", "N": 20, "num_edges": 25,
         "solver": "cplex_cp", "status": "UNAVAILABLE", "groups": ["exact_solvers"]},
        {"dataset": "graph_n20", "N": 20, "num_edges": 25,
         "solver": "gurobi", "status": "LICENSE_ERROR", "groups": ["exact_solvers"]},
    ]
    display, _, raw = make_table_methods(records, output_dirs=(str(tmp_path),))
    row = display.loc["graph_n20"]
    assert row["cplex"] == "SL"
    assert row["cplex_cp"] == "UNAV"
    assert row["gurobi"] == "LIC"
    assert raw.loc["graph_n20", "cplex"]["status"] == "SIZE_LIMIT"
    assert raw.loc["graph_n20", "cplex_cp"]["status"] == "UNAVAILABLE"
    assert raw.loc["graph_n20", "gurobi"]["status"] == "LICENSE_ERROR"
