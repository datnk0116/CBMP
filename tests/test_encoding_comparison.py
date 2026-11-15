from analysis.make_tables import make_table_encoding_comparison


def _encoding_record(encoding, status, solve_time=None, best_ub=5, dataset="graph_n20"):
    return {
        "dataset": dataset,
        "N": 20,
        "num_edges": 25,
        "solver": "cadical195",
        "status": status,
        "best_ub": best_ub,
        "solve_time": solve_time,
        "wall_time": None if solve_time is None else solve_time + 0.1,
        "encode_time": 0.02,
        "num_vars": 500,
        "num_clauses": 1800,
        "config": {
            "sb_rotation": True,
            "sb_reflection": True,
            "heuristic_ub": True,
            "encoding": encoding,
            "solving_mode": "incremental",
            "eo_strategy": "commander",
        },
        "groups": ["encodings"],
        "rep": 1,
    }


def test_encoding_table_preserves_status_and_pairs_only_valid_optima(tmp_path):
    records = [
        _encoding_record("order", "OPTIMAL", 1.0, 5),
        _encoding_record("pairwise", "OPTIMAL", 2.0, 5),
        _encoding_record("order", "TIMEOUT", None, 6, dataset="graph_n30"),
        _encoding_record("pairwise", "FEASIBLE", 4.0, 6, dataset="graph_n30"),
    ]
    detail, paired = make_table_encoding_comparison(records, output_dirs=(str(tmp_path),))

    assert set(detail["encoding"]) == {"order", "pairwise"}
    assert set(detail["status"]) == {"OPTIMAL", "TIMEOUT", "FEASIBLE"}

    n20 = paired.loc["graph_n20"]
    assert n20["n_both_encodings"] == 1
    assert n20["n_paired_optimal_runtime"] == 1
    assert n20["n_equal_optimum_pairs"] == 1
    assert n20["median_pairwise_over_order"] == 2.0

    n30 = paired.loc["graph_n30"]
    assert n30["n_timeout_pairs"] == 1
    assert n30["n_feasible_pairs"] == 1
    assert n30["n_paired_optimal_runtime"] == 0

    assert (tmp_path / "table_encoding_comparison.csv").is_file()
    assert (tmp_path / "table_encoding_comparison.tex").is_file()
    assert (tmp_path / "table_encoding_comparison_pairs.csv").is_file()
    assert (tmp_path / "table_encoding_comparison_pairs.tex").is_file()


def test_encoding_table_skips_without_encoding_records(tmp_path):
    result = make_table_encoding_comparison(
        [{"dataset": "graph_n20", "solver": "cadical195", "status": "OPTIMAL"}],
        output_dirs=(str(tmp_path),),
    )
    assert result is None
    assert list(tmp_path.iterdir()) == []
