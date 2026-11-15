import pandas as pd
import pytest

from analysis.make_tables import (
    DuplicateConflictError,
    make_table_encoding_comparison,
)


def _encoding_record(encoding, status, solve_time=None, best_ub=5, dataset="graph_n20",
                     seed=None, rep=1, sb_rotation=True, sb_reflection=True,
                     heuristic_ub=True, time_limit=600):
    return {
        "dataset": dataset, "N": 20, "num_edges": 25,
        "solver": "cadical195", "method": "sat", "relabel_seed": seed,
        "rep": rep, "status": status, "best_ub": best_ub,
        "solve_time": solve_time,
        "wall_time": None if solve_time is None else solve_time + 0.1,
        "encode_time": 0.02, "num_vars": 500, "num_clauses": 1800,
        "time_limit": time_limit,
        "config": {
            "sb_rotation": sb_rotation, "sb_reflection": sb_reflection,
            "heuristic_ub": heuristic_ub, "encoding": encoding,
            "solving_mode": "incremental", "eo_strategy": "commander",
            "time_limit": time_limit, "relabel_seed": seed,
        },
        "groups": ["encodings"],
    }


def test_encoding_table_equal_optimum_pair_has_valid_ratio(tmp_path):
    records = [
        _encoding_record("order", "OPTIMAL", 1.0, 5),
        _encoding_record("pairwise", "OPTIMAL", 2.0, 5),
    ]
    detail, paired = make_table_encoding_comparison(records, output_dirs=(str(tmp_path),))
    assert set(detail["encoding"]) == {"order", "pairwise"}
    row = paired.loc["graph_n20"]
    assert row["n_both_encodings"] == 1
    assert row["n_matched_repetition_pairs"] == 1
    assert row["n_equal_optimum_pairs"] == 1
    assert row["n_paired_optimal_runtime"] == 1
    assert row["median_pairwise_over_order"] == 2.0


def test_exact_duplicate_run_is_deduplicated_not_counted_as_repetition(tmp_path):
    order = _encoding_record("order", "OPTIMAL", 1.0, 5)
    pairwise = _encoding_record("pairwise", "OPTIMAL", 2.0, 5)
    detail, paired = make_table_encoding_comparison(
        [order, dict(order), pairwise], output_dirs=(str(tmp_path),)
    )
    assert detail[detail["encoding"] == "order"].iloc[0]["n_total"] == 1
    assert paired.loc["graph_n20", "n_matched_repetition_pairs"] == 1


def test_duplicate_run_with_conflicting_optimum_raises(tmp_path):
    records = [
        _encoding_record("order", "OPTIMAL", 1.0, 5),
        _encoding_record("order", "OPTIMAL", 1.1, 6),
        _encoding_record("pairwise", "OPTIMAL", 2.0, 5),
    ]
    with pytest.raises(DuplicateConflictError, match="Conflicting OPTIMAL bandwidths"):
        make_table_encoding_comparison(records, output_dirs=(str(tmp_path),))


def test_duplicate_run_with_conflicting_status_or_timing_raises(tmp_path):
    records = [
        _encoding_record("order", "OPTIMAL", 1.0, 5),
        _encoding_record("order", "TIMEOUT", None, 5),
        _encoding_record("pairwise", "OPTIMAL", 2.0, 5),
    ]
    with pytest.raises(DuplicateConflictError, match="Conflicting status or measurements"):
        make_table_encoding_comparison(records, output_dirs=(str(tmp_path),))


def test_different_seed_is_not_paired(tmp_path):
    records = [
        _encoding_record("order", "OPTIMAL", 1.0, 5, seed=0),
        _encoding_record("pairwise", "OPTIMAL", 2.0, 5, seed=1),
    ]
    _, paired = make_table_encoding_comparison(records, output_dirs=(str(tmp_path),))
    row = paired.loc["graph_n20"]
    assert row["n_both_encodings"] == 0
    assert row["n_order_only_configs"] == 1
    assert row["n_pairwise_only_configs"] == 1
    assert row["n_paired_optimal_runtime"] == 0


def test_different_symmetry_settings_are_not_paired(tmp_path):
    records = [
        _encoding_record("order", "OPTIMAL", 1.0, 5, sb_reflection=False),
        _encoding_record("pairwise", "OPTIMAL", 2.0, 5, sb_reflection=True),
    ]
    _, paired = make_table_encoding_comparison(records, output_dirs=(str(tmp_path),))
    assert paired.loc["graph_n20", "n_both_encodings"] == 0
    assert paired.loc["graph_n20", "n_matched_repetition_pairs"] == 0


def test_different_time_limits_are_not_paired(tmp_path):
    records = [
        _encoding_record("order", "OPTIMAL", 1.0, 5, time_limit=300),
        _encoding_record("pairwise", "OPTIMAL", 2.0, 5, time_limit=600),
    ]
    _, paired = make_table_encoding_comparison(records, output_dirs=(str(tmp_path),))
    assert paired.loc["graph_n20", "n_both_encodings"] == 0


def test_one_encoding_missing_is_reported_as_unmatched(tmp_path):
    _, paired = make_table_encoding_comparison(
        [_encoding_record("order", "OPTIMAL", 1.0, 5)],
        output_dirs=(str(tmp_path),),
    )
    row = paired.loc["graph_n20"]
    assert row["n_order_only_configs"] == 1
    assert row["n_unmatched_control_configs"] == 1
    assert row["n_order_only_repetitions"] == 1
    assert row["n_paired_optimal_runtime"] == 0


def test_both_timeout_records_are_counted(tmp_path):
    _, paired = make_table_encoding_comparison(
        [
            _encoding_record("order", "TIMEOUT", None, 8),
            _encoding_record("pairwise", "TIMEOUT", None, 8),
        ],
        output_dirs=(str(tmp_path),),
    )
    row = paired.loc["graph_n20"]
    assert row["n_timeout_pairs"] == 1
    assert row["n_paired_optimal_runtime"] == 0
    assert pd.isna(row["median_pairwise_over_order"])


def test_feasible_vs_optimal_is_not_counted_as_paired_optimal(tmp_path):
    _, paired = make_table_encoding_comparison(
        [
            _encoding_record("order", "OPTIMAL", 1.0, 5),
            _encoding_record("pairwise", "FEASIBLE", 2.0, 5),
        ],
        output_dirs=(str(tmp_path),),
    )
    row = paired.loc["graph_n20"]
    assert row["n_feasible_pairs"] == 1
    assert row["n_paired_optimal_runtime"] == 0


def test_both_optimal_with_different_bandwidths_are_flagged(tmp_path):
    _, paired = make_table_encoding_comparison(
        [
            _encoding_record("order", "OPTIMAL", 1.0, 5),
            _encoding_record("pairwise", "OPTIMAL", 2.0, 6),
        ],
        output_dirs=(str(tmp_path),),
    )
    row = paired.loc["graph_n20"]
    assert row["n_mismatched_optimum_pairs"] == 1
    assert row["n_equal_optimum_pairs"] == 0
    assert row["n_paired_optimal_runtime"] == 0


def test_equal_optimum_with_missing_timing_is_not_given_a_ratio(tmp_path):
    _, paired = make_table_encoding_comparison(
        [
            _encoding_record("order", "OPTIMAL", None, 5),
            _encoding_record("pairwise", "OPTIMAL", 2.0, 5),
        ],
        output_dirs=(str(tmp_path),),
    )
    row = paired.loc["graph_n20"]
    assert row["n_equal_optimum_pairs"] == 1
    assert row["n_optimal_pairs_missing_or_nonpositive_timing"] == 1
    assert row["n_paired_optimal_runtime"] == 0


def test_repeated_runs_with_mixed_statuses_keep_counters_and_pair_by_rep(tmp_path):
    detail, paired = make_table_encoding_comparison(
        [
            _encoding_record("order", "OPTIMAL", 1.0, 5, rep=1),
            _encoding_record("order", "TIMEOUT", None, 5, rep=2),
            _encoding_record("pairwise", "OPTIMAL", 2.0, 5, rep=1),
            _encoding_record("pairwise", "TIMEOUT", None, 5, rep=2),
        ],
        output_dirs=(str(tmp_path),),
    )
    order_row = detail[detail["encoding"] == "order"].iloc[0]
    pairwise_row = detail[detail["encoding"] == "pairwise"].iloc[0]
    assert order_row["status"] == "PARTIAL"
    assert pairwise_row["status"] == "PARTIAL"
    assert order_row["n_total"] == 2
    assert order_row["n_optimal"] == 1
    assert order_row["n_timeout"] == 1
    row = paired.loc["graph_n20"]
    assert row["n_matched_repetition_pairs"] == 2
    assert row["n_paired_optimal_runtime"] == 1
    assert row["n_timeout_pairs"] == 1


def test_exact_solvers_group_also_feeds_encoding_comparison(tmp_path):
    order = _encoding_record("order", "OPTIMAL", 1.0, 5)
    pairwise = _encoding_record("pairwise", "OPTIMAL", 2.0, 5)
    order["groups"] = ["exact_solvers"]
    pairwise["groups"] = ["exact_solvers"]

    detail, paired = make_table_encoding_comparison(
        [order, pairwise], output_dirs=(str(tmp_path),)
    )

    assert set(detail["encoding"]) == {"order", "pairwise"}
    row = paired.loc["graph_n20"]
    assert row["n_both_encodings"] == 1
    assert row["n_paired_optimal_runtime"] == 1
    assert row["median_pairwise_over_order"] == 2.0


def test_encoding_table_skips_without_encoding_records(tmp_path):
    result = make_table_encoding_comparison(
        [{"dataset": "graph_n20", "solver": "cadical195", "status": "OPTIMAL"}],
        output_dirs=(str(tmp_path),),
    )
    assert result is None
    assert list(tmp_path.iterdir()) == []
