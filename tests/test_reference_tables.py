"""
Regression tests for published literature reference tables (Table 2 & Table 3)
and aggregation integrity across encodings and status types.
"""

import os
import json
import pytest
from pathlib import Path

from analysis.reference_tables import (
    load_published_data,
    validate_table2,
    validate_table3,
    export_all_reference_tables,
)
from analysis.make_tables import aggregate_repetitions, get_source_key


# ============================================================================
# 1. TABLE 2 VALIDATION TESTS
# ============================================================================

def test_table2_structure_and_categories():
    """Verify Table 2 category names, order, instance counts, and Win/Match/Fail."""
    data = load_published_data()
    res = validate_table2(data)
    assert res["status"] == "VALID"
    assert res["num_rows"] == 8
    assert res["total_instances"] == 113
    assert res["summary"]["label"] == "Win/Match/Fail"
    assert res["summary"]["value"] == "26/87/0"


def test_table2_cell_values_exact():
    """Verify every cell in Table 2 matches the published reference data exactly."""
    data = load_published_data()
    rows = {r["graph_type"]: r for r in data["table2"]["rows"]}

    # Spot check each category
    assert rows["path"]["tscb"] == {"avg_cbb": "2.53", "avg_tb": "131.85", "o_rmse": "2.01", "pct_best": "66.67"}
    assert rows["path"]["itps"] == {"avg_cbb": "1.87", "avg_tb": "158.22", "o_rmse": "3.01", "pct_best": "80.00"}
    assert rows["path"]["nils"] == {"avg_cbb": "1.00", "avg_tb": "6.24", "o_rmse": "0.00", "pct_best": "100.00"}

    assert rows["cycle"]["tscb"] == {"avg_cbb": "2.40", "avg_tb": "40.71", "o_rmse": "1.82", "pct_best": "73.33"}
    assert rows["cycle"]["itps"] == {"avg_cbb": "2.60", "avg_tb": "162.75", "o_rmse": "4.22", "pct_best": "73.33"}
    assert rows["cycle"]["nils"] == {"avg_cbb": "1.00", "avg_tb": "9.38", "o_rmse": "0.00", "pct_best": "100.00"}

    assert rows["mesh2D"]["tscb"] == {"avg_cbb": "27.67", "avg_tb": "144.52", "o_rmse": "1.88", "pct_best": "66.67"}
    assert rows["mesh2D"]["itps"] == {"avg_cbb": "12.07", "avg_tb": "112.86", "o_rmse": "0.44", "pct_best": "40.00"}
    assert rows["mesh2D"]["nils"] == {"avg_cbb": "11.40", "avg_tb": "10.45", "o_rmse": "0.00", "pct_best": "100.00"}

    assert rows["mesh3D"]["tscb"] == {"avg_cbb": "180.30", "avg_tb": "328.32", "o_rmse": "1.47", "pct_best": "30.00"}
    assert rows["mesh3D"]["itps"] == {"avg_cbb": "140.50", "avg_tb": "266.65", "o_rmse": "1.39", "pct_best": "70.00"}
    assert rows["mesh3D"]["nils"] == {"avg_cbb": "64.50", "avg_tb": "132.87", "o_rmse": "0.36", "pct_best": "100.00"}

    assert rows["tree"]["tscb"] == {"avg_cbb": "55.08", "avg_tb": "75.90", "o_rmse": "0.02", "pct_best": "91.67"}
    assert rows["tree"]["itps"] == {"avg_cbb": "54.67", "avg_tb": "23.36", "o_rmse": "0.00", "pct_best": "100.00"}
    assert rows["tree"]["nils"] == {"avg_cbb": "54.67", "avg_tb": "1.52", "o_rmse": "0.00", "pct_best": "100.00"}

    assert rows["caterpillar"]["tscb"] == {"avg_cbb": "15.13", "avg_tb": "75.31", "o_rmse": "0.07", "pct_best": "93.33"}
    assert rows["caterpillar"]["itps"] == {"avg_cbb": "15.07", "avg_tb": "60.54", "o_rmse": "0.07", "pct_best": "100.00"}
    assert rows["caterpillar"]["nils"] == {"avg_cbb": "15.07", "avg_tb": "18.07", "o_rmse": "0.00", "pct_best": "100.00"}

    assert rows["hypercube"]["tscb"] == {"avg_cbb": "1551.67", "avg_tb": "546.23", "o_rmse": "0.34", "pct_best": "0.00"}
    assert rows["hypercube"]["itps"] == {"avg_cbb": "2017.33", "avg_tb": "591.41", "o_rmse": "0.59", "pct_best": "0.00"}
    assert rows["hypercube"]["nils"] == {"avg_cbb": "1492.00", "avg_tb": "584.21", "o_rmse": "0.26", "pct_best": "0.00"}

    assert rows["Harwell-Boeing"]["tscb"] == {"avg_cbb": "22.21", "avg_tb": "112.70", "o_rmse": "2.65", "pct_best": "28.57"}
    assert rows["Harwell-Boeing"]["itps"] == {"avg_cbb": "23.50", "avg_tb": "141.24", "o_rmse": "3.90", "pct_best": "28.57"}
    assert rows["Harwell-Boeing"]["nils"] == {"avg_cbb": "20.39", "avg_tb": "40.69", "o_rmse": "2.15", "pct_best": "28.57"}


# ============================================================================
# 2. TABLE 3 VALIDATION & DISCREPANCY TESTS
# ============================================================================

def test_table3_structure_and_discrepancy():
    """Verify Table 3 categories, statistical sums, and documented discrepancy."""
    data = load_published_data()
    res = validate_table3(data)
    assert res["status"] == "VALID"
    assert res["num_rows"] == 8
    assert res["arithmetic_sum_num"] == 113
    assert res["printed_total_num"] == 185

    discrepancy = res["discrepancy"]
    assert discrepancy["has_discrepancy"] is True
    assert discrepancy["discrepancy_delta"] == 72
    assert discrepancy["nils_vs_tscb_total"] == 113
    assert discrepancy["nils_vs_itps_total"] == 113
    assert "typesetting" in discrepancy["explanation"].lower() or "source" in discrepancy["explanation"].lower()


def test_table3_category_cells_exact():
    """Verify all statistical counts in Table 3 match published data."""
    data = load_published_data()
    rows = {r["graph_type"]: r for r in data["table3"]["rows"]}

    assert rows["path"]["nils_vs_tscb"] == {"better": 8, "equal": 7, "worse": 0}
    assert rows["path"]["nils_vs_itps"] == {"better": 5, "equal": 10, "worse": 0}

    assert rows["cycle"]["nils_vs_tscb"] == {"better": 5, "equal": 10, "worse": 0}
    assert rows["cycle"]["nils_vs_itps"] == {"better": 10, "equal": 5, "worse": 0}

    assert rows["mesh2D"]["nils_vs_tscb"] == {"better": 8, "equal": 7, "worse": 0}
    assert rows["mesh2D"]["nils_vs_itps"] == {"better": 10, "equal": 5, "worse": 0}

    assert rows["mesh3D"]["nils_vs_tscb"] == {"better": 10, "equal": 0, "worse": 0}
    assert rows["mesh3D"]["nils_vs_itps"] == {"better": 7, "equal": 3, "worse": 0}

    assert rows["tree"]["nils_vs_tscb"] == {"better": 4, "equal": 8, "worse": 0}
    assert rows["tree"]["nils_vs_itps"] == {"better": 1, "equal": 11, "worse": 0}

    assert rows["caterpillar"]["nils_vs_tscb"] == {"better": 6, "equal": 9, "worse": 0}
    assert rows["caterpillar"]["nils_vs_itps"] == {"better": 3, "equal": 12, "worse": 0}

    assert rows["hypercube"]["nils_vs_tscb"] == {"better": 2, "equal": 1, "worse": 0}
    assert rows["hypercube"]["nils_vs_itps"] == {"better": 3, "equal": 0, "worse": 0}

    assert rows["Harwell-Boeing"]["nils_vs_tscb"] == {"better": 15, "equal": 13, "worse": 0}
    assert rows["Harwell-Boeing"]["nils_vs_itps"] == {"better": 11, "equal": 17, "worse": 0}


# ============================================================================
# 3. EXPORT INTEGRITY AND ISOLATION
# ============================================================================

def test_reference_tables_export_isolation(tmp_path):
    """Ensure reference tables can be exported to CSV/LaTeX without modifying report/."""
    export_dir = tmp_path / "ref_export"
    res = export_all_reference_tables(str(export_dir))

    assert res["status"] == "SUCCESS"
    for path in res["exported_files"].values():
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


# ============================================================================
# 4. REPETITION AGGREGATION & ENCODING SEPARATION
# ============================================================================

def test_aggregate_repetitions_separates_encodings():
    """Ensure runs with different encodings or modes are never grouped together."""
    r_order = {
        "dataset": "g1",
        "solver": "cadical195",
        "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True, "encoding": "order"},
        "status": "OPTIMAL",
        "best_ub": 2,
        "solve_time": 1.0,
        "wall_time": 1.2,
        "encode_time": 0.1,
    }
    r_pairwise = {
        "dataset": "g1",
        "solver": "cadical195",
        "config": {"sb_rotation": True, "sb_reflection": True, "heuristic_ub": True, "encoding": "pairwise"},
        "status": "OPTIMAL",
        "best_ub": 2,
        "solve_time": 2.5,
        "wall_time": 2.8,
        "encode_time": 0.3,
    }

    agg = aggregate_repetitions([r_order, r_pairwise])
    assert len(agg) == 2, "Order and Pairwise records must not be merged into one group"


def test_aggregate_repetitions_mixed_statuses():
    """Ensure aggregation handles SIZE_LIMIT, TIMEOUT, and ERROR properly."""
    records = [
        {
            "dataset": "g_large",
            "solver": "cplex",
            "config": {"sb_rotation": True},
            "status": "SIZE_LIMIT",
            "best_ub": -1,
            "solve_time": 0.05,
        },
        {
            "dataset": "g_timeout",
            "solver": "cpsat",
            "config": {"time_limit": 10.0},
            "status": "TIMEOUT",
            "best_ub": -1,
            "solve_time": 10.0,
        },
        {
            "dataset": "g_error",
            "solver": "cadical195",
            "config": {},
            "status": "ERROR",
            "best_ub": -1,
            "solve_time": 0.01,
        },
    ]
    agg = aggregate_repetitions(records)
    assert len(agg) == 3
    statuses = {r["status"] for r in agg}
    assert statuses == {"SIZE_LIMIT", "TIMEOUT", "ERROR"}

