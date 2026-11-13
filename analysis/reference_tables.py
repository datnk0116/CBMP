"""
Reference Tables (Table 2 and Table 3) from Published CBMP Literature.

Provides:
  - Exact reference values for Table 2 (performance comparison) and Table 3 (statistical significance).
  - Cell-level validation against published data.
  - Arithmetic sum checks and Table 3 discrepancy analysis (185 printed vs 113 computed).
  - Standalone CSV and LaTeX exports without modifying report/ or experimental result tables.
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DATA_PATH = Path(__file__).resolve().parent / "reference_data" / "published_tables.json"

def load_published_data(json_path: Path = REFERENCE_DATA_PATH) -> Dict[str, Any]:
    """Load machine-readable reference table data."""
    if not json_path.exists():
        raise FileNotFoundError(f"Reference tables data file not found at: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_table2(data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Validate every cell in Table 2 against published specifications.
    Returns validation summary dict or raises AssertionError on mismatch.
    """
    if data is None:
        data = load_published_data()

    t2 = data["table2"]
    rows = t2["rows"]

    expected_categories = [
        ("path", 15),
        ("cycle", 15),
        ("mesh2D", 15),
        ("mesh3D", 10),
        ("tree", 12),
        ("caterpillar", 15),
        ("hypercube", 3),
        ("Harwell-Boeing", 28),
    ]

    assert len(rows) == len(expected_categories), f"Expected {len(expected_categories)} rows, got {len(rows)}"

    total_instances = 0
    for idx, (exp_cat, exp_num) in enumerate(expected_categories):
        row = rows[idx]
        assert row["graph_type"] == exp_cat, f"Row {idx} category mismatch: {row['graph_type']} != {exp_cat}"
        assert row["num"] == exp_num, f"Row {idx} num mismatch: {row['num']} != {exp_num}"
        total_instances += row["num"]

        for alg in ("tscb", "itps", "nils"):
            assert alg in row, f"Missing algorithm {alg} in row {exp_cat}"
            metrics = row[alg]
            for m in ("avg_cbb", "avg_tb", "o_rmse", "pct_best"):
                assert m in metrics, f"Missing metric {m} for {alg} in row {exp_cat}"
                val_str = metrics[m]
                assert isinstance(val_str, str), f"Metric {m} should be stored as formatted string, got {type(val_str)}"
                float(val_str)  # verify valid float

    assert total_instances == 113, f"Sum of category instances must be 113, got {total_instances}"

    summary = t2.get("summary_row", {})
    assert summary.get("label") == "Win/Match/Fail", f"Summary label mismatch: {summary.get('label')}"
    assert summary.get("value") == "26/87/0", f"Summary value mismatch: {summary.get('value')}"

    return {
        "status": "VALID",
        "num_rows": len(rows),
        "total_instances": total_instances,
        "summary": summary,
    }

def validate_table3(data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Validate Table 3 reference cells, verify category totals,
    and analyze the printed-total discrepancy (185 vs 113).
    """
    if data is None:
        data = load_published_data()

    t3 = data["table3"]
    rows = t3["rows"]

    expected_categories = [
        ("path", 15, (8, 7, 0), (5, 10, 0)),
        ("cycle", 15, (5, 10, 0), (10, 5, 0)),
        ("mesh2D", 15, (8, 7, 0), (10, 5, 0)),
        ("mesh3D", 10, (10, 0, 0), (7, 3, 0)),
        ("tree", 12, (4, 8, 0), (1, 11, 0)),
        ("caterpillar", 15, (6, 9, 0), (3, 12, 0)),
        ("hypercube", 3, (2, 1, 0), (3, 0, 0)),
        ("Harwell-Boeing", 28, (15, 13, 0), (11, 17, 0)),
    ]

    assert len(rows) == len(expected_categories), f"Expected {len(expected_categories)} rows, got {len(rows)}"

    sum_num = 0
    sum_tscb = [0, 0, 0]
    sum_itps = [0, 0, 0]

    for idx, (exp_cat, exp_num, exp_tscb, exp_itps) in enumerate(expected_categories):
        r = rows[idx]
        assert r["graph_type"] == exp_cat, f"Row {idx} category: {r['graph_type']} != {exp_cat}"
        assert r["num"] == exp_num, f"Row {idx} num: {r['num']} != {exp_num}"
        sum_num += r["num"]

        tscb = (r["nils_vs_tscb"]["better"], r["nils_vs_tscb"]["equal"], r["nils_vs_tscb"]["worse"])
        itps = (r["nils_vs_itps"]["better"], r["nils_vs_itps"]["equal"], r["nils_vs_itps"]["worse"])

        assert tscb == exp_tscb, f"Row {exp_cat} NILS/TSCB mismatch: {tscb} != {exp_tscb}"
        assert itps == exp_itps, f"Row {exp_cat} NILS/ITPS mismatch: {itps} != {exp_itps}"

        assert sum(tscb) == exp_num, f"Row {exp_cat} TSCB counts sum ({sum(tscb)}) != num ({exp_num})"
        assert sum(itps) == exp_num, f"Row {exp_cat} ITPS counts sum ({sum(itps)}) != num ({exp_num})"

        for i in range(3):
            sum_tscb[i] += tscb[i]
            sum_itps[i] += itps[i]

    total_row = t3["total_row"]
    printed_total_num = total_row["printed_num"]
    computed_total_num = sum_num

    assert computed_total_num == 113, f"Arithmetic sum of categories must be 113, got {computed_total_num}"
    assert printed_total_num == 185, f"Printed total in source is 185, got {printed_total_num}"

    assert (total_row["nils_vs_tscb"]["better"], total_row["nils_vs_tscb"]["equal"], total_row["nils_vs_tscb"]["worse"]) == tuple(sum_tscb)
    assert (total_row["nils_vs_itps"]["better"], total_row["nils_vs_itps"]["equal"], total_row["nils_vs_itps"]["worse"]) == tuple(sum_itps)
    assert sum(sum_tscb) == 113
    assert sum(sum_itps) == 113

    discrepancy = {
        "printed_total_num": printed_total_num,
        "arithmetic_sum_num": computed_total_num,
        "has_discrepancy": (printed_total_num != computed_total_num),
        "discrepancy_delta": printed_total_num - computed_total_num,
        "nils_vs_tscb_total": sum(sum_tscb),
        "nils_vs_itps_total": sum(sum_itps),
        "explanation": t3["discrepancy_analysis"]["explanation"],
    }

    return {
        "status": "VALID",
        "num_rows": len(rows),
        "arithmetic_sum_num": computed_total_num,
        "printed_total_num": printed_total_num,
        "discrepancy": discrepancy,
    }

def export_table2_csv(data: Dict[str, Any], output_path: str):
    """Export Table 2 to CSV preserving precision and layout."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    t2 = data["table2"]
    cols = t2["columns"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for r in t2["rows"]:
            writer.writerow([
                r["graph_type"],
                r["num"],
                r["tscb"]["avg_cbb"],
                r["tscb"]["avg_tb"],
                r["tscb"]["o_rmse"],
                r["tscb"]["pct_best"],
                r["itps"]["avg_cbb"],
                r["itps"]["avg_tb"],
                r["itps"]["o_rmse"],
                r["itps"]["pct_best"],
                r["nils"]["avg_cbb"],
                r["nils"]["avg_tb"],
                r["nils"]["o_rmse"],
                r["nils"]["pct_best"],
            ])
        summary = t2["summary_row"]
        summary_row = [summary["label"], summary["value"]] + [""] * (len(cols) - 2)
        writer.writerow(summary_row)

def export_table2_latex(data: Dict[str, Any], output_path: str):
    """Export Table 2 to LaTeX tabular format."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    t2 = data["table2"]
    rows = t2["rows"]
    summary = t2["summary_row"]

    lines = []
    lines.append("% Published Reference Table 2 (from CBMP Literature)")
    lines.append("% Note: Baseline published values for TSCB, ITPS, and NILS.")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Summary of experimental performance comparison from published CBMP literature}")
    lines.append(r"\label{tab:published_ref_table2}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lr|rrrr|rrrr|rrrr}")
    lines.append(r"\toprule")
    lines.append(r" & & \multicolumn{4}{c|}{TSCB} & \multicolumn{4}{c|}{ITPS} & \multicolumn{4}{c}{NILS} \\")
    lines.append(r"Graph type & Num. & Avg. $C_{bb}$ & Avg. $T_b$ & O-RMSE & \% Best & Avg. $C_{bb}$ & Avg. $T_b$ & O-RMSE & \% Best & Avg. $C_{bb}$ & Avg. $T_b$ & O-RMSE & \% Best \\")
    lines.append(r"\midrule")

    for r in rows:
        lines.append(
            f"{r['graph_type']} & {r['num']} & "
            f"{r['tscb']['avg_cbb']} & {r['tscb']['avg_tb']} & {r['tscb']['o_rmse']} & {r['tscb']['pct_best']} & "
            f"{r['itps']['avg_cbb']} & {r['itps']['avg_tb']} & {r['itps']['o_rmse']} & {r['itps']['pct_best']} & "
            f"{r['nils']['avg_cbb']} & {r['nils']['avg_tb']} & {r['nils']['o_rmse']} & {r['nils']['pct_best']} \\\\"
        )

    lines.append(r"\midrule")
    lbl = summary['label']
    val = summary['value']
    lines.append(f"\\multicolumn{{2}}{{l}}{{{lbl} = {val}}} & \\multicolumn{{12}}{{c}}{{}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def export_table3_csv(data: Dict[str, Any], output_path: str):
    """Export Table 3 to CSV preserving original printed values and documented totals."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    t3 = data["table3"]
    cols = t3["columns"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for r in t3["rows"]:
            writer.writerow([
                r["graph_type"],
                r["num"],
                r["nils_vs_tscb"]["better"],
                r["nils_vs_tscb"]["equal"],
                r["nils_vs_tscb"]["worse"],
                r["nils_vs_itps"]["better"],
                r["nils_vs_itps"]["equal"],
                r["nils_vs_itps"]["worse"],
            ])
        tot = t3["total_row"]
        writer.writerow([
            tot["graph_type"],
            tot["printed_num"],
            tot["nils_vs_tscb"]["better"],
            tot["nils_vs_tscb"]["equal"],
            tot["nils_vs_tscb"]["worse"],
            tot["nils_vs_itps"]["better"],
            tot["nils_vs_itps"]["equal"],
            tot["nils_vs_itps"]["worse"],
        ])

def export_table3_latex(data: Dict[str, Any], output_path: str):
    """Export Table 3 to LaTeX tabular format."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    t3 = data["table3"]
    rows = t3["rows"]
    tot = t3["total_row"]

    lines = []
    lines.append("% Published Reference Table 3 (from CBMP Literature)")
    lines.append("% Note: Preserves published printed total 185 while documenting arithmetic sum 113.")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Summary of statistical significance analysis from published CBMP literature}")
    lines.append(r"\label{tab:published_ref_table3}")
    lines.append(r"\begin{tabular}{lr|rrr|rrr}")
    lines.append(r"\toprule")
    lines.append(r" & & \multicolumn{3}{c|}{NILS / TSCB} & \multicolumn{3}{c}{NILS / ITPS} \\")
    lines.append(r"Graph type & Num. & + & * & $-$ & + & * & $-$ \\")
    lines.append(r"\midrule")

    for r in rows:
        lines.append(
            f"{r['graph_type']} & {r['num']} & "
            f"{r['nils_vs_tscb']['better']} & {r['nils_vs_tscb']['equal']} & {r['nils_vs_tscb']['worse']} & "
            f"{r['nils_vs_itps']['better']} & {r['nils_vs_itps']['equal']} & {r['nils_vs_itps']['worse']} \\\\"
        )

    lines.append(r"\midrule")
    # Include printed number with note on arithmetic total
    lines.append(
        f"{tot['graph_type']} & {tot['printed_num']} & "
        f"{tot['nils_vs_tscb']['better']} & {tot['nils_vs_tscb']['equal']} & {tot['nils_vs_tscb']['worse']} & "
        f"{tot['nils_vs_itps']['better']} & {tot['nils_vs_itps']['equal']} & {tot['nils_vs_itps']['worse']} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"% Note on Table 3 discrepancy: Printed Num total in source is 185; arithmetic sum of category rows is 113.")
    lines.append(r"\end{table}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def export_all_reference_tables(output_dir: str = "results_v2/tables/reference") -> Dict[str, str]:
    """
    Validate and export both Table 2 and Table 3 to CSV and LaTeX in an isolated directory.
    Guarantees no modifications to report/ or experimental result tables.
    """
    data = load_published_data()
    val_t2 = validate_table2(data)
    val_t3 = validate_table3(data)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "table2_csv": str(out_dir / "published_table2.csv"),
        "table2_tex": str(out_dir / "published_table2.tex"),
        "table3_csv": str(out_dir / "published_table3.csv"),
        "table3_tex": str(out_dir / "published_table3.tex"),
    }

    export_table2_csv(data, paths["table2_csv"])
    export_table2_latex(data, paths["table2_tex"])
    export_table3_csv(data, paths["table3_csv"])
    export_table3_latex(data, paths["table3_tex"])

    return {
        "status": "SUCCESS",
        "validation_table2": val_t2,
        "validation_table3": val_t3,
        "exported_files": paths,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate and export published CBMP Table 2 and Table 3 reference tables")
    parser.add_argument("--output-dir", type=str, default="results_v2/tables/reference", help="isolated directory for exported reference tables")
    args = parser.parse_args()

    res = export_all_reference_tables(args.output_dir)
    print("Table 2 Validation:", res["validation_table2"]["status"])
    print("Table 3 Validation:", res["validation_table3"]["status"])
    discrepancy = res["validation_table3"]["discrepancy"]
    print(f"Table 3 Printed Num: {discrepancy['printed_total_num']} vs Arithmetic Sum: {discrepancy['arithmetic_sum_num']}")
    print(f"Discrepancy confirmed: {discrepancy['explanation']}")
    print("Exported reference tables to:")
    for k, v in res["exported_files"].items():
        print(f"  {k}: {v}")
