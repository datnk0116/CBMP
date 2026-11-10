import os
import sys
import glob
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def aggregate_repetitions(records):
    """
    Group records by unique configuration (dataset, method/solver, config_signature),
    and aggregate across repetitions by computing the median for numerical timing metrics.
    """
    groups = {}
    for r in records:
        ds = r.get("dataset")
        sol = str(r.get("solver", "")).lower()
        cfg = r.get("config", {})
        rot = bool(cfg.get("sb_rotation", False))
        ref = bool(cfg.get("sb_reflection", False))
        heur = bool(cfg.get("heuristic_ub", False))
        num_w = cfg.get("num_workers", 1)
        key = (ds, sol, rot, ref, heur, num_w)
        groups.setdefault(key, []).append(r)

    aggregated = []
    for key, reps in groups.items():
        n_total = len(reps)
        optimal_reps = [r for r in reps if r.get("status") == "OPTIMAL"]
        n_optimal = len(optimal_reps)

        # Compute per-rep PAR2 values before aggregating:
        # Each OPTIMAL rep contributes solve_time; each non-optimal rep contributes 2T; SIZE_LIMIT is excluded.
        rep_par2_vals = []
        for r in reps:
            st = r.get("status")
            if st == "OPTIMAL":
                t = r.get("solve_time")
                if t is not None:
                    rep_par2_vals.append(float(t))
            elif st == "SIZE_LIMIT":
                pass
            else:
                r_timeout = r.get("config", {}).get("time_limit") or r.get("time_limit") or 600.0
                rep_par2_vals.append(2.0 * float(r_timeout))

        if n_total == 1:
            agg = dict(reps[0])
            st = agg.get("status")
            agg["n_optimal"] = 1 if st == "OPTIMAL" else 0
            agg["n_total"] = 1
            if rep_par2_vals:
                agg["par2"] = float(np.mean(rep_par2_vals))
            aggregated.append(agg)
            continue

        if n_optimal == n_total:
            # All reps optimal -> OPTIMAL
            reps_sorted = sorted(
                reps,
                key=lambda x: x.get("solve_time") if x.get("solve_time") is not None else float("inf"),
            )
            median_idx = len(reps_sorted) // 2
            rep_chosen = reps_sorted[median_idx]
            agg = dict(rep_chosen)
            agg["status"] = "OPTIMAL"
            agg["n_optimal"] = n_optimal
            agg["n_total"] = n_total

            solve_times = [r["solve_time"] for r in reps if r.get("solve_time") is not None]
            wall_times = [r["wall_time"] for r in reps if r.get("wall_time") is not None]
            encode_times = [r["encode_time"] for r in reps if r.get("encode_time") is not None]
        elif n_optimal > 0:
            # Mixed: some OPTIMAL, some not -> PARTIAL
            reps_sorted = sorted(
                optimal_reps,
                key=lambda x: x.get("solve_time") if x.get("solve_time") is not None else float("inf"),
            )
            median_idx = len(reps_sorted) // 2
            rep_chosen = reps_sorted[median_idx]
            agg = dict(rep_chosen)
            agg["status"] = "PARTIAL"
            agg["n_optimal"] = n_optimal
            agg["n_total"] = n_total
            agg["mixed"] = True
            agg["is_mixed"] = True

            solve_times = [r["solve_time"] for r in optimal_reps if r.get("solve_time") is not None]
            wall_times = [r["wall_time"] for r in optimal_reps if r.get("wall_time") is not None]
            encode_times = [r["encode_time"] for r in optimal_reps if r.get("encode_time") is not None]
        else:
            # 0 optimal reps: preserve non-optimal status
            reps_sorted = sorted(
                reps,
                key=lambda x: x.get("solve_time") if x.get("solve_time") is not None else float("inf"),
            )
            median_idx = len(reps_sorted) // 2
            rep_chosen = reps_sorted[median_idx]
            agg = dict(rep_chosen)
            agg["n_optimal"] = 0
            agg["n_total"] = n_total

            solve_times = [r["solve_time"] for r in reps if r.get("solve_time") is not None]
            wall_times = [r["wall_time"] for r in reps if r.get("wall_time") is not None]
            encode_times = [r["encode_time"] for r in reps if r.get("encode_time") is not None]

        if solve_times:
            agg["solve_time"] = float(np.median(solve_times))
        if wall_times:
            agg["wall_time"] = float(np.median(wall_times))
        if encode_times:
            agg["encode_time"] = float(np.median(encode_times))
        if rep_par2_vals:
            agg["par2"] = float(np.mean(rep_par2_vals))

        # Union of groups across reps
        all_groups = set()
        for r in reps:
            for g in r.get("groups", []):
                all_groups.add(g)
            if "_group" in r:
                all_groups.add(r["_group"])
        agg["groups"] = sorted(list(all_groups))

        aggregated.append(agg)

    return aggregated

def load_records(jsonl_source=None):
    """
    Load all records from runs.jsonl (or pattern/fallback),
    and aggregate across repetitions taking the median.
    """
    raw_records = []
    if jsonl_source is not None and os.path.isfile(jsonl_source):
        files = [jsonl_source]
    elif jsonl_source is not None and os.path.isdir(jsonl_source):
        preferred = os.path.join(jsonl_source, "runs.jsonl")
        files = [preferred] if os.path.exists(preferred) else sorted(glob.glob(os.path.join(jsonl_source, "*.jsonl")))
    elif os.path.exists("results/runs.jsonl"):
        files = ["results/runs.jsonl"]
    else:
        files = sorted(glob.glob("results/*.jsonl"))

    for filepath in files:
        group_name = Path(filepath).stem
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("_type") == "metadata" or "dataset" not in data:
                        continue
                    if "groups" not in data or not isinstance(data["groups"], list):
                        data["groups"] = [data.get("_group", group_name)]
                    if "rep" not in data:
                        data["rep"] = 1
                    data["_file"] = filepath
                    raw_records.append(data)
                except Exception:
                    pass

    return aggregate_repetitions(raw_records)

def cross_validate_optima(records, output_dirs=("analysis/tables", "results/tables")):
    """
    Cross-validation check:
    For every instance where >= 2 methods/solvers solved to OPTIMAL,
    assert that their best_ub are strictly identical.
    Exports cross_validation.csv to output_dirs.
    Raises ValueError if conflicting optima are detected.
    """
    by_dataset = {}
    for r in records:
        if r.get("status") == "OPTIMAL":
            ds = r.get("dataset")
            ub = r.get("best_ub")
            sol = r.get("solver") or r.get("method") or "unknown"
            grps = ",".join(r.get("groups", [])) if isinstance(r.get("groups"), list) else str(r.get("groups", ""))
            by_dataset.setdefault(ds, []).append({
                "dataset": ds,
                "solver_or_method": sol,
                "groups": grps,
                "best_ub": ub,
            })

    compared_rows = []
    mismatches = []
    for ds, items in by_dataset.items():
        if len(items) >= 2:
            unique_ubs = set(item["best_ub"] for item in items if item["best_ub"] is not None)
            is_agreed = (len(unique_ubs) == 1)
            for item in items:
                row = dict(item)
                row["agreed"] = is_agreed
                compared_rows.append(row)
            if not is_agreed:
                mismatches.append((ds, items))

    df_cv = pd.DataFrame(compared_rows, columns=["dataset", "solver_or_method", "groups", "best_ub", "agreed"])
    if output_dirs:
        for out_dir in output_dirs:
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, "cross_validation.csv")
            df_cv.to_csv(csv_path, index=False)
            print(f"Cross-validation table saved: {csv_path}")

    if mismatches:
        print("\n" + "!" * 80)
        print("[CROSS-VALIDATION FAILED] Conflicting optimal solutions detected:")
        for ds, results in mismatches:
            print(f"  Dataset '{ds}':")
            for item in results:
                print(f"    - {item['solver_or_method']} ({item['groups']}): best_ub = {item['best_ub']}")
        print("!" * 80 + "\n")
        raise ValueError(f"Cross-validation failed: conflicting optima on {len(mismatches)} dataset(s).")
    else:
        num_compared_datasets = sum(1 for ds, items in by_dataset.items() if len(items) >= 2)
        total_checked = len(compared_rows)
        print(f"[CROSS-VALIDATION PASSED] Verified {total_checked} optimal runs across {num_compared_datasets} dataset(s) with >= 2 optimal solutions. All optima match identically.")

def format_cell_value(status, solve_time=0.0, best_ub=None, n_optimal=None, n_total=None):
    """Format single cell according to required status representations."""
    if isinstance(status, dict):
        rec = status
        status = rec.get("status")
        solve_time = rec.get("solve_time", 0.0)
        best_ub = rec.get("best_ub")
        n_optimal = rec.get("n_optimal")
        n_total = rec.get("n_total")

    if solve_time is None:
        solve_time = 0.0

    if status == "OPTIMAL":
        return f"{solve_time:.4f}"
    elif status == "PARTIAL":
        k_n = f"{n_optimal}/{n_total}" if (n_optimal is not None and n_total is not None) else "PARTIAL"
        return f"{k_n} {solve_time:.4f}*"
    elif status == "TIMEOUT":
        return f"TO({best_ub})" if best_ub is not None and best_ub >= 0 else "TO"
    elif status == "SIZE_LIMIT":
        return "SL"
    elif status == "ERROR":
        return "ERR"
    else:
        return str(status)

def compute_summary_stats(df_raw, column_names, baseline_col=None, timeout=600.0):
    """
    Computes summary rows:
    - Solved count
    - SL (Size limit count)
    - PAR2 score (solve_time if optimal, else 2 * timeout; SIZE_LIMIT is excluded)
    - Geometric mean speedup relative to baseline_col
    """
    solved_counts = {}
    sl_counts = {}
    par2_scores = {}
    speedups = {}

    baseline_times = {}
    if baseline_col and baseline_col in df_raw:
        for ds, row in df_raw.iterrows():
            rec = row.get(baseline_col)
            if rec and rec.get("status") == "OPTIMAL":
                baseline_times[ds] = rec.get("solve_time", timeout)

    for col in column_names:
        solved = 0
        sl_count = 0
        par2_vals = []
        ratios = []

        for ds, row in df_raw.iterrows():
            rec = row.get(col)
            if not rec:
                continue
            st = rec.get("status")
            t = rec.get("solve_time", 0.0) or 0.0
            rec_timeout = rec.get("config", {}).get("time_limit", timeout) or timeout

            if st == "OPTIMAL":
                solved += 1
                par2_val = rec.get("par2") if rec.get("par2") is not None else t
                par2_vals.append(par2_val)
                if ds in baseline_times and t > 0:
                    ratios.append(baseline_times[ds] / t)
            elif st == "PARTIAL":
                if rec.get("par2") is not None:
                    par2_vals.append(rec["par2"])
                else:
                    n_opt = rec.get("n_optimal", 0)
                    n_tot = rec.get("n_total", 1)
                    par2_vals.append((n_opt * t + (n_tot - n_opt) * 2.0 * float(rec_timeout)) / n_tot)
            elif st == "SIZE_LIMIT":
                sl_count += 1
                # SIZE_LIMIT is reported separately in SL and not penalized in PAR2 as timeout
            else:
                par2_val = rec.get("par2") if rec.get("par2") is not None else (2.0 * float(rec_timeout))
                par2_vals.append(par2_val)

        solved_counts[col] = solved
        sl_counts[col] = sl_count
        par2_scores[col] = float(np.mean(par2_vals)) if par2_vals else float("nan")

        if col == baseline_col:
            speedups[col] = 1.00
        elif ratios:
            speedups[col] = float(np.exp(np.mean(np.log(ratios))))
        else:
            speedups[col] = float("nan")

    return solved_counts, sl_counts, par2_scores, speedups

def export_table(df_display, summary_rows, title, base_filename, output_dirs=("analysis/tables", "results/tables"), note=None):
    """Saves formatted table as both CSV and LaTeX."""
    full_df = df_display.astype(object).copy()
    for row_name, row_data in summary_rows:
        full_df.loc[row_name] = row_data

    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{base_filename}.csv")
        full_df.to_csv(csv_path, index=True)

        tex_path = os.path.join(out_dir, f"{base_filename}.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(f"% Table: {title}\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{{title}}}\n")
            f.write(f"\\label{{tab:{base_filename}}}\n")
            cols_align = "l" + "r" * (len(full_df.columns))
            f.write(f"\\begin{{tabular}}{{{cols_align}}}\n")
            f.write("\\toprule\n")
            header_escaped = [c.replace("_", "\\_") for c in full_df.columns]
            f.write("Dataset & " + " & ".join(header_escaped) + " \\\\\n")
            f.write("\\midrule\n")

            for idx, row in df_display.iterrows():
                idx_escaped = str(idx).replace("_", "\\_")
                row_vals = [str(v).replace("_", "\\_") for v in row]
                f.write(f"{idx_escaped} & " + " & ".join(row_vals) + " \\\\\n")

            f.write("\\midrule\n")
            for rname, rdata in summary_rows:
                rname_escaped = f"\\textbf{{{rname.replace('_', ' ')}}}"
                vals = [str(rdata[c]).replace("_", "\\_") for c in full_df.columns]
                f.write(f"{rname_escaped} & " + " & ".join(vals) + " \\\\\n")

            f.write("\\bottomrule\n")
            if note:
                total_cols = len(full_df.columns) + 1
                f.write(f"\\multicolumn{{{total_cols}}}{{l}}{{\\footnotesize {note}}} \\\\\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

    print(f"Table saved: {base_filename}.csv and {base_filename}.tex in {output_dirs[0]}/")
    return full_df

def make_table_ablation(records, group_name="ablation_small", base_filename="table_ablation_small", title="Ablation Study: Small Graphs", output_dirs=("analysis/tables", "results/tables")):
    """Generate Ablation study table for a specific ablation group."""
    ablation_recs = [
        r for r in records
        if group_name in r.get("groups", []) or (group_name == "ablation_small" and "ablation" in r.get("groups", []))
    ]
    if not ablation_recs:
        return

    def get_var_name(cfg):
        rot = cfg.get("sb_rotation", False)
        ref = cfg.get("sb_reflection", False)
        heur = cfg.get("heuristic_ub", False)
        if not rot and not ref and not heur:
            return "none"
        elif rot and not ref and not heur:
            return "rotation"
        elif rot and ref and not heur:
            return "rot_ref"
        elif not rot and not ref and heur:
            return "heur_ub"
        elif rot and ref and heur:
            return "all"
        return "other"

    columns = ["none", "rotation", "rot_ref", "heur_ub", "all"]

    data_by_ds = {}
    meta = {}
    for r in ablation_recs:
        ds = r["dataset"]
        meta[ds] = (r.get("N", 0), r.get("num_edges", 0))
        var = get_var_name(r.get("config", {}))
        data_by_ds.setdefault(ds, {})[var] = r

    ds_sorted = sorted(data_by_ds.keys(), key=lambda d: (meta[d][0], d))

    df_raw = pd.DataFrame.from_dict(data_by_ds, orient="index")
    df_display = pd.DataFrame(index=ds_sorted)
    df_display["N"] = [meta[d][0] for d in ds_sorted]
    df_display["|E|"] = [meta[d][1] for d in ds_sorted]

    for col in columns:
        col_vals = []
        for ds in ds_sorted:
            rec = data_by_ds[ds].get(col)
            if rec:
                col_vals.append(format_cell_value(rec))
            else:
                col_vals.append("-")
        df_display[col] = col_vals

    solved_cnt, sl_cnt, par2_val, spdups = compute_summary_stats(df_raw, columns, baseline_col="none")

    sum_solved = {"N": "-", "|E|": "-"}
    sum_sl = {"N": "-", "|E|": "-"}
    sum_par2 = {"N": "-", "|E|": "-"}
    sum_spdup = {"N": "-", "|E|": "-"}
    for col in columns:
        sum_solved[col] = str(solved_cnt[col])
        sum_sl[col] = str(sl_cnt[col])
        sum_par2[col] = f"{par2_val[col]:.3f}" if not math.isnan(par2_val[col]) else "-"
        sum_spdup[col] = f"{spdups[col]:.2f}x" if not math.isnan(spdups[col]) else "-"

    summary = [
        ("Solved", sum_solved),
    ]
    if any(v > 0 for v in sl_cnt.values()):
        summary.append(("SL", sum_sl))
    summary.extend([
        ("PAR2 (s)", sum_par2),
        ("Speedup (geom)", sum_spdup),
    ])

    export_table(df_display, summary, title, base_filename, output_dirs=output_dirs)

def make_table_solvers(records, group_name="solvers_small", base_filename="table_solvers_small", title="SAT Solver Performance Comparison (Small Graphs)", output_dirs=("analysis/tables", "results/tables")):
    """Generate Solvers comparison table for a specific solvers group."""
    solver_recs = [
        r for r in records
        if group_name in r.get("groups", []) or (group_name == "solvers_small" and "solvers" in r.get("groups", []))
    ]
    if not solver_recs:
        return

    solvers = ["cadical195", "glucose4", "maplechrono", "minisat22", "lingeling"]

    data_by_ds = {}
    meta = {}
    for r in solver_recs:
        ds = r["dataset"]
        meta[ds] = (r.get("N", 0), r.get("num_edges", 0))
        sol = r["solver"].lower()
        data_by_ds.setdefault(ds, {})[sol] = r

    ds_sorted = sorted(data_by_ds.keys(), key=lambda d: (meta[d][0], d))

    df_raw = pd.DataFrame.from_dict(data_by_ds, orient="index")
    df_display = pd.DataFrame(index=ds_sorted)
    df_display["N"] = [meta[d][0] for d in ds_sorted]
    df_display["|E|"] = [meta[d][1] for d in ds_sorted]

    for sol in solvers:
        col_vals = []
        for ds in ds_sorted:
            rec = data_by_ds[ds].get(sol)
            if rec:
                col_vals.append(format_cell_value(rec))
            else:
                col_vals.append("-")
        df_display[sol] = col_vals

    solved_cnt, sl_cnt, par2_val, spdups = compute_summary_stats(df_raw, solvers, baseline_col="cadical195")

    sum_solved = {"N": "-", "|E|": "-"}
    sum_sl = {"N": "-", "|E|": "-"}
    sum_par2 = {"N": "-", "|E|": "-"}
    sum_spdup = {"N": "-", "|E|": "-"}
    for sol in solvers:
        sum_solved[sol] = str(solved_cnt[sol])
        sum_sl[sol] = str(sl_cnt[sol])
        sum_par2[sol] = f"{par2_val[sol]:.3f}" if not math.isnan(par2_val[sol]) else "-"
        sum_spdup[sol] = f"{spdups[sol]:.2f}x" if not math.isnan(spdups[sol]) else "-"

    summary = [
        ("Solved", sum_solved),
    ]
    if any(v > 0 for v in sl_cnt.values()):
        summary.append(("SL", sum_sl))
    summary.extend([
        ("PAR2 (s)", sum_par2),
        ("Speedup (geom)", sum_spdup),
    ])

    export_table(df_display, summary, title, base_filename, output_dirs=output_dirs)

def make_table_methods(records, output_dirs=("analysis/tables", "results/tables")):
    """Generate Methods comparison table (SAT vs CP-SAT vs CPLEX)."""
    method_recs = [r for r in records if "methods" in r.get("groups", [])]
    if not method_recs:
        return

    methods = ["sat_best", "cpsat", "cplex"]

    data_by_ds = {}
    meta = {}
    for r in method_recs:
        ds = r["dataset"]
        meta[ds] = (r.get("N", 0), r.get("num_edges", 0))
        sol = r["solver"].lower()
        if sol in ("cp_sat", "cpsat"):
            m = "cpsat"
        elif sol == "cplex":
            m = "cplex"
        else:
            m = "sat_best"
        data_by_ds.setdefault(ds, {})[m] = r

    ds_sorted = sorted(data_by_ds.keys(), key=lambda d: (meta[d][0], d))

    df_raw = pd.DataFrame.from_dict(data_by_ds, orient="index")
    df_display = pd.DataFrame(index=ds_sorted)
    df_display["N"] = [meta[d][0] for d in ds_sorted]
    df_display["|E|"] = [meta[d][1] for d in ds_sorted]

    for m in methods:
        col_vals = []
        for ds in ds_sorted:
            rec = data_by_ds[ds].get(m)
            if rec:
                col_vals.append(format_cell_value(rec))
            else:
                col_vals.append("-")
        df_display[m] = col_vals

    solved_cnt, sl_cnt, par2_val, spdups = compute_summary_stats(df_raw, methods, baseline_col="sat_best")

    sum_solved = {"N": "-", "|E|": "-"}
    sum_sl = {"N": "-", "|E|": "-"}
    sum_par2 = {"N": "-", "|E|": "-"}
    sum_spdup = {"N": "-", "|E|": "-"}
    for m in methods:
        sum_solved[m] = str(solved_cnt[m])
        sum_sl[m] = str(sl_cnt[m])
        sum_par2[m] = f"{par2_val[m]:.3f}" if not math.isnan(par2_val[m]) else "-"
        sum_spdup[m] = f"{spdups[m]:.2f}x" if not math.isnan(spdups[m]) else "-"

    summary = [
        ("Solved", sum_solved),
        ("SL", sum_sl),
        ("PAR2 (s)", sum_par2),
        ("Speedup (geom)", sum_spdup),
    ]

    note = "Note: SL = Size Limit (exceeded CPLEX Community Edition limits, excluded from PAR2)."
    export_table(df_display, summary, "Comparison of Methods: SAT vs CP-SAT vs CPLEX", "table_methods", output_dirs=output_dirs, note=note)

def main():
    print("=" * 70)
    print("ANALYSIS: GENERATING LATEX AND CSV TABLES")
    print("=" * 70)

    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    output_dirs = (output_dir,) if output_dir else ("analysis/tables", "results/tables")
    records = load_records(input_file)
    if not records:
        print("No records found to analyze. Please run experiments first.")
        return

    # Step 1: Cross-validation check
    try:
        cross_validate_optima(records, output_dirs=output_dirs)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Make tables
    make_table_ablation(records, group_name="ablation_small", base_filename="table_ablation_small", title="Ablation Study: Small Graphs", output_dirs=output_dirs)
    make_table_ablation(records, group_name="ablation_medium", base_filename="table_ablation_medium", title="Ablation Study: Medium Graph (662_bus_n60)", output_dirs=output_dirs)
    make_table_solvers(records, group_name="solvers_small", base_filename="table_solvers_small", title="SAT Solver Performance Comparison (Small Graphs)", output_dirs=output_dirs)
    make_table_solvers(records, group_name="solvers_medium", base_filename="table_solvers_medium", title="SAT Solver Performance Comparison (Medium Graphs)", output_dirs=output_dirs)
    make_table_methods(records, output_dirs=output_dirs)

    print("\nAll tables generated successfully!")

if __name__ == "__main__":
    main()
