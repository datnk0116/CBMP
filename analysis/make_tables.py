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
        if len(reps) == 1:
            aggregated.append(reps[0])
            continue

        optimal_reps = [r for r in reps if r.get("status") == "OPTIMAL"]
        statuses = set(r.get("status") for r in reps)
        is_mixed = len(statuses) > 1

        if optimal_reps:
            # If >= 1 rep is OPTIMAL, status is OPTIMAL; median is computed ONLY over OPTIMAL reps
            reps_sorted = sorted(
                optimal_reps,
                key=lambda x: x.get("solve_time") if x.get("solve_time") is not None else float("inf"),
            )
            median_idx = len(reps_sorted) // 2
            rep_chosen = reps_sorted[median_idx]
            agg = dict(rep_chosen)
            agg["status"] = "OPTIMAL"
            if is_mixed:
                agg["mixed"] = True
                agg["is_mixed"] = True

            solve_times = [r["solve_time"] for r in optimal_reps if r.get("solve_time") is not None]
            wall_times = [r["wall_time"] for r in optimal_reps if r.get("wall_time") is not None]
            encode_times = [r["encode_time"] for r in optimal_reps if r.get("encode_time") is not None]
        else:
            # No optimal rep: preserve non-optimal status
            reps_sorted = sorted(
                reps,
                key=lambda x: x.get("solve_time") if x.get("solve_time") is not None else float("inf"),
            )
            median_idx = len(reps_sorted) // 2
            rep_chosen = reps_sorted[median_idx]
            agg = dict(rep_chosen)
            if is_mixed:
                agg["mixed"] = True
                agg["is_mixed"] = True

            solve_times = [r["solve_time"] for r in reps if r.get("solve_time") is not None]
            wall_times = [r["wall_time"] for r in reps if r.get("wall_time") is not None]
            encode_times = [r["encode_time"] for r in reps if r.get("encode_time") is not None]

        if solve_times:
            agg["solve_time"] = float(np.median(solve_times))
        if wall_times:
            agg["wall_time"] = float(np.median(wall_times))
        if encode_times:
            agg["encode_time"] = float(np.median(encode_times))

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

def cross_validate_optima(records):
    """
    Cross-validation check:
    For every instance where >= 2 methods/solvers solved to OPTIMAL,
    assert that their best_ub are strictly identical.
    """
    by_dataset = {}
    for r in records:
        if r.get("status") == "OPTIMAL":
            ds = r.get("dataset")
            ub = r.get("best_ub")
            solver_or_method = r.get("solver")
            by_dataset.setdefault(ds, []).append((solver_or_method, ub, r.get("groups", [])))

    mismatches = []
    for ds, results in by_dataset.items():
        unique_ubs = set(ub for _, ub, _ in results)
        if len(unique_ubs) > 1:
            mismatches.append((ds, results))

    if mismatches:
        print("\n" + "!" * 80)
        print("[CROSS-VALIDATION FAILED] Conflicting optimal solutions detected:")
        for ds, results in mismatches:
            print(f"  Dataset '{ds}':")
            for sol, ub, grps in results:
                print(f"    - {sol} ({grps}): best_ub = {ub}")
        print("!" * 80 + "\n")
        raise ValueError(f"Cross-validation failed: conflicting optima on {len(mismatches)} dataset(s).")
    else:
        total_checked = sum(len(v) for v in by_dataset.values())
        print(f"[CROSS-VALIDATION PASSED] Verified {total_checked} optimal runs across {len(by_dataset)} dataset(s). All optima match identically.")

def format_cell_value(status, solve_time, best_ub):
    """Format single cell according to required status representations."""
    if status == "OPTIMAL":
        return f"{solve_time:.4f}"
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
            t = rec.get("solve_time", 0.0)
            rec_timeout = rec.get("config", {}).get("time_limit", timeout) or timeout

            if st == "OPTIMAL":
                solved += 1
                par2_vals.append(t)
                if ds in baseline_times and t > 0:
                    ratios.append(baseline_times[ds] / t)
            elif st == "SIZE_LIMIT":
                sl_count += 1
                # SIZE_LIMIT is reported separately in SL and not penalized in PAR2 as timeout
            else:
                par2_vals.append(2.0 * float(rec_timeout))

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

    print(f"Table saved: {base_filename}.csv and {base_filename}.tex in {output_dirs[0]}/")
    return full_df

def make_table_ablation(records):
    """Generate Ablation study table."""
    ablation_recs = [r for r in records if "ablation" in r.get("groups", [])]
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
        meta[ds] = (r["N"], r["num_edges"])
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
                col_vals.append(format_cell_value(rec["status"], rec.get("solve_time", 0.0), rec.get("best_ub")))
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

    export_table(df_display, summary, "Ablation Study: Symmetry Breaking and Initial Upper Bound", "table_ablation")

def make_table_solvers(records):
    """Generate Solvers comparison table."""
    solver_recs = [r for r in records if "solvers" in r.get("groups", [])]
    if not solver_recs:
        return

    solvers = ["cadical195", "glucose4", "maplechrono", "minisat22", "lingeling"]

    data_by_ds = {}
    meta = {}
    for r in solver_recs:
        ds = r["dataset"]
        meta[ds] = (r["N"], r["num_edges"])
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
                col_vals.append(format_cell_value(rec["status"], rec.get("solve_time", 0.0), rec.get("best_ub")))
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

    export_table(df_display, summary, "SAT Solver Performance Comparison", "table_solvers")

def make_table_methods(records):
    """Generate Methods comparison table (SAT vs CP-SAT vs CPLEX)."""
    method_recs = [r for r in records if "methods" in r.get("groups", [])]
    if not method_recs:
        return

    methods = ["sat_best", "cpsat", "cplex"]

    data_by_ds = {}
    meta = {}
    for r in method_recs:
        ds = r["dataset"]
        meta[ds] = (r["N"], r["num_edges"])
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
                col_vals.append(format_cell_value(rec["status"], rec.get("solve_time", 0.0), rec.get("best_ub")))
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
    export_table(df_display, summary, "Comparison of Methods: SAT vs CP-SAT vs CPLEX", "table_methods", note=note)

def main():
    print("=" * 70)
    print("ANALYSIS: GENERATING LATEX AND CSV TABLES")
    print("=" * 70)

    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    records = load_records(input_file)
    if not records:
        print("No records found to analyze. Please run experiments first.")
        return

    # Step 1: Cross-validation check
    cross_validate_optima(records)

    # Step 2: Make tables
    make_table_ablation(records)
    make_table_solvers(records)
    make_table_methods(records)

    print("\nAll tables generated successfully!")

if __name__ == "__main__":
    main()
