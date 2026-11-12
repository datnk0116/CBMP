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

def extract_time_limit(record, default_timeout=600.0):
    """Safely extracts a validated positive numeric timeout from a record."""
    val = record.get("time_limit")
    if val is None:
        val = record.get("timeout")
    if val is None and isinstance(record.get("config"), dict):
        val = record.get("config", {}).get("time_limit")
    if val is not None:
        try:
            f_val = float(val)
            if f_val > 0 and not math.isnan(f_val) and not math.isinf(f_val):
                return f_val
        except (ValueError, TypeError):
            pass
    return float(default_timeout)

def categorize_status(raw_status):
    """Maps any status string to an explicit canonical status category."""
    st = str(raw_status).strip().upper() if raw_status is not None else ""
    if st == "OPTIMAL":
        return "OPTIMAL"
    elif st == "TIMEOUT":
        return "TIMEOUT"
    elif st == "INTERRUPTED":
        return "INTERRUPTED"
    elif st == "FEASIBLE":
        return "FEASIBLE"
    elif st in ("INFEASIBLE", "UNSAT"):
        return "INFEASIBLE"
    elif st in ("ERROR", "NUMERIC_ERROR"):
        return "ERROR"
    elif st == "SIZE_LIMIT":
        return "SIZE_LIMIT"
    elif st == "UNAVAILABLE":
        return "UNAVAILABLE"
    elif st == "LICENSE_ERROR":
        return "LICENSE_ERROR"
    else:
        return "UNKNOWN"

def aggregate_repetitions(records):
    """
    Group records by unique configuration (dataset, method/solver, config_signature),
    and aggregate across repetitions by computing the median for numerical timing metrics.

    Unified Population & PAR2 Semantics (applies identically for n_total == 1 and n_total > 1):
      - n_total: Total recorded repetitions for the configuration.
      - Explicit status counters (every repetition belongs to exactly one category):
          n_optimal: Proven optimal runs.
          n_timeout: Timed-out runs.
          n_interrupted: Runs interrupted by external signal / abort.
          n_feasible: Runs terminated with valid incumbent but without optimality proof.
          n_infeasible: Runs proved infeasible / UNSAT.
          n_error: Runs with unhandled exceptions, validation errors, or numerical issues.
          n_size_limit: Runs hitting engine capacity size limits (e.g. CPLEX Community Edition).
          n_unavailable: Solver binary/engine not installed in the environment.
          n_license_error: Solver license missing or expired.
          n_unknown: Repetitions with missing or unrecognized status strings.
      - n_algorithmic = n_optimal + n_timeout + n_interrupted + n_feasible + n_infeasible + n_error:
          The eligible population where the solver algorithm was actually executed.
          Invariant: n_algorithmic + n_size_limit + n_unavailable + n_license_error + n_unknown == n_total.
      - Algorithmic PAR2 is computed strictly over eligible algorithmic runs:
          Optimal runs contribute solve_time; non-optimal algorithmic runs contribute 2 * time_limit_r.
          SIZE_LIMIT, UNAVAILABLE, LICENSE_ERROR, and UNKNOWN are excluded from algorithmic PAR2.
      - Aggregated status:
          OPTIMAL if and only if n_optimal == n_total (100% of recorded repetitions solved to optimality).
          PARTIAL if n_optimal > 0 and n_optimal < n_total.
          Otherwise non-optimal algorithmic status if n_algorithmic > 0 (FEASIBLE, TIMEOUT, INTERRUPTED, INFEASIBLE, ERROR).
          Otherwise non-algorithmic barrier status (SIZE_LIMIT, LICENSE_ERROR, UNAVAILABLE, UNKNOWN).
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
        seed = r.get("relabel_seed")
        enc = str(r.get("encoding") or cfg.get("encoding", "order")).lower()
        mode = str(r.get("solving_mode") or cfg.get("solving_mode", "incremental")).lower()
        eo = str(r.get("eo_strategy") or cfg.get("eo_strategy", "commander")).lower()
        key = (ds, sol, rot, ref, heur, num_w, seed, enc, mode, eo)
        groups.setdefault(key, []).append(r)

    aggregated = []
    for key, reps in groups.items():
        # Clean and categorize each repetition
        cleaned_reps = []
        for r in reps:
            rc = dict(r)
            cat = categorize_status(rc.get("status"))
            if rc.get("validation_error"):
                cat = "ERROR"
                rc["status"] = "ERROR"
                rc["best_ub"] = -1
                rc["assignment"] = None
            elif cat in ("ERROR", "LICENSE_ERROR", "UNAVAILABLE", "SIZE_LIMIT", "INFEASIBLE", "UNKNOWN"):
                rc["best_ub"] = -1
                rc["assignment"] = None
            rc["_cat"] = cat
            cleaned_reps.append(rc)
        reps = cleaned_reps

        n_total = len(reps)
        if n_total == 0:
            continue

        n_optimal = sum(1 for r in reps if r["_cat"] == "OPTIMAL")
        n_timeout = sum(1 for r in reps if r["_cat"] == "TIMEOUT")
        n_interrupted = sum(1 for r in reps if r["_cat"] == "INTERRUPTED")
        n_feasible = sum(1 for r in reps if r["_cat"] == "FEASIBLE")
        n_infeasible = sum(1 for r in reps if r["_cat"] == "INFEASIBLE")
        n_error = sum(1 for r in reps if r["_cat"] == "ERROR")
        n_size_limit = sum(1 for r in reps if r["_cat"] == "SIZE_LIMIT")
        n_unavailable = sum(1 for r in reps if r["_cat"] == "UNAVAILABLE")
        n_license_error = sum(1 for r in reps if r["_cat"] == "LICENSE_ERROR")
        n_unknown = sum(1 for r in reps if r["_cat"] == "UNKNOWN")

        n_algorithmic = n_optimal + n_timeout + n_interrupted + n_feasible + n_infeasible + n_error

        # Compute per-rep PAR2 values strictly over eligible algorithmic runs
        rep_par2_vals = []
        for r in reps:
            cat = r["_cat"]
            if cat == "OPTIMAL":
                t = r.get("solve_time")
                if t is None:
                    t = r.get("wall_time")
                if t is not None:
                    try:
                        ft = float(t)
                        if ft >= 0 and not math.isnan(ft) and not math.isinf(ft):
                            rep_par2_vals.append(ft)
                        else:
                            t_lim = extract_time_limit(r)
                            rep_par2_vals.append(2.0 * t_lim)
                    except (ValueError, TypeError):
                        t_lim = extract_time_limit(r)
                        rep_par2_vals.append(2.0 * t_lim)
                else:
                    t_lim = extract_time_limit(r)
                    rep_par2_vals.append(2.0 * t_lim)
            elif cat in ("TIMEOUT", "INTERRUPTED", "FEASIBLE", "INFEASIBLE", "ERROR"):
                t_lim = extract_time_limit(r)
                rep_par2_vals.append(2.0 * t_lim)
            else:
                # Non-algorithmic exclusions: SIZE_LIMIT, UNAVAILABLE, LICENSE_ERROR, UNKNOWN
                pass

        # Determine aggregated status
        if n_optimal == n_total:
            agg_status = "OPTIMAL"
        elif n_optimal > 0:
            agg_status = "PARTIAL"
        elif n_algorithmic > 0:
            if n_feasible > 0:
                agg_status = "FEASIBLE"
            elif n_timeout > 0:
                agg_status = "TIMEOUT"
            elif n_interrupted > 0:
                agg_status = "INTERRUPTED"
            elif n_infeasible > 0:
                agg_status = "INFEASIBLE"
            else:
                agg_status = "ERROR"
        else:
            if n_size_limit > 0:
                agg_status = "SIZE_LIMIT"
            elif n_license_error > 0:
                agg_status = "LICENSE_ERROR"
            elif n_unavailable > 0:
                agg_status = "UNAVAILABLE"
            else:
                agg_status = "UNKNOWN"

        # Select representative record to seed the aggregated dictionary
        if agg_status in ("OPTIMAL", "PARTIAL"):
            opt_reps = [r for r in reps if r["_cat"] == "OPTIMAL"]
            reps_sorted = sorted(
                opt_reps,
                key=lambda x: x.get("solve_time") if x.get("solve_time") is not None else float("inf"),
            )
            rep_chosen = reps_sorted[len(reps_sorted) // 2]
            solve_times = [r["solve_time"] for r in opt_reps if r.get("solve_time") is not None]
            wall_times = [r["wall_time"] for r in opt_reps if r.get("wall_time") is not None]
        elif agg_status == "FEASIBLE":
            feas_reps = [r for r in reps if r["_cat"] == "FEASIBLE"]
            reps_sorted = sorted(
                feas_reps,
                key=lambda x: (x.get("best_ub", float("inf")), x.get("solve_time", float("inf"))),
            )
            rep_chosen = reps_sorted[0]
            solve_times = [r["solve_time"] for r in feas_reps if r.get("solve_time") is not None]
            wall_times = [r["wall_time"] for r in feas_reps if r.get("wall_time") is not None]
        else:
            matching = [r for r in reps if r["_cat"] == agg_status]
            candidates = matching if matching else reps
            reps_sorted = sorted(
                candidates,
                key=lambda x: x.get("solve_time") if x.get("solve_time") is not None else float("inf"),
            )
            rep_chosen = reps_sorted[len(reps_sorted) // 2]
            solve_times = [r["solve_time"] for r in candidates if r.get("solve_time") is not None]
            wall_times = [r["wall_time"] for r in candidates if r.get("wall_time") is not None]

        encode_times = [r["encode_time"] for r in reps if r.get("encode_time") is not None]

        agg = dict(rep_chosen)
        agg["status"] = agg_status
        if agg_status == "PARTIAL":
            agg["mixed"] = True
            agg["is_mixed"] = True

        # Attach unified population counters
        agg["n_total"] = n_total
        agg["n_algorithmic"] = n_algorithmic
        agg["n_optimal"] = n_optimal
        agg["n_timeout"] = n_timeout
        agg["n_interrupted"] = n_interrupted
        agg["n_feasible"] = n_feasible
        agg["n_infeasible"] = n_infeasible
        agg["n_error"] = n_error
        agg["n_size_limit"] = n_size_limit
        agg["n_unavailable"] = n_unavailable
        agg["n_license_error"] = n_license_error
        agg["n_unknown"] = n_unknown

        if solve_times:
            agg["solve_time"] = float(np.median(solve_times))
        elif "solve_time" not in agg or agg["solve_time"] is None:
            agg["solve_time"] = 0.0

        if wall_times:
            agg["wall_time"] = float(np.median(wall_times))
        if encode_times:
            agg["encode_time"] = float(np.median(encode_times))

        if rep_par2_vals:
            agg["par2"] = float(np.mean(rep_par2_vals))
        else:
            agg.pop("par2", None)

        if agg_status in ("ERROR", "NUMERIC_ERROR", "LICENSE_ERROR", "UNAVAILABLE", "SIZE_LIMIT", "INFEASIBLE", "UNKNOWN") or agg.get("validation_error"):
            agg["best_ub"] = -1
            agg["assignment"] = None

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
    elif os.path.exists("results_v2/runs.jsonl"):
        files = ["results_v2/runs.jsonl"]
    elif os.path.exists("results/runs.jsonl"):
        files = ["results/runs.jsonl"]
    else:
        files = sorted(glob.glob("results_v2/*.jsonl")) or sorted(glob.glob("results/*.jsonl"))

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

def get_source_key(record):
    """Extracts (solver_or_method, canonical_config) from a record."""
    sol = str(record.get("solver") or record.get("method") or "unknown").lower()
    cfg = record.get("config") if isinstance(record.get("config"), dict) else {}

    if sol in ("cp_sat", "cpsat"):
        solver_name = "cpsat"
        num_w = cfg.get("num_workers", record.get("num_workers", 1))
        canonical_config = ("cpsat", num_w)
    elif sol == "cplex":
        solver_name = "cplex"
        canonical_config = ("cplex",)
    else:
        solver_name = sol
        rot = bool(cfg.get("sb_rotation", record.get("sb_rotation", False)))
        ref = bool(cfg.get("sb_reflection", record.get("sb_reflection", False)))
        heur = bool(cfg.get("heuristic_ub", record.get("heuristic_ub", False)))
        enc = str(record.get("encoding") or cfg.get("encoding", "order")).lower()
        mode = str(record.get("solving_mode") or cfg.get("solving_mode", "incremental")).lower()
        eo = str(record.get("eo_strategy") or cfg.get("eo_strategy", "commander")).lower()
        variant = cfg.get("variant", record.get("variant"))
        extra_flags = (enc, mode, eo) if (enc != "order" or mode != "incremental" or eo != "commander") else ()
        if variant:
            canonical_config = (str(variant), rot, ref, heur) + extra_flags
        else:
            canonical_config = (rot, ref, heur) + extra_flags

    return (solver_name, canonical_config)

def cross_validate_optima(records, output_dir=None):
    """
    Cross-validation check:
    1. Group OPTIMAL records by (dataset, relabel_seed) and (solver_or_method, canonical_config).
       For a given (dataset, relabel_seed), verified only when >= 2 distinct keys reach identical OPTIMAL best_ub.
       Repetitions of the same key are deduplicated, not counted as separate sources.
       If only 1 distinct key is present, mark as 'single_source' and warn.
       If distinct keys report conflicting optima, mark as 'conflict' and raise ValueError.
    2. Check across seeds: all seeds of the same base dataset must reach the exact same optimum.
       If different seeds report conflicting optima, mark as 'conflict' and raise ValueError.
    3. Check incumbents: for any dataset with a proven optimum `opt`, every non-OPTIMAL record
       with best_ub >= 0 must satisfy best_ub >= opt; otherwise raises ValueError with the offending record.
    4. Exports cross_validation.csv to output_dir if output_dir is provided.
    """
    optimal_by_inst = {}
    for r in records:
        if r.get("status") == "OPTIMAL":
            ds = r.get("dataset")
            seed = r.get("relabel_seed")
            ub = r.get("best_ub")
            src_key = get_source_key(r)
            optimal_by_inst.setdefault((ds, seed), {}).setdefault(src_key, []).append(ub)

    distinct_sources = {}
    mismatches = []
    proven_optima_by_inst = {}
    proven_optima_by_ds = {}

    for (ds, seed), sources in optimal_by_inst.items():
        distinct_sources[(ds, seed)] = {}
        for src_key, ubs in sources.items():
            unique_key_ubs = set(u for u in ubs if u is not None)
            if len(unique_key_ubs) > 1:
                mismatches.append((ds, f"Internal conflict across repetitions of key {src_key} on seed {seed}: {unique_key_ubs}"))
            distinct_sources[(ds, seed)][src_key] = list(unique_key_ubs)[0] if unique_key_ubs else None

        all_inst_ubs = set(u for u in distinct_sources[(ds, seed)].values() if u is not None)
        if len(all_inst_ubs) > 1:
            mismatches.append((ds, f"Conflicting optima across sources on {ds} (seed {seed}): {distinct_sources[(ds, seed)]}"))
        elif len(all_inst_ubs) == 1:
            opt = list(all_inst_ubs)[0]
            proven_optima_by_inst[(ds, seed)] = opt
            if ds in proven_optima_by_ds and proven_optima_by_ds[ds] != opt:
                mismatches.append((ds, f"Conflicting optima across seeds for dataset '{ds}': {proven_optima_by_ds[ds]} vs {opt} (seed {seed})"))
            else:
                proven_optima_by_ds[ds] = opt

    # Check incumbents against proven optima
    incumbent_violations = []
    for r in records:
        ds = r.get("dataset")
        st = r.get("status")
        if ds in proven_optima_by_ds and st != "OPTIMAL":
            ub = r.get("best_ub")
            if ub is not None and ub >= 0:
                opt = proven_optima_by_ds[ds]
                if ub < opt:
                    incumbent_violations.append((ds, ub, opt, r))

    # Prepare DataFrame rows
    conflicting_ds = set(m[0] for m in mismatches)
    compared_rows = []
    has_any_seeds = any(seed is not None for (ds, seed) in distinct_sources.keys())
    for (ds, seed), sources in distinct_sources.items():
        all_inst_ubs = set(u for u in sources.values() if u is not None)
        has_conflict = len(all_inst_ubs) > 1 or ds in conflicting_ds
        num_keys = len(sources)
        if has_conflict:
            ds_status = "conflict"
        elif num_keys >= 2:
            ds_status = "verified"
        else:
            ds_status = "single_source"

        for (solver_name, canonical_cfg), ub in sources.items():
            row_dict = {
                "dataset": ds,
                "solver_or_method": solver_name,
                "canonical_config": str(canonical_cfg),
                "best_ub": ub,
                "num_distinct_sources": num_keys,
                "validation_status": ds_status,
            }
            if has_any_seeds:
                row_dict["relabel_seed"] = seed
            compared_rows.append(row_dict)

    cols = ["dataset"]
    if has_any_seeds:
        cols.append("relabel_seed")
    cols.extend(["solver_or_method", "canonical_config", "best_ub", "num_distinct_sources", "validation_status"])

    df_cv = pd.DataFrame(
        compared_rows,
        columns=cols
    )
    if "relabel_seed" in df_cv.columns:
        df_cv["relabel_seed"] = pd.Series([r.get("relabel_seed") for r in compared_rows], dtype="Int64")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "cross_validation.csv")
        df_cv.to_csv(csv_path, index=False)
        print(f"Cross-validation table saved: {csv_path}")

    # Report / raise errors
    if mismatches:
        print("\n" + "!" * 80)
        print("[CROSS-VALIDATION FAILED] Conflicting optimal solutions detected:")
        for ds, err_msg in mismatches:
            print(f"  Dataset '{ds}': {err_msg}")
        print("!" * 80 + "\n")
        raise ValueError(f"Cross-validation failed: conflicting optima on {len(mismatches)} dataset(s).")

    if incumbent_violations:
        print("\n" + "!" * 80)
        print("[CROSS-VALIDATION FAILED] Non-optimal record reported best_ub < proven optimum:")
        for ds, ub, opt, rec in incumbent_violations:
            print(f"  Dataset '{ds}': best_ub={ub} < proven optimum opt={opt}. Record: {rec}")
        print("!" * 80 + "\n")
        ds, ub, opt, rec = incumbent_violations[0]
        raise ValueError(f"Invalid incumbent on dataset '{ds}': best_ub={ub} < proven optimum opt={opt}. Record: {rec}")

    verified_count = sum(1 for (ds, seed), sources in distinct_sources.items() if len(sources) >= 2)
    single_source_count = sum(1 for (ds, seed), sources in distinct_sources.items() if len(sources) == 1)

    for (ds, seed), sources in distinct_sources.items():
        if len(sources) == 1:
            inst_label = ds if seed is None else f"{ds} (seed={seed})"
            opt = proven_optima_by_inst.get((ds, seed))
            print(f"[WARNING] Dataset '{inst_label}' has only 1 distinct configuration/solver source: best_ub = {opt} (single_source). Not independently cross-verified.")

    if single_source_count > 0 and verified_count == 0:
        print(f"[CROSS-VALIDATION INCOMPLETE] 0 dataset(s) independently verified; {single_source_count} dataset(s) have single_source only.")
    elif verified_count > 0:
        single_msg = f" ({single_source_count} dataset(s) have single_source only)" if single_source_count > 0 else ""
        print(f"[CROSS-VALIDATION PASSED] {verified_count} dataset(s) verified by >= 2 independent sources{single_msg}. All optima match identically.")

    return df_cv

def format_cell_value(status, solve_time=0.0, best_ub=None, n_optimal=None, n_total=None, n_algorithmic=None):
    """
    Format single cell according to required status representations:
      - OPTIMAL: solve_time in seconds (e.g. "0.0123")
      - PARTIAL: "k/n time*" (e.g. "3/5 0.1234*")
      - FEASIBLE: "FEAS(ub)" or "FEAS"
      - TIMEOUT: "TO(ub)" or "TO"
      - INTERRUPTED: "INT(ub)" or "INT"
      - SIZE_LIMIT: "SL"
      - UNAVAILABLE: "UNAV"
      - LICENSE_ERROR: "LIC"
      - INFEASIBLE: "UNSAT"
      - NUMERIC_ERROR: "NUM_ERR"
      - ERROR: "ERR"
    """
    if isinstance(status, dict):
        rec = status
        status = rec.get("status")
        solve_time = rec.get("solve_time", 0.0)
        best_ub = rec.get("best_ub")
        n_optimal = rec.get("n_optimal")
        n_total = rec.get("n_total")
        n_algorithmic = rec.get("n_algorithmic")

    if solve_time is None:
        solve_time = 0.0

    st = str(status).upper() if status is not None else ""

    if st == "OPTIMAL":
        return f"{solve_time:.4f}"
    elif st == "PARTIAL":
        denom = n_total if (n_total is not None and n_total > 0) else n_algorithmic
        k_n = f"{n_optimal}/{denom}" if (n_optimal is not None and denom is not None) else "PARTIAL"
        return f"{k_n} {solve_time:.4f}*"
    elif st == "FEASIBLE":
        return f"FEAS({best_ub})" if best_ub is not None and best_ub >= 0 else "FEAS"
    elif st == "TIMEOUT":
        return f"TO({best_ub})" if best_ub is not None and best_ub >= 0 else "TO"
    elif st == "INTERRUPTED":
        return f"INT({best_ub})" if best_ub is not None and best_ub >= 0 else "INT"
    elif st == "SIZE_LIMIT":
        return "SL"
    elif st == "UNAVAILABLE":
        return "UNAV"
    elif st == "LICENSE_ERROR":
        return "LIC"
    elif st in ("INFEASIBLE", "UNSAT"):
        return "UNSAT"
    elif st == "NUMERIC_ERROR":
        return "NUM_ERR"
    elif st == "ERROR":
        return "ERR"
    elif st == "UNKNOWN":
        return "UNK"
    elif not st or st == "NONE":
        return "-"
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
                t_base = rec.get("solve_time", timeout)
                if t_base >= 0.1:
                    baseline_times[ds] = t_base

    for col in column_names:
        solved = 0
        sl_count = 0
        par2_vals = []
        ratios = []

        for ds, row in df_raw.iterrows():
            rec = row.get(col)
            if not rec:
                continue
            st = str(rec.get("status", "")).upper()
            t = rec.get("solve_time", 0.0) or 0.0
            rec_timeout = extract_time_limit(rec, default_timeout=timeout)

            if st == "OPTIMAL":
                solved += 1
                par2_val = rec.get("par2") if ("par2" in rec and rec.get("par2") is not None) else t
                par2_vals.append(par2_val)
                if ds in baseline_times and t > 0:
                    ratios.append(baseline_times[ds] / t)
            elif st == "PARTIAL":
                if "par2" in rec and rec.get("par2") is not None:
                    par2_vals.append(rec["par2"])
                else:
                    n_opt = rec.get("n_optimal", 0)
                    n_alg = rec.get("n_algorithmic", rec.get("n_total", 1))
                    n_alg = max(n_alg, 1)
                    par2_vals.append((n_opt * t + (n_alg - n_opt) * 2.0 * float(rec_timeout)) / n_alg)
            elif st == "SIZE_LIMIT":
                sl_count += 1
                # SIZE_LIMIT is reported separately in SL and not penalized in PAR2 as timeout
            elif st in ("UNAVAILABLE", "LICENSE_ERROR", "UNKNOWN"):
                # Solver environment unavailability, missing license, or unknown status excluded from algorithmic PAR2
                pass
            else:
                par2_val = rec.get("par2") if ("par2" in rec and rec.get("par2") is not None) else (2.0 * float(rec_timeout))
                par2_vals.append(par2_val)

        solved_counts[col] = solved
        sl_counts[col] = sl_count
        par2_scores[col] = float(np.mean(par2_vals)) if par2_vals else float("nan")

        n_eligible = len(baseline_times)
        if n_eligible == 0:
            speedups[col] = float("nan")
        elif col == baseline_col:
            speedups[col] = 1.00
        elif len(ratios) == n_eligible:
            speedups[col] = float(np.exp(np.mean(np.log(ratios))))
        else:
            speedups[col] = float("nan")

    return solved_counts, sl_counts, par2_scores, speedups

def export_table(df_display, summary_rows, title, base_filename, output_dirs=("results_v2/tables",), note=None):
    """Saves formatted table as both CSV and LaTeX."""
    if isinstance(output_dirs, str):
        output_dirs = (output_dirs,)
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

def make_table_ablation(records, group_name="ablation_small", base_filename="table_ablation_small", title="Ablation Study: Small Graphs", output_dirs=("results_v2/tables",)):
    """Generate Ablation study table for a specific ablation group."""
    ablation_recs = [
        r for r in records
        if group_name in r.get("groups", []) or (group_name == "ablation_small" and "ablation" in r.get("groups", []))
    ]
    if not ablation_recs:
        print(f"[WARNING] Skipping {base_filename}: no records found for group '{group_name}'.")
        return None

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
    variants_present = set(get_var_name(r.get("config", {})) for r in ablation_recs)
    if not all(col in variants_present for col in columns):
        print(f"[WARNING] Skipping {base_filename}: not all variants present in records (need {columns}).")
        return None

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
    n_used = sum(1 for ds, row in df_raw.iterrows() if row.get("none") and row.get("none").get("status") == "OPTIMAL" and (row.get("none").get("solve_time") or 0.0) >= 0.1)
    n_total = len(df_raw)

    sum_solved = {"N": "-", "|E|": "-"}
    sum_sl = {"N": "-", "|E|": "-"}
    sum_par2 = {"N": "-", "|E|": "-"}
    sum_spdup = {"N": "-", "|E|": "-"}

    has_timeout = any(solved_cnt[col] + sl_cnt[col] < len(ds_sorted) for col in columns)
    par2_label = "PAR2 (s)" if has_timeout else "mean time (s)"

    for col in columns:
        sum_solved[col] = str(solved_cnt[col])
        sum_sl[col] = str(sl_cnt[col])
        if par2_label == "mean time (s)" and solved_cnt[col] < len(ds_sorted):
            sum_par2[col] = "-"
        else:
            sum_par2[col] = f"{par2_val[col]:.3f}" if not math.isnan(par2_val[col]) else "-"
        sum_spdup[col] = f"{spdups[col]:.2f}x" if not math.isnan(spdups[col]) else "-"

    spdup_label = f"Speedup (geom) [{n_used}/{n_total}]" if n_used > 0 else "Speedup (geom)"

    summary = [
        ("Solved", sum_solved),
    ]
    if any(v > 0 for v in sl_cnt.values()):
        summary.append(("SL", sum_sl))
    summary.extend([
        (par2_label, sum_par2),
        (spdup_label, sum_spdup),
    ])

    note = f"Speedup chỉ tính trên instance mà baseline >= 0.1s ({n_used}/{n_total}). Các variant chưa giải hết các instance này ghi '-'."
    export_table(df_display, summary, title, base_filename, output_dirs=output_dirs, note=note)

def make_table_solvers(records, group_name="solvers_small", base_filename="table_solvers_small", title="SAT Solver Performance Comparison (Small Graphs)", output_dirs=("results_v2/tables",)):
    """Generate Solvers comparison table for a specific solvers group."""
    solver_recs = [
        r for r in records
        if group_name in r.get("groups", []) or (group_name == "solvers_small" and "solvers" in r.get("groups", []))
    ]
    if not solver_recs:
        return

    solvers = ["cadical195", "glucose4", "maplechrono", "minisat22", "lingeling"]

    recs_by_ds_sol = {}
    meta = {}
    for r in solver_recs:
        ds = r["dataset"]
        meta[ds] = (r.get("N", 0), r.get("num_edges", 0))
        sol = r["solver"].lower()
        recs_by_ds_sol.setdefault((ds, sol), []).append(r)

    data_by_ds = {}
    for (ds, sol), rec_list in recs_by_ds_sol.items():
        if len(rec_list) == 1:
            data_by_ds.setdefault(ds, {})[sol] = rec_list[0]
        else:
            n_optimal = sum(1 for x in rec_list if x.get("status") == "OPTIMAL")
            n_total = len(rec_list)
            solve_times = [x.get("solve_time", 0.0) for x in rec_list if x.get("solve_time") is not None]
            wall_times = [x.get("wall_time", 0.0) for x in rec_list if x.get("wall_time") is not None]
            par2_vals = [x.get("par2", (x.get("solve_time") if x.get("status") == "OPTIMAL" else 2.0 * float(x.get("config", {}).get("time_limit", 600.0)))) for x in rec_list]

            if n_optimal == n_total:
                chosen = sorted(rec_list, key=lambda x: x.get("solve_time", 0.0))[len(rec_list) // 2]
                agg = dict(chosen)
                agg["status"] = "OPTIMAL"
                agg["n_optimal"] = n_optimal
                agg["n_total"] = n_total
                agg["solve_time"] = float(np.median(solve_times)) if solve_times else 0.0
                agg["wall_time"] = float(np.median(wall_times)) if wall_times else 0.0
                agg["par2"] = float(np.mean(par2_vals)) if par2_vals else float("nan")
            elif n_optimal > 0:
                opt_recs = [x for x in rec_list if x.get("status") == "OPTIMAL"]
                chosen = sorted(opt_recs, key=lambda x: x.get("solve_time", 0.0))[len(opt_recs) // 2]
                agg = dict(chosen)
                agg["status"] = "PARTIAL"
                agg["n_optimal"] = n_optimal
                agg["n_total"] = n_total
                agg["solve_time"] = float(np.median([x.get("solve_time", 0.0) for x in opt_recs]))
                agg["wall_time"] = float(np.median([x.get("wall_time", 0.0) for x in opt_recs]))
                agg["par2"] = float(np.mean(par2_vals)) if par2_vals else float("nan")
            else:
                chosen = sorted(rec_list, key=lambda x: x.get("solve_time", 0.0))[len(rec_list) // 2]
                agg = dict(chosen)
                agg["n_optimal"] = 0
                agg["n_total"] = n_total
                agg["solve_time"] = float(np.median(solve_times)) if solve_times else 0.0
                agg["wall_time"] = float(np.median(wall_times)) if wall_times else 0.0
                agg["par2"] = float(np.mean(par2_vals)) if par2_vals else float("nan")
            data_by_ds.setdefault(ds, {})[sol] = agg

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
    n_used = sum(1 for ds, row in df_raw.iterrows() if row.get("cadical195") and row.get("cadical195").get("status") == "OPTIMAL" and (row.get("cadical195").get("solve_time") or 0.0) >= 0.1)
    n_total = len(df_raw)

    sum_solved = {"N": "-", "|E|": "-"}
    sum_sl = {"N": "-", "|E|": "-"}
    sum_par2 = {"N": "-", "|E|": "-"}
    sum_spdup = {"N": "-", "|E|": "-"}

    has_timeout = any(solved_cnt[sol] + sl_cnt[sol] < len(ds_sorted) for sol in solvers)
    par2_label = "PAR2 (s)" if has_timeout else "mean time (s)"

    for sol in solvers:
        sum_solved[sol] = str(solved_cnt[sol])
        sum_sl[sol] = str(sl_cnt[sol])
        if par2_label == "mean time (s)" and solved_cnt[sol] < len(ds_sorted):
            sum_par2[sol] = "-"
        else:
            sum_par2[sol] = f"{par2_val[sol]:.3f}" if not math.isnan(par2_val[sol]) else "-"
        sum_spdup[sol] = f"{spdups[sol]:.2f}x" if not math.isnan(spdups[sol]) else "-"

    spdup_label = f"Speedup (geom) [{n_used}/{n_total}]" if n_used > 0 else "Speedup (geom)"

    summary = [
        ("Solved", sum_solved),
    ]
    if any(v > 0 for v in sl_cnt.values()):
        summary.append(("SL", sum_sl))
    summary.extend([
        (par2_label, sum_par2),
        (spdup_label, sum_spdup),
    ])

    speedup_note = f"Speedup chỉ tính trên instance mà baseline >= 0.1s ({n_used}/{n_total}). Các solver chưa giải hết các instance này ghi '-'."
    if group_name == "solvers_small":
        note = f"Thời gian giải là trung vị (median) trên 5 hoán vị nhãn (seeds 0..4). PAR2 tính trung bình (mean) trên 5 seed. {speedup_note}"
    else:
        note = speedup_note

    export_table(df_display, summary, title, base_filename, output_dirs=output_dirs, note=note)

def make_table_methods(records, output_dirs=("results_v2/tables",)):
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
    n_used = sum(1 for ds, row in df_raw.iterrows() if row.get("sat_best") and row.get("sat_best").get("status") == "OPTIMAL" and (row.get("sat_best").get("solve_time") or 0.0) >= 0.1)
    n_total = len(df_raw)

    sum_solved = {"N": "-", "|E|": "-"}
    sum_sl = {"N": "-", "|E|": "-"}
    sum_par2 = {"N": "-", "|E|": "-"}
    sum_spdup = {"N": "-", "|E|": "-"}

    has_timeout = any(solved_cnt[m] + sl_cnt[m] < len(ds_sorted) for m in methods)
    par2_label = "PAR2 (s)" if has_timeout else "mean time (s)"

    for m in methods:
        sum_solved[m] = str(solved_cnt[m])
        sum_sl[m] = str(sl_cnt[m])
        if par2_label == "mean time (s)" and solved_cnt[m] < len(ds_sorted):
            sum_par2[m] = "-"
        else:
            sum_par2[m] = f"{par2_val[m]:.3f}" if not math.isnan(par2_val[m]) else "-"
        sum_spdup[m] = f"{spdups[m]:.2f}x" if not math.isnan(spdups[m]) else "-"

    spdup_label = f"Speedup (geom) [{n_used}/{n_total}]" if n_used > 0 else "Speedup (geom)"

    summary = [
        ("Solved", sum_solved),
        ("SL", sum_sl),
        (par2_label, sum_par2),
        (spdup_label, sum_spdup),
    ]

    note = f"SL = Size Limit (vượt giới hạn CPLEX Community Edition, loại khỏi PAR2). Speedup chỉ tính trên instance mà baseline >= 0.1s ({n_used}/{n_total}). Các method chưa giải hết các instance này ghi '-'."
    export_table(df_display, summary, "Comparison of Methods: SAT vs CP-SAT vs CPLEX", "table_methods", output_dirs=output_dirs, note=note)

def make_table_ablation_family(records, base_filename="table_ablation_family", output_dirs=("results_v2/tables",)):
    """
    Generate table_ablation_family.csv:
    Each row: (comparison_group, variant): median time ratio, IQR (25-75%), min-max, faster_pairs / total_pairs.
    Comparison groups:
      - variant vs none (for rotation, rot_ref, heur_ub, all)
      - rotation_reflection vs rotation
      - all vs rotation_reflection
      - heuristic_ub vs none
    Pairs: matching (dataset, relabel_seed).
    """
    if isinstance(output_dirs, str):
        output_dirs = (output_dirs,)

    def get_var(rec):
        cfg = rec.get("config", {})
        rot = cfg.get("sb_rotation", False)
        ref = cfg.get("sb_reflection", False)
        heur = cfg.get("heuristic_ub", False)
        if not rot and not ref and not heur:
            return "none"
        elif rot and not ref and not heur:
            return "rotation"
        elif rot and ref and not heur:
            return "rotation_reflection"
        elif not rot and not ref and heur:
            return "heuristic_ub"
        elif rot and ref and heur:
            return "all"
        return "other"

    family_recs = [r for r in records if r.get("relabel_seed") is not None]
    if not family_recs:
        print("[WARNING] Skipping table_ablation_family: no records with relabel_seed found.")
        return None

    pairs_map = {}
    for r in family_recs:
        ds = r.get("dataset")
        seed = r.get("relabel_seed")
        var = get_var(r)
        pairs_map.setdefault((ds, seed), {})[var] = r

    comparisons = [
        ("variant vs none", "rotation", "rotation", "none"),
        ("variant vs none", "rotation_reflection", "rotation_reflection", "none"),
        ("variant vs none", "heuristic_ub", "heuristic_ub", "none"),
        ("variant vs none", "all", "all", "none"),
        ("rotation_reflection vs rotation", "rotation_reflection", "rotation_reflection", "rotation"),
        ("all vs rotation_reflection", "all", "all", "rotation_reflection"),
        ("all vs rotation", "all", "all", "rotation"),
    ]

    # Condition: >= 1 valid pair for EACH comparison
    for comp_group, var_label, var_target, var_ref in comparisons:
        n_pairs = sum(
            1 for (ds, seed), vmap in pairs_map.items()
            if var_target in vmap and var_ref in vmap
            and vmap[var_target].get("status") == "OPTIMAL"
            and vmap[var_ref].get("status") == "OPTIMAL"
            and vmap[var_ref].get("solve_time", 0.0) > 0
        )
        if n_pairs < 1:
            print(f"[WARNING] Skipping table_ablation_family: comparison '{var_target} vs {var_ref}' has 0 valid pairs (>=1 required).")
            return None

    rows = []
    for comp_group, var_label, var_target, var_ref in comparisons:
        ratios = []
        faster_count = 0
        total_count = 0
        for (ds, seed), vmap in pairs_map.items():
            if var_target in vmap and var_ref in vmap:
                rec_t = vmap[var_target]
                rec_r = vmap[var_ref]
                if rec_t.get("status") == "OPTIMAL" and rec_r.get("status") == "OPTIMAL":
                    t_target = rec_t.get("solve_time", 0.0)
                    t_ref = rec_r.get("solve_time", 0.0)
                    if t_ref > 0:
                        ratio = t_target / t_ref
                        ratios.append(ratio)
                        total_count += 1
                        if t_target < t_ref:
                            faster_count += 1

        if ratios:
            med = float(np.median(ratios))
            q25 = float(np.percentile(ratios, 25))
            q75 = float(np.percentile(ratios, 75))
            min_r = float(np.min(ratios))
            max_r = float(np.max(ratios))
            iqr_str = f"[{q25:.4f}, {q75:.4f}]"
            min_max_str = f"[{min_r:.4f}, {max_r:.4f}]"
            faster_str = f"{faster_count}/{total_count}"
        else:
            med = float("nan")
            iqr_str = "-"
            min_max_str = "-"
            faster_str = "0/0"

        rows.append({
            "comparison_group": comp_group,
            "variant": var_label,
            "median_time_ratio": f"{med:.4f}" if not math.isnan(med) else "-",
            "iqr_25_75": iqr_str,
            "min_max": min_max_str,
            "faster_pairs": faster_str,
        })

    df = pd.DataFrame(rows)
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{base_filename}.csv")
        df.to_csv(csv_path, index=False)
    print(f"Table saved: {base_filename}.csv in {output_dirs[0]}/")

    # Generate table_ablation_family_per_graph.csv
    datasets = sorted(list(set(ds for (ds, seed) in pairs_map.keys())))
    can_generate_pg = True
    if not datasets:
        can_generate_pg = False
    else:
        for ds in datasets:
            for comp_group, var_label, var_target, var_ref in comparisons:
                count = sum(
                    1 for (d, seed), vmap in pairs_map.items()
                    if d == ds and var_target in vmap and var_ref in vmap
                    and vmap[var_target].get("status") == "OPTIMAL"
                    and vmap[var_ref].get("status") == "OPTIMAL"
                    and vmap[var_ref].get("solve_time", 0.0) > 0
                )
                if count < 1:
                    can_generate_pg = False
                    break
            if not can_generate_pg:
                break

    if not can_generate_pg:
        print("[WARNING] Skipping table_ablation_family_per_graph: not enough valid pairs for each comparison per graph (>=1 required).")
    else:
        per_graph_rows = []
        for ds in datasets:
            for comp_group, var_label, var_target, var_ref in comparisons:
                comp_name = f"{var_target} vs {var_ref}"
                ratios = []
                faster_count = 0
                total_count = 0
                for (d, seed), vmap in pairs_map.items():
                    if d == ds and var_target in vmap and var_ref in vmap:
                        rec_t = vmap[var_target]
                        rec_r = vmap[var_ref]
                        if rec_t.get("status") == "OPTIMAL" and rec_r.get("status") == "OPTIMAL":
                            t_target = rec_t.get("solve_time", 0.0)
                            t_ref = rec_r.get("solve_time", 0.0)
                            if t_ref > 0:
                                ratio = t_target / t_ref
                                ratios.append(ratio)
                                total_count += 1
                                if t_target < t_ref:
                                    faster_count += 1
                if ratios:
                    med = float(np.median(ratios))
                    min_r = float(np.min(ratios))
                    max_r = float(np.max(ratios))
                    med_str = f"{med:.4f}"
                    min_str = f"{min_r:.4f}"
                    max_str = f"{max_r:.4f}"
                    faster_str = f"{faster_count}/{total_count}"
                else:
                    med_str = "-"
                    min_str = "-"
                    max_str = "-"
                    faster_str = "0/0"

                per_graph_rows.append({
                    "dataset": ds,
                    "comparison": comp_name,
                    "median_time_ratio": med_str,
                    "min": min_str,
                    "max": max_str,
                    "faster_pairs": faster_str,
                })

        df_per_graph = pd.DataFrame(per_graph_rows)
        for out_dir in output_dirs:
            os.makedirs(out_dir, exist_ok=True)
            csv_path_pg = os.path.join(out_dir, "table_ablation_family_per_graph.csv")
            df_per_graph.to_csv(csv_path_pg, index=False)
        print(f"Table saved: table_ablation_family_per_graph.csv in {output_dirs[0]}/")

    return df

def make_table_ablation_phases(records, base_filename="table_ablation_phases", output_dirs=("results_v2/tables",)):
    """
    Generate table_ablation_phases.csv:
    Median last_iter_time, median prev_iters_time, ratio of last_iter/solve_time by variant, split by N (40 and 50).
    """
    if isinstance(output_dirs, str):
        output_dirs = (output_dirs,)

    def get_var(rec):
        cfg = rec.get("config", {})
        rot = cfg.get("sb_rotation", False)
        ref = cfg.get("sb_reflection", False)
        heur = cfg.get("heuristic_ub", False)
        if not rot and not ref and not heur:
            return "none"
        elif rot and not ref and not heur:
            return "rotation"
        elif rot and ref and not heur:
            return "rotation_reflection"
        elif not rot and not ref and heur:
            return "heuristic_ub"
        elif rot and ref and heur:
            return "all"
        return "other"

    family_recs = [r for r in records if r.get("relabel_seed") is not None]
    if not family_recs:
        print("[WARNING] Skipping table_ablation_phases: no records with relabel_seed found.")
        return None

    variants = ["none", "rotation", "rotation_reflection", "heuristic_ub", "all"]
    sizes = sorted(list(set(r.get("N", 0) for r in family_recs if r.get("N"))))
    if not sizes:
        print("[WARNING] Skipping table_ablation_phases: no sizes found.")
        return None

    # Condition: each (size, variant) must have >= 1 valid OPTIMAL record
    can_generate_phases = True
    for n in sizes:
        for var in variants:
            count = sum(1 for r in family_recs if r.get("N") == n and get_var(r) == var and r.get("status") == "OPTIMAL")
            if count < 1:
                can_generate_phases = False
                break
        if not can_generate_phases:
            break

    if not can_generate_phases:
        print("[WARNING] Skipping table_ablation_phases: not all (size, variant) pairs have >= 1 valid OPTIMAL record.")
        return None

    rows = []
    for n in sizes:
        for var in variants:
            last_times = []
            prev_times = []
            ratios = []

            for r in family_recs:
                if r.get("N") == n and get_var(r) == var and r.get("status") == "OPTIMAL":
                    per_iter = r.get("per_iter_times", [])
                    st = r.get("solve_time", 0.0)
                    if per_iter:
                        last_t = per_iter[-1]
                        prev_t = sum(per_iter[:-1])
                    else:
                        last_t = 0.0
                        prev_t = 0.0
                    last_times.append(last_t)
                    prev_times.append(prev_t)
                    if st > 0:
                        ratios.append(last_t / st)

            med_last = float(np.median(last_times)) if last_times else float("nan")
            med_prev = float(np.median(prev_times)) if prev_times else float("nan")
            med_ratio = float(np.median(ratios)) if ratios else float("nan")

            rows.append({
                "N": n,
                "variant": var,
                "median_last_iter_time": f"{med_last:.4f}" if not math.isnan(med_last) else "-",
                "median_prev_iters_time": f"{med_prev:.4f}" if not math.isnan(med_prev) else "-",
                "last_iter_ratio": f"{med_ratio:.4f}" if not math.isnan(med_ratio) else "-",
            })

    df = pd.DataFrame(rows)
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{base_filename}.csv")
        df.to_csv(csv_path, index=False)
    print(f"Table saved: {base_filename}.csv in {output_dirs[0]}/")
    return df

def get_lb_degree(dataset_name):
    """Compute lb_degree = ceil(Delta / 2) from datasets/<dataset_name>.mtx, excluding self-loops."""
    filepath = f"datasets/{dataset_name}.mtx" if not dataset_name.endswith(".mtx") else dataset_name
    if not os.path.exists(filepath):
        return 0
    try:
        from scipy.io import mmread
        try:
            mat = mmread(filepath, spmatrix=True).tocsr()
        except TypeError:
            mat = mmread(filepath).tocsr()
        mat.setdiag(0)
        mat.eliminate_zeros()
        degrees = np.diff(mat.indptr)
        max_deg = int(degrees.max()) if len(degrees) > 0 else 0
        return int(math.ceil(max_deg / 2))
    except Exception:
        return 0

def make_table_per_dataset(records, base_filename="table_per_dataset", output_dirs=("results_v2/tables",)):
    """Generate summary table for per_dataset group (cadical195 on all datasets).
    SAT does not produce a solver lower bound (lb_solver is '-').
    ub_minus_lb is best_ub - lb, representing a weak bound gap rather than distance to optimality.
    """
    if isinstance(output_dirs, str):
        output_dirs = (output_dirs,)
    ds_recs = [
        r for r in records
        if "per_dataset" in r.get("groups", []) or (
            str(r.get("solver")).lower() == "cadical195"
            and r.get("relabel_seed") is None
            and r.get("config", {}).get("sb_rotation") is True
            and r.get("config", {}).get("sb_reflection") is True
            and r.get("config", {}).get("heuristic_ub") is True
        )
    ]
    if not ds_recs:
        return

    by_ds = {}
    for r in ds_recs:
        ds = r["dataset"]
        by_ds[ds] = r

    ds_sorted = sorted(by_ds.keys(), key=lambda d: (by_ds[d].get("N", 0), d))
    rows = []
    for ds in ds_sorted:
        r = by_ds[ds]
        st = r.get("status")
        ub = r.get("best_ub")
        solve_t = r.get("solve_time", 0.0)
        wall_t = r.get("wall_time", 0.0)
        n_iters = r.get("num_iterations", len(r.get("per_iter_times", [])))

        lb_deg = get_lb_degree(ds)
        best_lb = r.get("best_lb")
        lb_solver = best_lb if best_lb is not None else "-"
        lb = max(lb_deg, best_lb) if best_lb is not None else lb_deg
        if st == "TIMEOUT":
            ub_minus_lb_str = str(ub - lb) if (ub is not None and ub >= 0) else "-"
        elif st == "OPTIMAL":
            ub_minus_lb_str = "0"
        else:
            ub_minus_lb_str = "-"

        rows.append({
            "dataset": ds,
            "N": r.get("N", 0),
            "|E|": r.get("num_edges", 0),
            "status": st,
            "best_ub": ub if ub is not None else "-",
            "lb_degree": lb_deg,
            "lb_solver": lb_solver,
            "lb": lb,
            "ub_minus_lb": ub_minus_lb_str,
            "solve_time": f"{solve_t:.4f}",
            "wall_time": f"{wall_t:.4f}",
            "n_sat_iterations": n_iters,
        })

    df = pd.DataFrame(rows)
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{base_filename}.csv")
        df.to_csv(csv_path, index=False)
    print(f"Table saved: {base_filename}.csv in {output_dirs[0]}/")
    return df

def make_table_methods_family(records, base_filename="table_methods_family", output_dirs=("results_v2/tables",)):
    """
    Generate table_methods_family.csv:
    Compare cpsat vs sat_best across random vertex relabelings.
    Each row (dataset): median ratio (cpsat/sat), min, max, faster pairs / total pairs.
    Pairs: matching (dataset, relabel_seed).
    """
    if isinstance(output_dirs, str):
        output_dirs = (output_dirs,)

    pairs_map = {}
    for r in records:
        seed = r.get("relabel_seed")
        if seed is None:
            continue
        ds = r.get("dataset")
        sol = str(r.get("solver") or "").lower()
        m = str(r.get("method") or "").lower()
        if m == "cpsat" or sol in ("cp_sat", "cpsat"):
            pairs_map.setdefault((ds, seed), {})["cpsat"] = r
        elif m == "sat" or sol == "cadical195":
            cfg = r.get("config", {})
            if cfg.get("sb_rotation") and cfg.get("sb_reflection") and cfg.get("heuristic_ub"):
                pairs_map.setdefault((ds, seed), {})["sat"] = r

    if not pairs_map:
        return

    datasets = sorted(list(set(ds for (ds, seed), mmap in pairs_map.items() if "cpsat" in mmap and "sat" in mmap)))
    rows = []
    for ds in datasets:
        ratios = []
        faster_count = 0
        total_count = 0
        for (d, seed), mmap in pairs_map.items():
            if d == ds and "cpsat" in mmap and "sat" in mmap:
                rec_cp = mmap["cpsat"]
                rec_sat = mmap["sat"]
                if rec_cp.get("status") == "OPTIMAL" and rec_sat.get("status") == "OPTIMAL":
                    t_cp = rec_cp.get("solve_time", 0.0)
                    t_sat = rec_sat.get("solve_time", 0.0)
                    if t_sat > 0:
                        ratio = t_cp / t_sat
                        ratios.append(ratio)
                        total_count += 1
                        if t_cp < t_sat:
                            faster_count += 1
        if ratios:
            med = float(np.median(ratios))
            min_r = float(np.min(ratios))
            max_r = float(np.max(ratios))
            med_str = f"{med:.4f}"
            min_str = f"{min_r:.4f}"
            max_str = f"{max_r:.4f}"
            faster_str = f"{faster_count}/{total_count}"
        else:
            med_str = "-"
            min_str = "-"
            max_str = "-"
            faster_str = "0/0"

        rows.append({
            "dataset": ds,
            "median_cpsat_to_sat_ratio": med_str,
            "min": min_str,
            "max": max_str,
            "faster_pairs": faster_str,
        })

    df = pd.DataFrame(rows)
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{base_filename}.csv")
        df.to_csv(csv_path, index=False)
    print(f"Table saved: {base_filename}.csv in {output_dirs[0]}/")
    return df

def make_table_methods_large(records, base_filename="table_methods_large", output_dirs=("results_v2/tables",)):
    """
    Generate table_methods_large.csv:
    Compare cpsat vs sat_best on large datasets (N >= 80, original labels).
    Columns: dataset, N, |E|, sat_status, sat_best_ub, sat_time,
             cpsat_status, cpsat_best_ub, lb_degree, cpsat_solver_lb, lb_combined, cpsat_time.
    Note: SAT does not generate lower bounds; lb_degree is graph-theoretic.
          cpsat_solver_lb is raw BestObjectiveBound from CP-SAT (can be 1).
          lb_combined = max(lb_degree, cpsat_solver_lb).
    """
    if isinstance(output_dirs, str):
        output_dirs = (output_dirs,)

    target_datasets = [
        "494_bus_n80", "494_bus_n100", "662_bus_n100",
        "494_bus_n150", "662_bus_n150", "494_bus", "662_bus"
    ]

    sat_recs = {}
    cpsat_recs = {}
    meta = {}

    for r in records:
        if r.get("relabel_seed") is not None:
            continue
        ds = r.get("dataset")
        if ds not in target_datasets:
            continue
        meta[ds] = (r.get("N", 0), r.get("num_edges", 0))
        sol = str(r.get("solver") or "").lower()
        m = str(r.get("method") or "").lower()

        if m == "cpsat" or sol in ("cp_sat", "cpsat"):
            cpsat_recs[ds] = r
        elif m == "sat" or sol == "cadical195":
            cfg = r.get("config", {})
            if cfg.get("sb_rotation") and cfg.get("sb_reflection") and cfg.get("heuristic_ub"):
                sat_recs[ds] = r

    if not cpsat_recs and not sat_recs:
        return

    rows = []
    for ds in target_datasets:
        n, e = meta.get(ds, (0, 0))
        lb_deg = get_lb_degree(ds)

        rec_sat = sat_recs.get(ds, {})
        sat_st = rec_sat.get("status", "-")
        sat_ub = rec_sat.get("best_ub", "-")
        sat_time = f"{rec_sat.get('solve_time', 0.0):.4f}" if "solve_time" in rec_sat else "-"

        rec_cp = cpsat_recs.get(ds, {})
        cp_st = rec_cp.get("status", "-")
        cp_ub = rec_cp.get("best_ub", "-")
        raw_cp_lb = rec_cp.get("best_lb")
        cpsat_solver_lb = raw_cp_lb if raw_cp_lb is not None else "-"
        if raw_cp_lb is not None:
            lb_combined = max(lb_deg, raw_cp_lb)
        else:
            lb_combined = lb_deg if (cp_st != "-" or sat_st != "-") else "-"
        cp_time = f"{rec_cp.get('solve_time', 0.0):.4f}" if "solve_time" in rec_cp else "-"

        rows.append({
            "dataset": ds,
            "N": n,
            "|E|": e,
            "sat_status": sat_st,
            "sat_best_ub": sat_ub,
            "sat_time": sat_time,
            "cpsat_status": cp_st,
            "cpsat_best_ub": cp_ub,
            "lb_degree": lb_deg,
            "cpsat_solver_lb": cpsat_solver_lb,
            "lb_combined": lb_combined,
            "cpsat_time": cp_time,
        })

    df = pd.DataFrame(rows)
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{base_filename}.csv")
        df.to_csv(csv_path, index=False)
    print(f"Table saved: {base_filename}.csv in {output_dirs[0]}/")
    return df

def make_table_methods_large_seeds(records, base_filename="table_methods_large_seeds", output_dirs=("results_v2/tables",)):
    """
    Generate table_methods_large_seeds.csv:
    For datasets 494_bus_n80, 494_bus_n100, 662_bus_n80, 662_bus_n100:
    Rows: each dataset with original label (relabel_seed=None) and seeds 0, 1.
    Columns: dataset, seed, sat_status, sat_best_ub, sat_time, cpsat_status, cpsat_best_ub, cpsat_time.
    """
    if isinstance(output_dirs, str):
        output_dirs = (output_dirs,)

    target_datasets = ["494_bus_n80", "662_bus_n80", "494_bus_n100", "662_bus_n100"]
    seeds = [None, 0, 1]

    sat_map = {}
    cpsat_map = {}

    for r in records:
        ds = r.get("dataset")
        if ds not in target_datasets:
            continue
        seed = r.get("relabel_seed")
        if seed not in seeds:
            continue

        sol = str(r.get("solver") or "").lower()
        m = str(r.get("method") or "").lower()

        if m == "cpsat" or sol in ("cp_sat", "cpsat"):
            cpsat_map[(ds, seed)] = r
        elif m == "sat" or sol == "cadical195":
            cfg = r.get("config", {})
            if cfg.get("sb_rotation") and cfg.get("sb_reflection") and cfg.get("heuristic_ub"):
                sat_map[(ds, seed)] = r

    if not sat_map and not cpsat_map:
        print("[WARNING] Skipping table_methods_large_seeds: no records found.")
        return None

    rows = []
    for ds in target_datasets:
        for seed in seeds:
            rec_sat = sat_map.get((ds, seed), {})
            sat_st = rec_sat.get("status", "-")
            sat_ub = rec_sat.get("best_ub", "-")
            sat_time = f"{rec_sat.get('solve_time', 0.0):.4f}" if "solve_time" in rec_sat else "-"

            rec_cp = cpsat_map.get((ds, seed), {})
            cp_st = rec_cp.get("status", "-")
            cp_ub = rec_cp.get("best_ub", "-")
            cp_time = f"{rec_cp.get('solve_time', 0.0):.4f}" if "solve_time" in rec_cp else "-"

            seed_str = str(seed) if seed is not None else "original"
            rows.append({
                "dataset": ds,
                "seed": seed_str,
                "sat_status": sat_st,
                "sat_best_ub": sat_ub,
                "sat_time": sat_time,
                "cpsat_status": cp_st,
                "cpsat_best_ub": cp_ub,
                "cpsat_time": cp_time,
            })

    df = pd.DataFrame(rows)
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{base_filename}.csv")
        df.to_csv(csv_path, index=False)
    print(f"Table saved: {base_filename}.csv in {output_dirs[0]}/")
    return df

def main():
    print("=" * 70)
    print("ANALYSIS: GENERATING LATEX AND CSV TABLES")
    print("=" * 70)

    import argparse
    parser = argparse.ArgumentParser(description="Generate LaTeX and CSV tables from experiment records")
    parser.add_argument("input_file", nargs="?", default=None, help="path to runs.jsonl or results directory")
    parser.add_argument("output_dir_pos", nargs="?", default=None, help="optional positional output directory")
    parser.add_argument("--output-dir", type=str, default=None, help="directory to store tables (defaults to <run_dir>/tables)")
    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir or args.output_dir_pos
    if not output_dir:
        if input_file and (os.path.isfile(input_file) or os.path.isdir(input_file)):
            run_dir = os.path.dirname(input_file) if os.path.isfile(input_file) else input_file
            run_dir = run_dir or "results_v2"
        else:
            run_dir = "results_v2"
        output_dir = os.path.join(run_dir, "tables")

    records = load_records(input_file)
    if not records:
        print("No records found to analyze. Please run experiments first.")
        return

    # Step 1: Cross-validation check
    try:
        cross_validate_optima(records, output_dir=output_dir)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Make tables
    output_dirs = (output_dir,)
    make_table_ablation(records, group_name="ablation_small", base_filename="table_ablation_small", title="Ablation Study: Small Graphs", output_dirs=output_dirs)
    make_table_ablation(records, group_name="ablation_medium", base_filename="table_ablation_medium", title="Ablation Study: Medium Graph (662_bus_n60)", output_dirs=output_dirs)
    make_table_ablation_family(records, base_filename="table_ablation_family", output_dirs=output_dirs)
    make_table_ablation_phases(records, base_filename="table_ablation_phases", output_dirs=output_dirs)
    make_table_solvers(records, group_name="solvers_small", base_filename="table_solvers_small", title="SAT Solver Performance Comparison (Small Graphs)", output_dirs=output_dirs)
    make_table_solvers(records, group_name="solvers_medium", base_filename="table_solvers_medium", title="SAT Solver Performance Comparison (Medium Graphs)", output_dirs=output_dirs)
    make_table_methods(records, output_dirs=output_dirs)
    make_table_per_dataset(records, output_dirs=output_dirs)
    make_table_methods_family(records, output_dirs=output_dirs)
    make_table_methods_large(records, output_dirs=output_dirs)
    make_table_methods_large_seeds(records, output_dirs=output_dirs)

    print(f"\nAll tables generated successfully in {output_dir}/!")

if __name__ == "__main__":
    main()
