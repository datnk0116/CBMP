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

DATA_CSV_COLUMNS = [
    "dataset", "column", "status",
    "par2", "solve_time", "wall_time", "encode_time", "best_ub",
    "n_total", "n_algorithmic", "n_optimal", "n_timeout", "n_interrupted",
    "n_feasible", "n_infeasible", "n_error", "n_size_limit",
    "n_unavailable", "n_license_error", "n_unknown",
    "termination_reason",
]

def validate_best_ub(val):
    """
    Validates that best_ub is a genuine finite non-negative integer.
    Accepts:
      - Python int (non-negative)
      - NumPy integer types (non-negative)
      - Python float or NumPy floating representing an exact integer (e.g. 3.0, np.float64(3.0))
      - Strings representing an exact non-negative integer (e.g. "3", "3.0")
    Rejects:
      - Fractional floats (e.g. 3.7, 0.9) - NEVER truncates or rounds
      - Booleans (bool is a subclass of int in Python)
      - Negative values
      - None, NaN, +inf, -inf
      - Malformed strings, lists, dicts
    Returns:
        int if valid, else None.
    """
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, np.integer)):
        return int(val) if val >= 0 else None
    if isinstance(val, (float, np.floating)):
        try:
            if math.isfinite(val) and val >= 0.0 and float(val).is_integer():
                return int(val)
            return None
        except (ValueError, TypeError, OverflowError):
            return None
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            i = int(s)
            return i if i >= 0 else None
        except ValueError:
            try:
                f = float(s)
                if math.isfinite(f) and f >= 0.0 and f.is_integer():
                    return int(f)
                return None
            except (ValueError, TypeError, OverflowError):
                return None
    return None

def validate_solve_time(val):
    """
    Validates that a timing metric is a genuine finite, non-negative float.
    Rejects booleans (since bool is a subclass of int), None, negatives, NaN, inf, -inf,
    and unparseable/invalid strings.
    Returns:
        float if valid, else None.
    """
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
        if math.isfinite(f) and f >= 0.0:
            return f
        return None
    except (ValueError, TypeError):
        return None

def extract_repetition_solve_time(r):
    """
    Extracts a validated solve time from a repetition record.
    Checks 'solve_time' first; if invalid or missing, falls back to 'wall_time'.
    Returns validated non-negative finite float, or None if neither is valid.
    """
    if not isinstance(r, dict):
        return None
    t = validate_solve_time(r.get("solve_time"))
    if t is not None:
        return t
    return validate_solve_time(r.get("wall_time"))

def extract_time_limit(record, default_timeout=600.0):
    """Safely extracts a validated positive numeric timeout from a record."""
    if not isinstance(record, dict):
        return float(default_timeout)
    val = record.get("time_limit")
    if val is None:
        val = record.get("timeout")
    if val is None and isinstance(record.get("config"), dict):
        val = record.get("config", {}).get("time_limit")
    if val is not None and not isinstance(val, bool):
        try:
            f_val = float(val)
            if f_val > 0 and math.isfinite(f_val):
                return f_val
        except (ValueError, TypeError):
            pass
    return float(default_timeout)

def is_valid_record_dict(val):
    """
    Checks if val is a valid non-empty dictionary representing a result record.
    Rejects None, NaN, pd.NA, scalars, empty dicts, booleans, and non-dict objects.
    """
    if val is None or isinstance(val, bool) or val is pd.NA:
        return False
    if isinstance(val, float) and math.isnan(val):
        return False
    if not isinstance(val, dict):
        return False
    return len(val) > 0

def normalize_status(val):
    """
    Normalizes a status string or record dict to uppercase canonical status taxonomy.
    Returns canonical status string, or None if missing/invalid/empty.
    Preserves taxonomy:
      OPTIMAL, TIMEOUT, INTERRUPTED, FEASIBLE, SIZE_LIMIT, UNAVAILABLE,
      LICENSE_ERROR, INFEASIBLE, NUMERIC_ERROR, ERROR, PARTIAL, UNKNOWN.
    """
    if val is None or isinstance(val, bool) or val is pd.NA:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, dict):
        val = val.get("status")
        if val is None or isinstance(val, bool) or val is pd.NA:
            return None
        if isinstance(val, float) and math.isnan(val):
            return None

    s = str(val).strip().upper()
    if not s or s in ("NONE", "NAN", "<NA>", "NULL", "-"):
        return None

    if s == "OPTIMAL":
        return "OPTIMAL"
    elif s == "TIMEOUT":
        return "TIMEOUT"
    elif s == "INTERRUPTED":
        return "INTERRUPTED"
    elif s == "FEASIBLE":
        return "FEASIBLE"
    elif s == "SIZE_LIMIT":
        return "SIZE_LIMIT"
    elif s == "UNAVAILABLE":
        return "UNAVAILABLE"
    elif s == "LICENSE_ERROR":
        return "LICENSE_ERROR"
    elif s in ("INFEASIBLE", "UNSAT"):
        return "INFEASIBLE"
    elif s in ("ERROR",):
        return "ERROR"
    elif s in ("NUMERIC_ERROR",):
        return "NUMERIC_ERROR"
    elif s == "PARTIAL":
        return "PARTIAL"
    elif s in ("UNKNOWN", "UNK"):
        return "UNKNOWN"
    else:
        return "UNKNOWN"

def is_optimal_status(val):
    """Returns True iff val (record dict or status string) represents an OPTIMAL status."""
    return normalize_status(val) == "OPTIMAL"

def categorize_status(raw_status):
    """Maps any status string to an explicit canonical status category."""
    st = normalize_status(raw_status)
    if st is None or st == "UNKNOWN":
        return "UNKNOWN"
    if st in ("INFEASIBLE", "UNSAT"):
        return "INFEASIBLE"
    elif st in ("ERROR", "NUMERIC_ERROR"):
        return "ERROR"
    return st

def get_baseline_eligible_instances(df_raw, baseline_col):
    """
    Identifies baseline eligible instances from df_raw for baseline_col.
    Criteria for eligibility:
      - Valid non-empty record dict
      - Normalized status is strictly OPTIMAL
      - Valid finite positive timing (via extract_repetition_solve_time)
      - Timing >= 0.1 seconds
    Returns:
      dict mapping dataset_name -> validated baseline timing (float)
    """
    baseline_times = {}
    if not baseline_col or df_raw is None or not hasattr(df_raw, "iterrows") or baseline_col not in df_raw.columns:
        return baseline_times

    for ds, row in df_raw.iterrows():
        rec = row.get(baseline_col)
        if not is_valid_record_dict(rec):
            continue
        if not is_optimal_status(rec):
            continue
        t_base = extract_repetition_solve_time(rec)
        if t_base is not None and math.isfinite(t_base) and t_base >= 0.1:
            baseline_times[ds] = float(t_base)

    return baseline_times

def compute_repetition_par2(reps, default_timeout=600.0):
    """
    Computes exact arithmetic PAR2 score across individual repetition records according to canonical rules:
    - OPTIMAL: validated solve_time (falling back to wall_time), or 2 * time_limit if timing is missing/invalid.
    - TIMEOUT, INTERRUPTED, FEASIBLE, INFEASIBLE, ERROR: penalized with 2 * time_limit.
    - SIZE_LIMIT, UNAVAILABLE, LICENSE_ERROR, UNKNOWN: non-algorithmic runs, excluded from numerator and denominator.
    Returns:
        float mean PAR2 if at least one algorithmic repetition exists, else None.
    """
    if not reps:
        return None
    rep_par2_vals = []
    for r in reps:
        if not isinstance(r, dict):
            continue
        cat = r.get("_cat") or categorize_status(r.get("status"))
        if r.get("validation_error"):
            cat = "ERROR"
        if cat == "OPTIMAL":
            t = extract_repetition_solve_time(r)
            if t is not None:
                rep_par2_vals.append(t)
            else:
                t_lim = extract_time_limit(r, default_timeout=default_timeout)
                rep_par2_vals.append(2.0 * t_lim)
        elif cat in ("TIMEOUT", "INTERRUPTED", "FEASIBLE", "INFEASIBLE", "ERROR"):
            t_lim = extract_time_limit(r, default_timeout=default_timeout)
            rep_par2_vals.append(2.0 * t_lim)
        else:
            # Excluded: SIZE_LIMIT, UNAVAILABLE, LICENSE_ERROR, UNKNOWN
            pass
    if rep_par2_vals:
        return float(np.mean(rep_par2_vals))
    return None

def aggregate_repetition_group(reps, default_timeout=600.0):
    """
    Aggregates a group of repetition runs for the same configuration/instance.
    Populates:
    - Canonical aggregated status (OPTIMAL, PARTIAL, FEASIBLE, TIMEOUT, etc.)
    - Exact 12 population counters (n_total, n_algorithmic, n_optimal, etc.)
    - Exact PAR2 via compute_repetition_par2 (excluding non-algorithmic runs)
    - Median solve_time, wall_time, encode_time
    - Attached list of reps
    """
    if not reps:
        return None

    cleaned_reps = []
    for r in reps:
        if not isinstance(r, dict):
            continue
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
        return None

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

    def _safe_solve_key(r):
        t = extract_repetition_solve_time(r)
        return t if t is not None else float("inf")

    def _safe_ub_key(r):
        ub = validate_best_ub(r.get("best_ub"))
        return ub if ub is not None else float("inf")

    # Select representative record to seed the aggregated dictionary
    if agg_status in ("OPTIMAL", "PARTIAL"):
        rep_pool = [r for r in reps if r["_cat"] == "OPTIMAL"]
        reps_sorted = sorted(rep_pool, key=_safe_solve_key)
        rep_chosen = reps_sorted[len(reps_sorted) // 2]
    elif agg_status == "FEASIBLE":
        rep_pool = [r for r in reps if r["_cat"] == "FEASIBLE"]
        reps_sorted = sorted(rep_pool, key=lambda x: (_safe_ub_key(x), _safe_solve_key(x)))
        rep_chosen = reps_sorted[0]
    else:
        matching = [r for r in reps if r["_cat"] == agg_status]
        rep_pool = matching if matching else reps
        reps_sorted = sorted(rep_pool, key=_safe_solve_key)
        rep_chosen = reps_sorted[len(reps_sorted) // 2]

    solve_times = [extract_repetition_solve_time(r) for r in rep_pool if extract_repetition_solve_time(r) is not None]
    wall_times = [validate_solve_time(r.get("wall_time")) for r in rep_pool if validate_solve_time(r.get("wall_time")) is not None]
    has_fallback = any(validate_solve_time(r.get("solve_time")) is None and validate_solve_time(r.get("wall_time")) is not None for r in rep_pool)
    has_genuine = any(validate_solve_time(r.get("solve_time")) is not None for r in rep_pool)

    encode_times = [validate_solve_time(r.get("encode_time")) for r in reps if validate_solve_time(r.get("encode_time")) is not None]

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
        agg["timing_fallback"] = has_fallback
        if has_fallback and has_genuine:
            agg["solve_time_source"] = "mixed"
        elif has_fallback:
            agg["solve_time_source"] = "wall_time"
        else:
            agg["solve_time_source"] = "solve_time"
    else:
        t_chosen = extract_repetition_solve_time(rep_chosen)
        agg["solve_time"] = t_chosen
        agg["timing_fallback"] = False
        agg["solve_time_source"] = None

    if wall_times:
        agg["wall_time"] = float(np.median(wall_times))
    else:
        agg["wall_time"] = validate_solve_time(rep_chosen.get("wall_time"))

    if encode_times:
        agg["encode_time"] = float(np.median(encode_times))
    else:
        agg["encode_time"] = validate_solve_time(rep_chosen.get("encode_time"))

    par2_val = compute_repetition_par2(reps, default_timeout=default_timeout)
    if par2_val is not None:
        agg["par2"] = par2_val
    else:
        agg.pop("par2", None)

    if agg_status in ("ERROR", "NUMERIC_ERROR", "LICENSE_ERROR", "UNAVAILABLE", "SIZE_LIMIT", "INFEASIBLE", "UNKNOWN") or agg.get("validation_error"):
        agg["best_ub"] = -1
        agg["assignment"] = None
    elif agg_status == "FEASIBLE":
        ub_c = validate_best_ub(rep_chosen.get("best_ub"))
        agg["best_ub"] = ub_c if ub_c is not None else -1
    elif agg_status in ("OPTIMAL", "PARTIAL"):
        ub_c = validate_best_ub(rep_chosen.get("best_ub"))
        if ub_c is not None:
            agg["best_ub"] = ub_c

    # Union of groups across reps
    all_groups = set()
    for r in reps:
        for g in r.get("groups", []):
            all_groups.add(g)
        if "_group" in r:
            all_groups.add(r["_group"])
    agg["groups"] = sorted(list(all_groups))
    agg["reps"] = reps

    return agg

def aggregate_repetitions(records):
    """
    Group records by unique configuration (dataset, method/solver, config_signature),
    and aggregate across repetitions using aggregate_repetition_group.
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
        agg = aggregate_repetition_group(reps)
        if agg is not None:
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
        if is_optimal_status(r):
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

def format_cell_value(status, solve_time=None, best_ub=None, n_optimal=None, n_total=None, n_algorithmic=None):
    """
    Format single cell according to required status representations:
      - OPTIMAL: solve_time in seconds (e.g. "0.0123"), or "-" if timing is missing/invalid
      - PARTIAL: "k/n time*" (e.g. "3/5 0.1234*"), or "k/n -*" if timing is missing/invalid
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
        if not status:
            return "-"
        rec = status
        status = rec.get("status")
        solve_time = extract_repetition_solve_time(rec)
        best_ub = rec.get("best_ub")
        n_optimal = rec.get("n_optimal")
        n_total = rec.get("n_total")
        n_algorithmic = rec.get("n_algorithmic")

    st = normalize_status(status)
    if st is None:
        return "-"

    t_val = validate_solve_time(solve_time)
    ub_val = validate_best_ub(best_ub)

    if st == "OPTIMAL":
        return f"{t_val:.4f}" if t_val is not None else "-"
    elif st == "PARTIAL":
        denom = n_total if (n_total is not None and n_total > 0) else n_algorithmic
        k_n = f"{n_optimal}/{denom}" if (n_optimal is not None and denom is not None) else "PARTIAL"
        t_str = f"{t_val:.4f}*" if t_val is not None else "-*"
        return f"{k_n} {t_str}"
    elif st == "FEASIBLE":
        return f"FEAS({ub_val})" if ub_val is not None and ub_val >= 0 else "FEAS"
    elif st == "TIMEOUT":
        return f"TO({ub_val})" if ub_val is not None and ub_val >= 0 else "TO"
    elif st == "INTERRUPTED":
        return f"INT({ub_val})" if ub_val is not None and ub_val >= 0 else "INT"
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

    baseline_times = get_baseline_eligible_instances(df_raw, baseline_col)

    for col in column_names:
        solved = 0
        sl_count = 0
        par2_vals = []
        ratios = []

        for ds, row in df_raw.iterrows():
            rec = row.get(col)
            if not is_valid_record_dict(rec):
                continue
            st = normalize_status(rec)
            rec_timeout = extract_time_limit(rec, default_timeout=timeout)

            if st == "OPTIMAL":
                solved += 1
                par2_val = validate_solve_time(rec.get("par2"))
                if par2_val is None and "reps" in rec and rec["reps"]:
                    par2_val = compute_repetition_par2(rec["reps"], default_timeout=rec_timeout)
                if par2_val is None:
                    is_multi_rep = (
                        (rec.get("n_total") is not None and rec.get("n_total") > 1)
                        or (rec.get("n_algorithmic") is not None and rec.get("n_algorithmic") > 1)
                        or bool(rec.get("mixed"))
                        or bool(rec.get("is_mixed"))
                    )
                    if not is_multi_rep:
                        t_single = extract_repetition_solve_time(rec)
                        if t_single is not None:
                            par2_val = t_single
                        else:
                            par2_val = 2.0 * float(rec_timeout)
                    else:
                        # Aggregated record with multiple repetitions but no reps list and invalid par2:
                        # Do NOT guess from median solve_time.
                        pass
                if par2_val is not None:
                    par2_vals.append(par2_val)
                if ds in baseline_times:
                    t_cand = extract_repetition_solve_time(rec)
                    if t_cand is not None and t_cand > 0:
                        ratio = baseline_times[ds] / t_cand
                        if math.isfinite(ratio) and ratio > 0:
                            ratios.append(ratio)
            elif st == "PARTIAL":
                p = validate_solve_time(rec.get("par2"))
                if p is None and "reps" in rec and rec["reps"]:
                    p = compute_repetition_par2(rec["reps"], default_timeout=rec_timeout)
                if p is not None:
                    par2_vals.append(p)
                else:
                    # Legacy record without precomputed par2 and without individual repetition data:
                    # Do not reconstruct exact PAR2 from median timing.
                    pass
            elif st == "SIZE_LIMIT":
                sl_count += 1
                # SIZE_LIMIT is reported separately in SL and not penalized in PAR2 as timeout
            elif st in ("UNAVAILABLE", "LICENSE_ERROR", "UNKNOWN") or st is None:
                # Solver environment unavailability, missing license, or unknown status excluded from algorithmic PAR2
                pass
            else:
                par2_val = validate_solve_time(rec.get("par2"))
                if par2_val is None and "reps" in rec and rec["reps"]:
                    par2_val = compute_repetition_par2(rec["reps"], default_timeout=rec_timeout)
                if par2_val is None:
                    par2_val = 2.0 * float(rec_timeout)
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

def export_table(df_display, summary_rows, title, base_filename, output_dirs=("results_v2/tables",), note=None, df_raw=None):
    """Saves formatted table as both CSV and LaTeX, and optionally exports machine-readable raw metrics."""
    if isinstance(output_dirs, str):
        output_dirs = (output_dirs,)
    full_df = df_display.astype(object).copy()
    for row_name, row_data in summary_rows:
        full_df.loc[row_name] = row_data

    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{base_filename}.csv")
        clean_display_df = full_df.copy()
        for c in clean_display_df.columns:
            clean_display_df[c] = [
                "-" if (v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v))) else v
                for v in clean_display_df[c]
            ]
        clean_display_df.to_csv(csv_path, index=True)

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
                row_vals = [
                    "-" if (v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)))
                    else str(v).replace("_", "\\_")
                    for v in row
                ]
                f.write(f"{idx_escaped} & " + " & ".join(row_vals) + " \\\\\n")

            f.write("\\midrule\n")
            for rname, rdata in summary_rows:
                rname_escaped = f"\\textbf{{{rname.replace('_', ' ')}}}"
                vals = []
                for c in full_df.columns:
                    v = rdata.get(c, "-") if isinstance(rdata, dict) else "-"
                    if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
                        vals.append("-")
                    else:
                        vals.append(str(v).replace("_", "\\_"))
                f.write(f"{rname_escaped} & " + " & ".join(vals) + " \\\\\n")

            f.write("\\bottomrule\n")
            if note:
                total_cols = len(full_df.columns) + 1
                f.write(f"\\multicolumn{{{total_cols}}}{{l}}{{\\footnotesize {note}}} \\\\\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        # Dedicated machine-readable aggregation export preserving exact population counters
        if df_raw is not None:
            raw_rows = []
            if hasattr(df_raw, "iterrows") and hasattr(df_raw, "columns"):
                for ds, row in df_raw.iterrows():
                    for col in df_raw.columns:
                        cell = row.get(col)
                        if is_valid_record_dict(cell):
                            raw_rows.append({
                                "dataset": ds,
                                "column": col,
                                "status": cell.get("status"),
                                "par2": cell.get("par2"),
                                "solve_time": cell.get("solve_time"),
                                "wall_time": cell.get("wall_time"),
                                "encode_time": cell.get("encode_time"),
                                "best_ub": cell.get("best_ub"),
                                "n_total": cell.get("n_total"),
                                "n_algorithmic": cell.get("n_algorithmic"),
                                "n_optimal": cell.get("n_optimal"),
                                "n_timeout": cell.get("n_timeout"),
                                "n_interrupted": cell.get("n_interrupted"),
                                "n_feasible": cell.get("n_feasible"),
                                "n_infeasible": cell.get("n_infeasible"),
                                "n_error": cell.get("n_error"),
                                "n_size_limit": cell.get("n_size_limit"),
                                "n_unavailable": cell.get("n_unavailable"),
                                "n_license_error": cell.get("n_license_error"),
                                "n_unknown": cell.get("n_unknown"),
                                "termination_reason": cell.get("termination_reason"),
                            })
            df_metrics = pd.DataFrame(raw_rows, columns=DATA_CSV_COLUMNS)
            metrics_csv_path = os.path.join(out_dir, f"{base_filename}_data.csv")
            df_metrics.to_csv(metrics_csv_path, index=False)

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
            if is_valid_record_dict(rec):
                col_vals.append(format_cell_value(rec))
            else:
                col_vals.append("-")
        df_display[col] = col_vals

    solved_cnt, sl_cnt, par2_val, spdups = compute_summary_stats(df_raw, columns, baseline_col="none")
    n_used = len(get_baseline_eligible_instances(df_raw, "none"))
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
    export_table(df_display, summary, title, base_filename, output_dirs=output_dirs, note=note, df_raw=df_raw)

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
            r0 = rec_list[0]
            if r0.get("n_total") is not None:
                data_by_ds.setdefault(ds, {})[sol] = r0
            else:
                agg = aggregate_repetition_group(rec_list)
                data_by_ds.setdefault(ds, {})[sol] = agg if agg is not None else r0
        else:
            agg = aggregate_repetition_group(rec_list)
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
            if is_valid_record_dict(rec):
                col_vals.append(format_cell_value(rec))
            else:
                col_vals.append("-")
        df_display[sol] = col_vals

    solved_cnt, sl_cnt, par2_val, spdups = compute_summary_stats(df_raw, solvers, baseline_col="cadical195")
    n_used = len(get_baseline_eligible_instances(df_raw, "cadical195"))
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

    export_table(df_display, summary, title, base_filename, output_dirs=output_dirs, note=note, df_raw=df_raw)

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
            if is_valid_record_dict(rec):
                col_vals.append(format_cell_value(rec))
            else:
                col_vals.append("-")
        df_display[m] = col_vals

    solved_cnt, sl_cnt, par2_val, spdups = compute_summary_stats(df_raw, methods, baseline_col="sat_best")
    n_used = len(get_baseline_eligible_instances(df_raw, "sat_best"))
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
    export_table(df_display, summary, "Comparison of Methods: SAT vs CP-SAT vs CPLEX", "table_methods", output_dirs=output_dirs, note=note, df_raw=df_raw)

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
            and is_optimal_status(vmap[var_target])
            and is_optimal_status(vmap[var_ref])
            and (extract_repetition_solve_time(vmap[var_ref]) or 0.0) > 0
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
                if is_optimal_status(rec_t) and is_optimal_status(rec_r):
                    t_target = extract_repetition_solve_time(rec_t) or 0.0
                    t_ref = extract_repetition_solve_time(rec_r) or 0.0
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
                    and is_optimal_status(vmap[var_target])
                    and is_optimal_status(vmap[var_ref])
                    and (extract_repetition_solve_time(vmap[var_ref]) or 0.0) > 0
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
                        if is_optimal_status(rec_t) and is_optimal_status(rec_r):
                            t_target = extract_repetition_solve_time(rec_t) or 0.0
                            t_ref = extract_repetition_solve_time(rec_r) or 0.0
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
            count = sum(1 for r in family_recs if r.get("N") == n and get_var(r) == var and is_optimal_status(r))
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
                if r.get("N") == n and get_var(r) == var and is_optimal_status(r):
                    per_iter = r.get("per_iter_times", [])
                    st = extract_repetition_solve_time(r) or 0.0
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
                if is_optimal_status(rec_cp) and is_optimal_status(rec_sat):
                    t_cp = extract_repetition_solve_time(rec_cp) or 0.0
                    t_sat = extract_repetition_solve_time(rec_sat) or 0.0
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
