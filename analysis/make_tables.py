import os
import sys
import glob
import json
import math
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger("make_tables")

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
      - None, NaN, +inf, -inf, pd.NA, pd.NaT
      - Malformed strings, lists, dicts
    Returns:
        int if valid, else None.
    """
    if val is None or isinstance(val, bool) or val is pd.NA or val is pd.NaT:
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

def validate_lower_bound(val):
    """
    Validates that a lower bound is a genuine finite non-negative integer.
    Bandwidth bounds (e.g. ceil(Delta / 2) or solver lower bounds) are integers >= 0.
    Rejects:
      - Fractional floats (e.g. 3.7) - NEVER truncates or rounds
      - Booleans (bool is a subclass of int in Python)
      - Negative values
      - None, NaN, +inf, -inf, pd.NA, pd.NaT
      - Malformed strings, lists, dicts
    Returns:
        int if valid, else None.
    """
    return validate_best_ub(val)

def strict_int(val, min_val=None, max_val=None):
    """
    Strictly validates and parses an integer value.
    Rejects:
      - Booleans (True, False)
      - Fractional floats (e.g. 2.7, 4.8)
      - Non-numeric or malformed strings
      - Values outside [min_val, max_val]
    Returns:
      int if valid, else None.
    """
    if val is None or isinstance(val, bool) or val is pd.NA or val is pd.NaT:
        return None
    if isinstance(val, (int, np.integer)):
        i_val = int(val)
    elif isinstance(val, (float, np.floating)):
        if not math.isfinite(val) or not float(val).is_integer():
            return None
        i_val = int(val)
    elif isinstance(val, str):
        s = val.strip()
        try:
            i_val = int(s)
        except ValueError:
            try:
                f = float(s)
                if not math.isfinite(f) or not f.is_integer():
                    return None
                i_val = int(f)
            except (ValueError, TypeError):
                return None
    else:
        return None

    if min_val is not None and i_val < min_val:
        return None
    if max_val is not None and i_val > max_val:
        return None
    return i_val

def strict_positive_int(val):
    """Strictly validates a positive integer (>= 1)."""
    return strict_int(val, min_val=1)

def strict_bool(val, default=None):
    """
    Strictly validates and parses a boolean value.
    Accepts:
      - Python bool (True, False)
      - NumPy bool (np.bool_)
      - Exact string representations: 'true', 'false', '1', '0', 'yes', 'no'
    Rejects:
      - Integers (1, 0, 2) when passed as numeric types (to prevent int/bool confusion)
      - Floats, NaN, inf
      - Malformed or arbitrary strings
    Returns:
      bool if valid, default if val is None, else None (invalid).
    """
    if val is None:
        return default
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, float, np.number)):
        return None
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "t", "yes", "1"):
            return True
        elif s in ("false", "f", "no", "0"):
            return False
        return None
    return None

EXACT_SOLVER_ALIASES = {
    # SAT solvers
    "cadical": "cadical195",
    "cadical195": "cadical195",
    "cadical_195": "cadical195",
    "kissat": "kissat",
    "glucose": "glucose4",
    "glucose4": "glucose4",
    "glucose_4": "glucose4",
    "maple": "maplechrono",
    "maplechrono": "maplechrono",
    "minisat": "minisat22",
    "minisat22": "minisat22",
    "lingeling": "lingeling",
    # CP-SAT
    "cp_sat": "cp_sat",
    "cpsat": "cp_sat",
    "ortools": "cp_sat",
    "ortools_cpsat": "cp_sat",
    # CPLEX MILP
    "cplex": "cplex",
    "cplex_milp": "cplex",
    "ibm_cplex": "cplex",
    # CPLEX CP Optimizer
    "cplex_cp": "cplex_cp",
    "cpoptimizer": "cplex_cp",
    "docplex_cp": "cplex_cp",
    # Gurobi
    "gurobi": "gurobi",
    "grb": "gurobi",
}

EXACT_METHOD_ALIASES = {
    "sat": "sat",
    "sat_solver": "sat",
    "cpsat": "cpsat",
    "cp_sat": "cpsat",
    "cp": "cpsat",
    "ortools": "cpsat",
    "ortools_cpsat": "cpsat",
    "cplex": "cplex",
    "cplex_milp": "cplex",
    "ibm_cplex": "cplex",
    "cplex_cp": "cplex_cp",
    "cpoptimizer": "cplex_cp",
    "docplex_cp": "cplex_cp",
    "gurobi": "gurobi",
    "grb": "gurobi",
}

def normalize_solver_alias(raw_solver):
    """
    Normalizes known solver aliases to their canonical engine names using exact dictionary lookup.
    Preserves unknown, custom, or mock solver strings (e.g. 's1', 'sol_a') without corrupting them.
    """
    if raw_solver is None:
        return ""
    s = str(raw_solver).strip()
    if not s:
        return ""
    s_lower = s.lower()
    return EXACT_SOLVER_ALIASES.get(s_lower, s)


def normalize_method_alias(raw_method):
    """
    Normalizes known method aliases to their canonical names using exact dictionary lookup.
    Preserves unknown method identifiers as stripped lowercase strings.
    """
    if raw_method is None:
        return ""
    s = str(raw_method).strip()
    if not s:
        return ""
    s_lower = s.lower()
    return EXACT_METHOD_ALIASES.get(s_lower, s_lower)


class DuplicateConflictError(ValueError):
    """Raised when duplicate runs have divergent OPTIMAL best_ub."""
    pass


def get_repetition_identity(record):
    """
    Extracts validated repetition integer (>= 1) from record.
    Checks 'rep' and 'repetition', ensuring consistency and strict positive integer.
    Rejects booleans, floats, non-numeric strings, negatives/zeros.
    Returns int >= 1, or None if invalid.
    """
    if not isinstance(record, dict):
        return None
    raw_rep = record.get("rep")
    raw_repetition = record.get("repetition")
    if raw_rep is not None and raw_repetition is not None:
        r1 = strict_positive_int(raw_rep)
        r2 = strict_positive_int(raw_repetition)
        if r1 is None or r2 is None or r1 != r2:
            return None
        return r1
    val = raw_rep if raw_rep is not None else raw_repetition
    if val is None:
        return 1
    return strict_positive_int(val)

def is_valid_assignment(assignment, n=None, allow_none=False):
    """
    Validates that assignment represents a valid vertex permutation bijection.

    Requirements:
    - assignment must be a list or tuple. If assignment is None:
        - returns True only if allow_none is True
        - returns False if allow_none is False (default)
    - Each element must be a strict non-negative integer (reject bool, float, str, dict, etc.)
    - If n is provided:
        - n must be a strict positive integer (n >= 1). If n is invalid, returns False.
        - len(assignment) must be exactly n.
        - The set of elements must be exactly {0, 1, ..., n-1} (a permutation of 0..n-1).
    - If n is None:
        - len(assignment) must be >= 1.
        - Elements must be unique and form {0, 1, ..., len(assignment)-1}.
    - Handles unhashable objects safely without raising TypeError.
    """
    if assignment is None:
        return bool(allow_none)
    if not isinstance(assignment, (list, tuple)):
        return False
    if isinstance(assignment, (str, bytes, dict)):
        return False

    if n is not None:
        n_int = strict_int(n, min_val=1)
        if n_int is None:
            return False
        if len(assignment) != n_int:
            return False
        target_n = n_int
    else:
        if len(assignment) == 0:
            return False
        target_n = len(assignment)

    seen = bytearray(target_n)
    for x in assignment:
        if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
            return False
        int_x = int(x)
        if int_x < 0 or int_x >= target_n:
            return False
        if seen[int_x]:
            return False
        seen[int_x] = 1

    return True

def resolve_field_with_conflict_check(record, cfg, field_name, validator, default=None):
    """
    Extracts field from record and config, validates both.
    If both are present and differ, detects conflict and returns (None, True).
    Returns: (resolved_value, has_conflict)
    """
    rec_val = record.get(field_name) if field_name in record else None
    cfg_val = cfg.get(field_name) if field_name in cfg else None

    has_rec = (field_name in record and rec_val is not None)
    has_cfg = (field_name in cfg and cfg_val is not None)

    if has_rec and has_cfg:
        v_rec = validator(rec_val)
        v_cfg = validator(cfg_val)
        if v_rec is None or v_cfg is None:
            return None, True
        if v_rec != v_cfg:
            logger.warning("Field conflict for '%s': record=%r vs config=%r", field_name, rec_val, cfg_val)
            return None, True
        return v_rec, False
    elif has_rec:
        v = validator(rec_val)
        if v is None and rec_val is not None:
            return None, True
        return (v if v is not None else default), False
    elif has_cfg:
        v = validator(cfg_val)
        if v is None and cfg_val is not None:
            return None, True
        return (v if v is not None else default), False
    else:
        return default, False

def validate_solve_time(val):
    """
    Validates that a timing metric is a genuine finite, non-negative float.
    Rejects booleans (since bool is a subclass of int), None, negatives, NaN, inf, -inf,
    pd.NA, pd.NaT, and unparseable/invalid strings.
    Returns:
        float if valid, else None.
    """
    if val is None or isinstance(val, bool) or val is pd.NA or val is pd.NaT:
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
    Extracts a validated solve time strictly from the 'solve_time' field of a record.
    Does NOT fall back to 'wall_time'.
    Returns validated non-negative finite float, or None if invalid or missing.
    """
    if not isinstance(r, dict):
        return None
    return validate_solve_time(r.get("solve_time"))

def extract_effective_time(r):
    """
    Extracts timing with explicit provenance.
    Checks 'solve_time' first; if invalid/missing, falls back to 'wall_time'.
    Returns (time_value, source_name) where source_name is 'solve_time', 'wall_time', or None.
    """
    if not isinstance(r, dict):
        return None, None
    st = validate_solve_time(r.get("solve_time"))
    if st is not None:
        return st, "solve_time"
    wt = validate_solve_time(r.get("wall_time"))
    if wt is not None:
        return wt, "wall_time"
    return None, None

VALID_SOLVER_METHODS = {
    "cadical195": "sat",
    "kissat": "sat",
    "glucose4": "sat",
    "maplechrono": "sat",
    "minisat22": "sat",
    "lingeling": "sat",
    "cp_sat": "cpsat",
    "cplex": "cplex",
    "cplex_cp": "cplex_cp",
    "gurobi": "gurobi",
}


def get_canonical_key(record):
    """
    Extracts a deterministic canonical key (dataset, method, solver, flags, relabel_seed)
    from a task or result record dictionary.

    Guarantees:
    - Different method or solver produce different keys.
    - Missing solver does not falsely collapse different methods.
    - Different encodings, solving modes, EO strategies, symmetry breaking flags,
      worker counts, or relabel seeds produce different keys.
    - Checks and flags conflict between top-level record and config dictionary.
    - Preserves seed=0 distinctly from seed=None; rejects fractional/bool seeds.
    - Normalizes solver aliases (e.g. cp_sat/cpsat, cadical/cadical195).
    - Enforces solver <-> method consistency; rejects contradictory records.
    - Returns None if record is invalid, has conflict, or has unparseable mandatory identity.
    """
    if not isinstance(record, dict) or not record:
        return None

    dataset = record.get("dataset")
    if dataset is None or not isinstance(dataset, str) or not dataset.strip():
        return None
    dataset = str(dataset).strip()

    if "config" in record:
        cfg = record.get("config")
        if cfg is not None and not isinstance(cfg, dict):
            logger.warning("Record has invalid config type: %s (expected dict)", type(cfg).__name__)
            return None
    else:
        cfg = {}
    if cfg is None:
        cfg = {}

    # Relabel seed: check conflict and strict int >= 0
    seed, conflict = resolve_field_with_conflict_check(
        record, cfg, "relabel_seed",
        lambda v: strict_int(v, min_val=0),
        default=None
    )
    if conflict:
        return None
    relabel_seed = seed

    # Solver & Method: check top-level vs config
    rec_sol = record.get("solver")
    cfg_sol = cfg.get("solver")
    if rec_sol is not None and cfg_sol is not None:
        if normalize_solver_alias(rec_sol) != normalize_solver_alias(cfg_sol):
            logger.warning("Solver conflict: record=%r vs config=%r", rec_sol, cfg_sol)
            return None

    raw_solver = rec_sol if rec_sol is not None else cfg_sol
    norm_solver = normalize_solver_alias(raw_solver)

    rec_m = record.get("method")
    cfg_m = cfg.get("method")
    if rec_m is not None and cfg_m is not None:
        if normalize_method_alias(rec_m) != normalize_method_alias(cfg_m):
            logger.warning("Method conflict: record=%r vs config=%r", rec_m, cfg_m)
            return None

    raw_method = rec_m if rec_m is not None else cfg_m
    norm_method = normalize_method_alias(raw_method)

    if not norm_solver and not norm_method:
        # Gracefully handle SAT ablation test records where solver/method defaults to cadical195 / sat
        is_sat_context = (
            "relabel_seed" in record
            or "relabel_seed" in cfg
            or any(k in record or k in cfg for k in ("sb_rotation", "sb_reflection", "heuristic_ub", "encoding", "solving_mode", "eo_strategy"))
        )
        if is_sat_context:
            norm_method = "sat"
            norm_solver = "cadical195"
        else:
            logger.warning("Record missing required solver and method identity on dataset '%s'", dataset)
            return None

    # Check solver <-> method consistency
    if norm_solver and norm_solver in VALID_SOLVER_METHODS:
        expected_method = VALID_SOLVER_METHODS[norm_solver]
        if norm_method and norm_method != expected_method:
            logger.warning(
                "Solver/method conflict: solver=%r expects method=%r, got method=%r",
                norm_solver, expected_method, norm_method
            )
            return None
        method = expected_method
        solver = norm_solver
    elif norm_solver:
        # Synthetic / mock solver (e.g. s1, sol_a, sol_b in unit tests)
        method = norm_method if norm_method else "sat"
        solver = norm_solver
    else:
        # Solver missing but method provided
        method = norm_method
        solver = norm_method

    # Classify flags per method
    if method == "cpsat":
        # Worker count: strict positive int
        workers, conflict = resolve_field_with_conflict_check(
            record, cfg, "num_workers",
            strict_positive_int,
            default=1
        )
        if conflict or workers is None:
            return None

        # SB rotation: strict bool
        sb_rot, conflict = resolve_field_with_conflict_check(
            record, cfg, "sb_rotation",
            strict_bool,
            default=True
        )
        if conflict or sb_rot is None:
            return None
        flags = (workers, sb_rot)

    elif method in ("cplex", "cplex_cp", "gurobi"):
        sb_rot, conflict = resolve_field_with_conflict_check(
            record, cfg, "sb_rotation",
            strict_bool,
            default=None
        )
        if conflict:
            return None
        flags = (sb_rot,) if sb_rot is not None else ()

    elif method == "sat":
        rot, conflict = resolve_field_with_conflict_check(
            record, cfg, "sb_rotation",
            strict_bool,
            default=False
        )
        if conflict or rot is None:
            return None

        ref, conflict = resolve_field_with_conflict_check(
            record, cfg, "sb_reflection",
            strict_bool,
            default=False
        )
        if conflict or ref is None:
            return None

        heur, conflict = resolve_field_with_conflict_check(
            record, cfg, "heuristic_ub",
            strict_bool,
            default=False
        )
        if conflict or heur is None:
            return None

        enc, conflict = resolve_field_with_conflict_check(
            record, cfg, "encoding",
            lambda v: str(v).strip().lower() if v and isinstance(v, str) else None,
            default="order"
        )
        if conflict or not enc:
            return None

        mode, conflict = resolve_field_with_conflict_check(
            record, cfg, "solving_mode",
            lambda v: str(v).strip().lower() if v and isinstance(v, str) else None,
            default="incremental"
        )
        if conflict or not mode:
            return None

        eo, conflict = resolve_field_with_conflict_check(
            record, cfg, "eo_strategy",
            lambda v: str(v).strip().lower() if v and isinstance(v, str) else None,
            default="commander"
        )
        if conflict or not eo:
            return None

        if enc != "order" or mode != "incremental" or eo != "commander":
            flags = (rot, ref, heur, enc, mode, eo)
        else:
            flags = (rot, ref, heur)

    else:
        return None

    return (dataset, method, solver, flags, relabel_seed)

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
    Rejects None, NaN, pd.NA, pd.NaT, scalars, empty dicts, booleans, and non-dict objects.
    """
    if val is None or isinstance(val, bool) or val is pd.NA or val is pd.NaT:
        return False
    if isinstance(val, (float, np.floating)) and math.isnan(val):
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
    if val is None or isinstance(val, bool) or val is pd.NA or val is pd.NaT:
        return None
    if isinstance(val, (float, np.floating)) and math.isnan(val):
        return None
    if isinstance(val, dict):
        val = val.get("status")
        if val is None or isinstance(val, bool) or val is pd.NA or val is pd.NaT:
            return None
        if isinstance(val, (float, np.floating)) and math.isnan(val):
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
    elif s in ("ERROR", "MEMOUT"):
        return "ERROR"
    elif s in ("NUMERIC_ERROR",):
        return "NUMERIC_ERROR"
    elif s == "PARTIAL":
        return "PARTIAL"
    elif s == "CONFLICT":
        return "CONFLICT"
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

def get_run_identity(record):
    """
    Extracts (canonical_key, repetition) identifying an exact single execution run.
    Returns (canonical_key, rep) or None if either component is invalid.
    """
    canon = get_canonical_key(record)
    if canon is None:
        return None
    rep = get_repetition_identity(record)
    if rep is None:
        return None
    return (canon, rep)

def get_independent_engine_identity(record):
    """
    Extracts (method, solver) representing the independent algorithmic engine.
    Solver and method are normalized to canonical engine taxonomy.
    """
    canon = get_canonical_key(record)
    if canon is not None:
        return (canon[1], canon[2])
    if not isinstance(record, dict):
        return (None, None)
    cfg = record.get("config", {}) if isinstance(record.get("config"), dict) else {}
    sol = normalize_solver_alias(record.get("solver") or cfg.get("solver"))
    m = normalize_method_alias(record.get("method") or cfg.get("method"))
    return (m or None, sol or None)

def get_pairing_identity(record):
    """
    Extracts (dataset, solver, encoding, solving_mode, eo_strategy, relabel_seed)
    used to pair different parameter configurations (e.g. SB on vs off) on the same graph instance.
    Checks top-level and config dictionary with conflict detection via get_canonical_key.
    """
    canon = get_canonical_key(record)
    if canon is None:
        return None
    ds, method, solver, flags, seed = canon
    if method != "sat":
        return (ds, solver, None, None, None, seed)
    if len(flags) == 6:
        _rot, _ref, _heur, enc, mode, eo = flags
    else:
        enc, mode, eo = "order", "incremental", "commander"
    return (ds, solver, enc, mode, eo, seed)


_ABLATION_VARIANT_LABELS = {
    "table": {
        (False, False, False): "none",
        (True, False, False): "rotation",
        (True, True, False): "rot_ref",
        (False, False, True): "heur_ub",
        (True, True, True): "all",
    },
    "family": {
        (False, False, False): "none",
        (True, False, False): "rotation",
        (True, True, False): "rotation_reflection",
        (False, False, True): "heuristic_ub",
        (True, True, True): "all",
    },
    "plot": {
        (False, False, False): "none",
        (True, False, False): "rotation",
        (True, True, False): "rot+ref",
        (False, False, True): "heuristic-ub",
        (True, True, True): "all (SB+Heur)",
    },
}


def get_ablation_variant(rec_or_cfg, style="table"):
    """
    Single canonical source of truth for classifying ablation variants.
    Resolves sb_rotation, sb_reflection, heuristic_ub from top-level fields and/or
    nested config dict via canonical identity / resolve_field_with_conflict_check.
    """
    if not isinstance(rec_or_cfg, dict):
        return "other"

    rot = ref = heur = None
    has_identity = any(k in rec_or_cfg for k in ("dataset", "solver", "method"))
    if has_identity:
        canon = get_canonical_key(rec_or_cfg)
        if canon is None or canon[1] != "sat" or len(canon[3]) < 3:
            return "other"
        flags = canon[3]
        rot, ref, heur = bool(flags[0]), bool(flags[1]), bool(flags[2])
    else:
        cfg = rec_or_cfg.get("config") if isinstance(rec_or_cfg.get("config"), dict) else rec_or_cfg
        rec_view = rec_or_cfg if "config" in rec_or_cfg else {}
        rot, c1 = resolve_field_with_conflict_check(rec_view, cfg, "sb_rotation", strict_bool, default=False)
        ref, c2 = resolve_field_with_conflict_check(rec_view, cfg, "sb_reflection", strict_bool, default=False)
        heur, c3 = resolve_field_with_conflict_check(rec_view, cfg, "heuristic_ub", strict_bool, default=False)
        if c1 or c2 or c3 or rot is None or ref is None or heur is None:
            return "other"

    style_map = _ABLATION_VARIANT_LABELS.get(style, _ABLATION_VARIANT_LABELS["table"])
    return style_map.get((bool(rot), bool(ref), bool(heur)), "other")


def _ensure_aggregated_records(records, strict=True):
    """
    Ensures records are deduplicated on (canonical_key, rep) and aggregated across
    multiple repetitions per canonical_key. Already-aggregated singleton records
    (n_total is not None) are preserved without re-wrapping.
    """
    by_canon = {}
    for idx, r in enumerate(records, start=1):
        if not is_valid_record_dict(r):
            logger.warning("_ensure_aggregated_records: skipped invalid record at index %d", idx)
            continue
        ckey = get_canonical_key(r)
        if ckey is None:
            logger.warning("[%s] Skipped record with invalid canonical identity", _format_rec_location(r))
            continue
        by_canon.setdefault(ckey, []).append(r)

    result = []
    for ckey in sorted(by_canon.keys(), key=lambda x: str(x)):
        group = by_canon[ckey]
        if len(group) == 1 and group[0].get("n_total") is not None:
            result.append(group[0])
        else:
            deduped = resolve_duplicate_records(group, strict=strict)
            agg = aggregate_repetition_group(deduped)
            if agg is not None:
                agg["canonical_key"] = ckey
                result.append(agg)
    return result

def _format_rec_location(rec):
    if not isinstance(rec, dict):
        return "<unknown>:?"
    f = rec.get("_file") or "<memory>"
    ln = rec.get("_line")
    return f"{f}:{ln if ln is not None else '?'}"


def _merge_record_groups(rec_a, rec_b):
    gs = set()
    for r in (rec_a, rec_b):
        if not isinstance(r, dict):
            continue
        if isinstance(r.get("groups"), (list, tuple, set)):
            gs.update(str(g) for g in r["groups"])
        if r.get("_group"):
            gs.add(str(r["_group"]))
    return sorted(gs)


def _float_timing_differs(val_a, val_b):
    if val_a is None and val_b is None:
        return False
    if val_a is None or val_b is None:
        return True
    return not math.isclose(float(val_a), float(val_b), rel_tol=1e-12, abs_tol=1e-12)


def _build_conflict_record(rec_a, rec_b, reason_list):
    """Builds a deterministic CONFLICT sentinel record independent of input order."""
    # Pick deterministic base dictionary for identity fields
    try:
        repr_a = json.dumps({k: v for k, v in rec_a.items() if k not in ("_file", "_line", "_group", "groups")}, sort_keys=True, default=str)
        repr_b = json.dumps({k: v for k, v in rec_b.items() if k not in ("_file", "_line", "_group", "groups")}, sort_keys=True, default=str)
        base = dict(rec_a if repr_a <= repr_b else rec_b)
    except Exception:
        base = dict(rec_a)

    sources = set()
    for r in (rec_a, rec_b):
        if isinstance(r, dict):
            for s in r.get("_conflict_sources", []):
                sources.add(tuple(s) if isinstance(s, (list, tuple)) else (str(s), "?"))
            if r.get("_file") is not None or r.get("_line") is not None:
                sources.add((str(r.get("_file") or "<memory>"), str(r.get("_line") if r.get("_line") is not None else "?")))

    existing_reasons = set(reason_list)
    for r in (rec_a, rec_b):
        if isinstance(r, dict) and r.get("conflict_reason"):
            for part in str(r["conflict_reason"]).split("; "):
                if part:
                    existing_reasons.add(part)

    merged_groups = _merge_record_groups(rec_a, rec_b)
    if merged_groups:
        base["groups"] = merged_groups

    base["status"] = "CONFLICT"
    base["_conflict"] = True
    base["conflict_reason"] = "; ".join(sorted(existing_reasons))
    base["_conflict_sources"] = [list(s) for s in sorted(sources)]
    base["solve_time"] = None
    base["wall_time"] = None
    base["encode_time"] = None
    base["best_ub"] = -1
    base["assignment"] = None
    return base


def pick_better_record(rec_a, rec_b, strict=False):
    """
    Deterministically resolves two execution records representing the same run (canonical_key, rep).
    Symmetric and order-invariant: pick_better_record(A, B) matches pick_better_record(B, A).

    Duplicate Resolution Policy (A1):
    1. Exact duplicates (identical status, best_ub, assignment, solve_time, wall_time, validation_error):
       deduplicate cleanly, merging group membership and logging dropped source.
    2. Divergent status, best_ub, assignment, OR divergent runtime (solve_time / wall_time)
       for the same (canonical_key, rep):
       - Logs (file, line, reason) for the conflicting records.
       - If strict=True: raises DuplicateConflictError.
       - If strict=False: returns a deterministic CONFLICT sentinel record (never picks min/max runtime).
    """
    if rec_a is rec_b:
        return rec_a
    if not isinstance(rec_a, dict):
        return rec_b
    if not isinstance(rec_b, dict):
        return rec_a

    st_a = normalize_status(rec_a.get("status"))
    st_b = normalize_status(rec_b.get("status"))

    # Propagate existing conflict deterministically
    if st_a == "CONFLICT" or rec_a.get("_conflict") or st_b == "CONFLICT" or rec_b.get("_conflict"):
        conflict_rec = _build_conflict_record(rec_a, rec_b, [])
        msg = (
            f"Duplicate conflict propagated at {_format_rec_location(rec_a)} vs "
            f"{_format_rec_location(rec_b)}: {conflict_rec.get('conflict_reason', 'conflict')}"
        )
        logger.error(msg)
        if strict:
            raise DuplicateConflictError(msg)
        return conflict_rec

    ub_a = validate_best_ub(rec_a.get("best_ub"))
    ub_b = validate_best_ub(rec_b.get("best_ub"))
    err_a = bool(rec_a.get("validation_error"))
    err_b = bool(rec_b.get("validation_error"))
    asgn_a = rec_a.get("assignment")
    asgn_b = rec_b.get("assignment")
    t_a = validate_solve_time(rec_a.get("solve_time"))
    t_b = validate_solve_time(rec_b.get("solve_time"))
    w_a = validate_solve_time(rec_a.get("wall_time"))
    w_b = validate_solve_time(rec_b.get("wall_time"))

    # Also treat explicit None vs numeric or invalid non-None timing consistently:
    # if one record has raw solve_time=None and other has numeric, t_a != t_b catches it.
    reasons = []
    if st_a != st_b:
        pair = tuple(sorted((str(st_a), str(st_b))))
        reasons.append(f"status mismatch ({pair[0]} vs {pair[1]})")
    if ub_a != ub_b:
        pair = tuple(sorted((str(ub_a), str(ub_b))))
        reasons.append(f"best_ub mismatch ({pair[0]} vs {pair[1]})")
    if err_a != err_b:
        pair = tuple(sorted((str(err_a), str(err_b))))
        reasons.append(f"validation_error mismatch ({pair[0]} vs {pair[1]})")
    if asgn_a != asgn_b:
        reasons.append("assignment mismatch")
    if _float_timing_differs(t_a, t_b):
        pair = tuple(sorted((str(t_a), str(t_b))))
        reasons.append(f"solve_time mismatch ({pair[0]} vs {pair[1]})")
    if _float_timing_differs(w_a, w_b):
        pair = tuple(sorted((str(w_a), str(w_b))))
        reasons.append(f"wall_time mismatch ({pair[0]} vs {pair[1]})")

    if reasons:
        reasons_sorted = sorted(reasons)
        loc_pair = tuple(sorted((_format_rec_location(rec_a), _format_rec_location(rec_b))))
        msg = (
            f"Duplicate conflict between {loc_pair[0]} and {loc_pair[1]}: "
            f"{'; '.join(reasons_sorted)}"
        )
        logger.error(msg)
        if strict:
            raise DuplicateConflictError(msg)
        return _build_conflict_record(rec_a, rec_b, reasons_sorted)

    # Identical core data -> exact duplicate, deduplicate deterministically
    merged_groups = _merge_record_groups(rec_a, rec_b)
    try:
        repr_a = json.dumps({k: v for k, v in rec_a.items() if k not in ("_file", "_line", "_group", "groups")}, sort_keys=True, default=str)
        repr_b = json.dumps({k: v for k, v in rec_b.items() if k not in ("_file", "_line", "_group", "groups")}, sort_keys=True, default=str)
        if repr_a <= repr_b:
            winner, dropped = dict(rec_a), rec_b
        else:
            winner, dropped = dict(rec_b), rec_a
    except Exception:
        winner, dropped = dict(rec_a), rec_b

    if merged_groups:
        winner["groups"] = merged_groups

    logger.info(
        "[%s] Exact duplicate dropped (identical data to %s)",
        _format_rec_location(dropped),
        _format_rec_location(winner),
    )
    return winner

def resolve_duplicate_records(records, strict=False):
    """
    Deduplicates a list/iterable of execution records by run identity (canonical_key, rep).
    Input-order invariant.
    """
    by_run_id = {}
    for idx, r in enumerate(records, start=1):
        if not is_valid_record_dict(r):
            logger.warning("resolve_duplicate_records: skipped invalid record at index %d (non-dict or empty)", idx)
            continue
        run_id = get_run_identity(r)
        if run_id is None:
            logger.warning(
                "[%s] Skipped record with invalid canonical or repetition identity",
                _format_rec_location(r),
            )
            continue
        if run_id in by_run_id:
            by_run_id[run_id] = pick_better_record(by_run_id[run_id], r, strict=strict)
        else:
            by_run_id[run_id] = r
    return [by_run_id[k] for k in sorted(by_run_id.keys(), key=lambda x: (str(x[0]), x[1]))]

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
    - Median genuine solve_time, wall_time, encode_time
    - Attached list of reps
    """
    if not reps:
        return None

    cleaned_reps = []
    for r in reps:
        if not isinstance(r, dict):
            continue
        rc = dict(r)
        st_norm = normalize_status(rc.get("status"))
        if st_norm == "CONFLICT" or rc.get("_conflict"):
            cat = "CONFLICT"
            rc["status"] = "CONFLICT"
            rc["_conflict"] = True
            rc["solve_time"] = None
            rc["wall_time"] = None
            rc["encode_time"] = None
            rc["best_ub"] = -1
            rc["assignment"] = None
            rc["_cat"] = cat
            cleaned_reps.append(rc)
            continue

        ub_val = validate_best_ub(rc.get("best_ub"))
        lb_val = validate_lower_bound(rc.get("lower_bound") if rc.get("lower_bound") is not None else rc.get("best_lb"))

        # Bound contradiction: ub < lb is impossible in minimization
        has_bound_conflict = (ub_val is not None and lb_val is not None and ub_val < lb_val)
        if rc.get("validation_error") or has_bound_conflict:
            rc["validation_error"] = True
            cat = "ERROR"
            rc["status"] = "ERROR"
            rc["best_ub"] = -1
            rc["assignment"] = None
        else:
            cat = categorize_status(st_norm)
            if cat in ("ERROR", "LICENSE_ERROR", "UNAVAILABLE", "SIZE_LIMIT", "INFEASIBLE", "UNKNOWN"):
                rc["best_ub"] = -1
                rc["assignment"] = None
        rc["_cat"] = cat
        cleaned_reps.append(rc)
    reps = cleaned_reps

    n_total = len(reps)
    if n_total == 0:
        return None

    n_conflict = sum(1 for r in reps if r["_cat"] == "CONFLICT" or r.get("_conflict"))
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

    if n_conflict > 0:
        conflict_reasons = sorted(
            {str(r.get("conflict_reason")) for r in reps if r.get("conflict_reason")}
        )
        rep_base = sorted(
            [r for r in reps if r["_cat"] == "CONFLICT"],
            key=lambda x: str(x.get("conflict_reason", "")),
        )[0]
        logger.warning(
            "aggregate_repetition_group: excluding entire conflicting group (dataset=%s, solver=%s) from aggregate and marking CONFLICT: %s",
            rep_base.get("dataset"),
            rep_base.get("solver"),
            "; ".join(conflict_reasons) or "duplicate conflict",
        )
        all_groups = set()
        for r in reps:
            for g in r.get("groups", []):
                all_groups.add(g)
            if "_group" in r:
                all_groups.add(r["_group"])
        agg = dict(rep_base)
        agg["status"] = "CONFLICT"
        agg["_conflict"] = True
        agg["conflict_reason"] = "; ".join(conflict_reasons) or "duplicate conflict"
        agg["n_total"] = n_total
        agg["n_algorithmic"] = 0
        agg["n_optimal"] = 0
        agg["n_timeout"] = 0
        agg["n_interrupted"] = 0
        agg["n_feasible"] = 0
        agg["n_infeasible"] = 0
        agg["n_error"] = 0
        agg["n_size_limit"] = 0
        agg["n_unavailable"] = 0
        agg["n_license_error"] = 0
        agg["n_unknown"] = 0
        agg["solve_time"] = None
        agg["solve_time_source"] = None
        agg["timing_fallback"] = False
        agg["wall_time"] = None
        agg["effective_time"] = None
        agg["effective_time_source"] = None
        agg["encode_time"] = None
        agg["num_iterations"] = None
        agg["last_iter_time"] = None
        agg.pop("par2", None)
        agg["best_ub"] = -1
        agg["assignment"] = None
        agg["groups"] = sorted(list(all_groups))
        agg["reps"] = reps
        return agg

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
        t = validate_solve_time(r.get("solve_time"))
        return (0, t) if t is not None else (1, float("inf"))

    def _safe_ub_key(r):
        ub = validate_best_ub(r.get("best_ub"))
        return (0, ub) if ub is not None else (1, float("inf"))

    # Select representative record to seed the aggregated dictionary
    if agg_status in ("OPTIMAL", "PARTIAL"):
        rep_pool = [r for r in reps if r["_cat"] == "OPTIMAL"]
        valid_ub_pool = [r for r in rep_pool if validate_best_ub(r.get("best_ub")) is not None]
        cand_pool = valid_ub_pool if valid_ub_pool else rep_pool
        reps_sorted = sorted(cand_pool, key=_safe_solve_key)
        rep_chosen = reps_sorted[len(reps_sorted) // 2]
    elif agg_status == "FEASIBLE":
        rep_pool = [r for r in reps if r["_cat"] == "FEASIBLE"]
        valid_ub_pool = [r for r in rep_pool if validate_best_ub(r.get("best_ub")) is not None]
        cand_pool = valid_ub_pool if valid_ub_pool else rep_pool
        reps_sorted = sorted(cand_pool, key=lambda x: (_safe_ub_key(x), _safe_solve_key(x)))
        rep_chosen = reps_sorted[0]
    else:
        matching = [r for r in reps if r["_cat"] == agg_status]
        rep_pool = matching if matching else reps
        reps_sorted = sorted(rep_pool, key=_safe_solve_key)
        rep_chosen = reps_sorted[len(reps_sorted) // 2]

    genuine_solve_times = [
        validate_solve_time(r.get("solve_time"))
        for r in rep_pool
        if validate_solve_time(r.get("solve_time")) is not None
    ]
    wall_times = [
        validate_solve_time(r.get("wall_time"))
        for r in rep_pool
        if validate_solve_time(r.get("wall_time")) is not None
    ]

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

    if genuine_solve_times:
        agg["solve_time"] = float(np.median(genuine_solve_times))
        agg["solve_time_source"] = "solve_time"
        agg["timing_fallback"] = False
    else:
        agg["solve_time"] = None
        agg["solve_time_source"] = None
        agg["timing_fallback"] = False

    if wall_times:
        agg["wall_time"] = float(np.median(wall_times))
    else:
        agg["wall_time"] = validate_solve_time(rep_chosen.get("wall_time"))

    # Explicit effective time with provenance
    if genuine_solve_times:
        agg["effective_time"] = float(np.median(genuine_solve_times))
        agg["effective_time_source"] = "solve_time"
    elif wall_times:
        agg["effective_time"] = float(np.median(wall_times))
        agg["effective_time_source"] = "wall_time"
    else:
        agg["effective_time"] = None
        agg["effective_time_source"] = None

    if encode_times:
        agg["encode_time"] = float(np.median(encode_times))
    else:
        agg["encode_time"] = validate_solve_time(rep_chosen.get("encode_time"))

    iter_counts = []
    last_iter_times = []
    for r in rep_pool:
        it = r.get("num_iterations")
        if isinstance(it, (int, float)) and not isinstance(it, bool) and math.isfinite(it) and it >= 0:
            iter_counts.append(float(it))
        t_last = validate_solve_time(r.get("last_iter_time"))
        if t_last is None:
            per_iter = r.get("per_iter_times")
            if isinstance(per_iter, list) and per_iter:
                t_last = validate_solve_time(per_iter[-1])
        if t_last is not None:
            last_iter_times.append(t_last)

    if iter_counts:
        med_iter = float(np.median(iter_counts))
        agg["num_iterations"] = int(med_iter) if med_iter.is_integer() else med_iter
    else:
        agg["num_iterations"] = None

    if last_iter_times:
        agg["last_iter_time"] = float(np.median(last_iter_times))
    else:
        agg["last_iter_time"] = None

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
        agg["best_ub"] = ub_c if ub_c is not None else -1

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
    Group records by unique configuration using get_canonical_key(record),
    and aggregate across repetitions using aggregate_repetition_group.
    """
    groups = {}
    for r in records:
        if not is_valid_record_dict(r):
            continue
        key = get_canonical_key(r)
        if key is None:
            continue
        groups.setdefault(key, []).append(r)

    aggregated = []
    for key, reps in groups.items():
        agg = aggregate_repetition_group(reps)
        if agg is not None:
            agg["canonical_key"] = key
            aggregated.append(agg)

    return aggregated

def load_records(jsonl_source=None, strict=False):
    """
    Load all valid records from runs.jsonl (or pattern/fallback),
    and aggregate across repetitions using aggregate_repetitions.

    Data integrity and error reporting:
    - Distinguishes empty lines, metadata records, JSON syntax errors,
      non-dict JSON objects, and missing required identity (dataset, solver/method).
    - In strict=True mode: raises ValueError on syntax errors, non-objects, or missing identity.
    - In strict=False mode (default): logs structured warnings with file, line number, and reason.
    - Deduplicates duplicate (canonical_key, rep) entries safely.
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

    if not files:
        logger.warning("load_records: no JSONL files found.")
        return []

    total_lines = 0
    empty_count = 0
    meta_count = 0
    syntax_error_count = 0
    non_obj_count = 0
    missing_id_count = 0
    seen_runs = {}  # (canonical_key, rep) -> index in raw_records

    for filepath in files:
        group_name = Path(filepath).stem
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    total_lines += 1
                    s_line = line.strip()
                    if not s_line:
                        empty_count += 1
                        continue

                    try:
                        data = json.loads(s_line)
                    except json.JSONDecodeError as e:
                        syntax_error_count += 1
                        msg = f"[{filepath}:{line_no}] JSON syntax error: {e}"
                        if strict:
                            raise ValueError(f"JSONL integrity violations: {msg}") from e
                        logger.warning(msg)
                        continue

                    if not isinstance(data, dict):
                        non_obj_count += 1
                        msg = f"[{filepath}:{line_no}] Expected JSON object, got {type(data).__name__}"
                        if strict:
                            raise ValueError(f"JSONL integrity violations: {msg}")
                        logger.warning(msg)
                        continue

                    if data.get("_type") == "metadata" or data.get("type") == "metadata" or ("environment" in data and "dataset" not in data):
                        meta_count += 1
                        continue

                    dataset = data.get("dataset")
                    if not dataset or not isinstance(dataset, str) or not dataset.strip():
                        missing_id_count += 1
                        msg = f"[{filepath}:{line_no}] Missing mandatory 'dataset' in record"
                        if strict:
                            raise ValueError(f"JSONL integrity violations: {msg}")
                        logger.warning(msg)
                        continue

                    raw_sol = data.get("solver")
                    raw_method = data.get("method")
                    cfg = data.get("config") if isinstance(data.get("config"), dict) else {}
                    if not raw_sol and not raw_method and not cfg.get("solver") and not cfg.get("method"):
                        missing_id_count += 1
                        msg = f"[{filepath}:{line_no}] Record on dataset '{dataset}' missing both 'solver' and 'method'"
                        if strict:
                            raise ValueError(f"JSONL integrity violations: {msg}")
                        logger.warning(msg)
                        continue

                    if "groups" not in data or not isinstance(data["groups"], list):
                        data["groups"] = [data.get("_group", group_name)]

                    raw_rep = data.get("rep")
                    raw_repetition = data.get("repetition")
                    if raw_rep is not None and raw_repetition is not None:
                        r1 = strict_positive_int(raw_rep)
                        r2 = strict_positive_int(raw_repetition)
                        if r1 is not None and r2 is not None and r1 != r2:
                            missing_id_count += 1
                            msg = f"[{filepath}:{line_no}] Conflicting repetition fields: 'rep'={raw_rep} vs 'repetition'={raw_repetition}"
                            if strict:
                                raise ValueError(f"JSONL integrity violations: {msg}")
                            logger.warning(msg)
                            continue

                    raw_val = raw_rep if raw_rep is not None else (raw_repetition if raw_repetition is not None else 1)
                    rep_num = strict_positive_int(raw_val)
                    if rep_num is None:
                        missing_id_count += 1
                        msg = f"[{filepath}:{line_no}] Invalid repetition number: {raw_val!r}"
                        if strict:
                            raise ValueError(f"JSONL integrity violations: {msg}")
                        logger.warning(msg)
                        continue

                    data["rep"] = rep_num
                    data["repetition"] = rep_num
                    data["_file"] = filepath
                    data["_line"] = line_no

                    c_key = get_canonical_key(data)
                    if c_key is None:
                        missing_id_count += 1
                        msg = f"[{filepath}:{line_no}] Invalid canonical identity on dataset '{dataset}'"
                        if strict:
                            raise ValueError(f"JSONL integrity violations: {msg}")
                        logger.warning(msg)
                        continue

                    run_id = (c_key, rep_num)
                    if run_id in seen_runs:
                        prev_idx = seen_runs[run_id]
                        prev_rec = raw_records[prev_idx]
                        try:
                            better = pick_better_record(prev_rec, data, strict=strict)
                            raw_records[prev_idx] = better
                        except DuplicateConflictError as e:
                            msg = f"[{filepath}:{line_no}] Critical duplicate conflict: {e} for {c_key} rep {rep_num}"
                            if strict:
                                raise DuplicateConflictError(f"JSONL integrity violations: {msg}") from e
                            logger.error(msg)
                        continue

                    seen_runs[run_id] = len(raw_records)
                    raw_records.append(data)
        except Exception as e:
            if strict:
                raise
            logger.error("Error reading JSONL file %s: %s", filepath, e)

    skipped_total = syntax_error_count + non_obj_count + missing_id_count
    if skipped_total > 0:
        logger.warning(
            "load_records: scanned %d lines in %d file(s). Valid: %d, Meta: %d, Empty: %d, Skipped: %d (syntax: %d, non-obj: %d, missing_id: %d)",
            total_lines, len(files), len(raw_records), meta_count, empty_count, skipped_total,
            syntax_error_count, non_obj_count, missing_id_count
        )

    return aggregate_repetitions(raw_records)

def get_source_key(record):
    """Extracts (solver_or_method, canonical_config) from a record using get_canonical_key."""
    canon_key = get_canonical_key(record)
    if canon_key is None:
        return ("unknown", ())
    dataset, method, solver, flags, seed = canon_key
    src_solver = solver if solver else method
    return (src_solver, (method, flags))

def cross_validate_optima(records, output_dir=None, strict_engines=True):
    """
    Cross-validation check:
    1. Group OPTIMAL records by (dataset, relabel_seed) and independent source.
       If strict_engines=True: independent source is strictly (method, normalized_solver).
       If strict_engines=False: independent source is (method, normalized_solver, flags).
       Verified only when >= 2 distinct independent sources reach identical valid OPTIMAL best_ub.
       Repetitions of the same source are deduplicated, not counted as separate sources.
       If only 1 distinct source is present, mark as 'single_source'.
       If distinct sources report conflicting optima, mark as 'conflict' and raise ValueError.
    2. Check across seeds: all seeds of the same base dataset must reach the exact same optimum.
       If different seeds report conflicting optima, mark as 'conflict' and raise ValueError.
    3. Check incumbents: for any dataset with a proven optimum `opt`, every non-OPTIMAL record
       with valid best_ub must satisfy best_ub >= opt; otherwise raises ValueError with offending record.
    4. Exports cross_validation.csv to output_dir if output_dir is provided.
    """
    optimal_by_inst = {}
    for r in records:
        if not is_valid_record_dict(r) or r.get("validation_error"):
            continue
        if is_optimal_status(r):
            canon = get_canonical_key(r)
            if canon is None:
                continue
            ds = canon[0]
            seed = canon[4]
            ub = validate_best_ub(r.get("best_ub"))
            if ub is None:
                continue
            if strict_engines:
                src_key = (canon[1], canon[2])
            else:
                src_key = (canon[1], canon[2], canon[3])
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
                mismatches.append((ds, f"Internal conflict across repetitions of source {src_key} on seed {seed}: {unique_key_ubs}"))
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
        if not is_valid_record_dict(r) or r.get("validation_error"):
            continue
        ds = r.get("dataset")
        st = normalize_status(r.get("status"))
        if ds in proven_optima_by_ds and st != "OPTIMAL":
            ub = validate_best_ub(r.get("best_ub"))
            if ub is not None:
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

        for src_key, ub in sources.items():
            if strict_engines:
                method_name, solver_name = src_key
                flags = ()
            else:
                method_name, solver_name, flags = src_key
            disp_sol = solver_name if solver_name else method_name
            row_dict = {
                "dataset": ds,
                "solver_or_method": disp_sol,
                "canonical_config": str((method_name, flags)),
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
    elif st == "CONFLICT":
        return "CONFLICT"
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
            elif st in ("UNAVAILABLE", "LICENSE_ERROR", "UNKNOWN", "CONFLICT") or st is None:
                # Solver environment unavailability, missing license, unknown, or duplicate conflict excluded from algorithmic PAR2
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

    recs_by_ds_var = {}
    meta = {}
    for r in ablation_recs:
        if not is_valid_record_dict(r) or r.get("validation_error"):
            continue
        raw_sol = r.get("solver")
        if not raw_sol:
            continue
        sol = normalize_solver_alias(raw_sol)
        if sol != "cadical195":
            continue

        canon = get_canonical_key(r)
        if canon is None:
            continue
        if canon[4] is not None:
            continue

        cfg = r.get("config", {}) if isinstance(r.get("config"), dict) else {}
        enc = str(r.get("encoding") or cfg.get("encoding", "order")).strip().lower()
        if enc != "order":
            continue
        mode = str(r.get("solving_mode") or cfg.get("solving_mode", "incremental")).strip().lower()
        if mode != "incremental":
            continue
        eo = str(r.get("eo_strategy") or cfg.get("eo_strategy", "commander")).strip().lower()
        if eo != "commander":
            continue

        ds = r["dataset"]
        meta[ds] = (r.get("N", 0), r.get("num_edges", 0))
        var = get_ablation_variant(r, style="table")
        if var == "other":
            continue

        recs_by_ds_var.setdefault((ds, var), []).append(r)

    data_by_ds = {}
    for (ds, var), var_recs in recs_by_ds_var.items():
        # Check duplicate OPTIMAL conflict
        optimal_ubs = set()
        for r in var_recs:
            if normalize_status(r.get("status")) == "OPTIMAL":
                ub = validate_best_ub(r.get("best_ub"))
                if ub is not None:
                    optimal_ubs.add(ub)
        if len(optimal_ubs) > 1:
            raise ValueError(
                f"make_table_ablation: conflicting OPTIMAL best_ub on dataset '{ds}' variant '{var}': {sorted(list(optimal_ubs))}"
            )

        if len(var_recs) == 1:
            data_by_ds.setdefault(ds, {})[var] = var_recs[0]
        else:
            sorted_var_recs = sorted(var_recs, key=lambda x: (
                x.get("rep", 0),
                validate_solve_time(x.get("solve_time")) or float("inf"),
                validate_solve_time(x.get("wall_time")) or float("inf")
            ))
            agg = aggregate_repetition_group(sorted_var_recs)
            if agg is not None:
                data_by_ds.setdefault(ds, {})[var] = agg

    columns = ["none", "rotation", "rot_ref", "heur_ub", "all"]
    variants_present = {var for ds_map in data_by_ds.values() for var in ds_map}
    missing_variants = [col for col in columns if col not in variants_present]
    if missing_variants:
        print(f"[WARNING] Skipping {base_filename}: missing variants {missing_variants} in records (need {columns}).")
        return None

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
    return df_display, summary, df_raw

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
        sol = str(r.get("solver") or "").lower()
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
    return df_display, summary, df_raw

def make_table_methods(records, output_dirs=("results_v2/tables",)):
    """Generate Methods comparison table (SAT vs CP-SAT vs CPLEX)."""
    method_recs = [r for r in records if "methods" in r.get("groups", [])]
    if not method_recs:
        return

    methods = ["sat_best", "cpsat", "cplex"]

    sat_candidates = {}
    cpsat_candidates = {}
    cplex_candidates = {}
    meta = {}

    for r in method_recs:
        ds = r["dataset"]
        meta[ds] = (r.get("N", 0), r.get("num_edges", 0))
        sol = normalize_solver_alias(r.get("solver"))
        m = str(r.get("method") or "").strip().lower()

        if sol == "cp_sat" or m in ("cpsat", "cp_sat"):
            cpsat_candidates.setdefault(ds, []).append(r)
        elif sol == "cplex" or m == "cplex":
            cplex_candidates.setdefault(ds, []).append(r)
        elif (m == "sat" or sol in ("cadical195", "glucose4", "kissat", "minisat22", "maplechrono", "lingeling")) and m not in ("cpsat", "cp_sat", "cplex", "cplex_cp", "gurobi") and sol not in ("cp_sat", "cplex", "cplex_cp", "gurobi"):
            sat_candidates.setdefault(ds, []).append(r)
        else:
            logger.warning("make_table_methods: ignoring unrecognized candidate for dataset %s: sol=%r, method=%r", ds, sol, m)

    data_by_ds = {}
    all_ds = sorted(
        list(set(list(sat_candidates.keys()) + list(cpsat_candidates.keys()) + list(cplex_candidates.keys()))),
        key=lambda d: (meta.get(d, (0, 0))[0], d)
    )

    def _sat_rank(rc):
        st = normalize_status(rc.get("status"))
        tier_map = {
            "OPTIMAL": 0, "FEASIBLE": 1, "TIMEOUT": 2, "INFEASIBLE": 3,
            "SIZE_LIMIT": 4, "ERROR": 5, "UNKNOWN": 6
        }
        tier = tier_map.get(st, 6)
        t = extract_repetition_solve_time(rc)
        ub = validate_best_ub(rc.get("best_ub"))
        c_key = get_canonical_key(rc)
        c_key_str = str(c_key) if c_key is not None else ""

        sol = normalize_solver_alias(rc.get("solver"))
        sol_prefs = ["cadical195", "kissat", "glucose4", "maplechrono", "minisat22", "lingeling"]
        sol_idx = sol_prefs.index(sol) if sol in sol_prefs else len(sol_prefs)

        if tier == 0:
            time_metric = t if (t is not None and t > 0) else float("inf")
            ub_metric = ub if ub is not None else float("inf")
            return (tier, time_metric, ub_metric, sol_idx, str(sol), c_key_str)
        elif tier == 1:
            time_metric = t if (t is not None and t > 0) else float("inf")
            ub_metric = ub if ub is not None else float("inf")
            return (tier, ub_metric, time_metric, sol_idx, str(sol), c_key_str)
        else:
            # Non-solved tiers do NOT use runtime as success metric
            return (tier, float("inf"), float("inf"), sol_idx, str(sol), c_key_str)

    for ds in all_ds:
        data_by_ds[ds] = {}

        # 1. Deterministic selection for sat_best across aggregated SAT configurations
        cand_sat = sat_candidates.get(ds, [])
        if cand_sat:
            # Group candidate SAT runs by exact configuration identity (get_canonical_key)
            sat_by_config = {}
            for r in cand_sat:
                c_key = get_canonical_key(r)
                if c_key is None:
                    continue
                sat_by_config.setdefault(c_key, []).append(r)

            config_aggs = []
            for c_key, cfg_recs in sorted(sat_by_config.items(), key=lambda x: str(x[0])):
                if len(cfg_recs) == 1:
                    config_aggs.append(cfg_recs[0])
                else:
                    sorted_cfg = sorted(cfg_recs, key=lambda x: (
                        get_repetition_identity(x) or 1,
                        validate_solve_time(x.get("solve_time")) or float("inf"),
                        validate_solve_time(x.get("wall_time")) or float("inf"),
                        str(x)
                    ))
                    agg = aggregate_repetition_group(sorted_cfg)
                    if agg is not None:
                        config_aggs.append(agg)

            if config_aggs:
                best_sat = sorted(config_aggs, key=_sat_rank)[0]
                data_by_ds[ds]["sat_best"] = best_sat

        # 2. Select cpsat
        cand_cp = cpsat_candidates.get(ds, [])
        if cand_cp:
            cp_by_config = {}
            for r in cand_cp:
                c_key = get_canonical_key(r)
                if c_key is None:
                    continue
                cp_by_config.setdefault(c_key, []).append(r)
            cp_aggs = []
            for c_key, cfg_recs in sorted(cp_by_config.items(), key=lambda x: str(x[0])):
                if len(cfg_recs) == 1:
                    cp_aggs.append(cfg_recs[0])
                else:
                    sorted_cfg = sorted(cfg_recs, key=lambda x: (
                        get_repetition_identity(x) or 1,
                        validate_solve_time(x.get("solve_time")) or float("inf"),
                        validate_solve_time(x.get("wall_time")) or float("inf"),
                        str(x)
                    ))
                    agg = aggregate_repetition_group(sorted_cfg)
                    if agg is not None:
                        cp_aggs.append(agg)
            if cp_aggs:
                data_by_ds[ds]["cpsat"] = sorted(cp_aggs, key=_sat_rank)[0]

        # 3. Select cplex
        cand_cpl = cplex_candidates.get(ds, [])
        if cand_cpl:
            cpl_by_config = {}
            for r in cand_cpl:
                c_key = get_canonical_key(r)
                if c_key is None:
                    continue
                cpl_by_config.setdefault(c_key, []).append(r)
            cpl_aggs = []
            for c_key, cfg_recs in sorted(cpl_by_config.items(), key=lambda x: str(x[0])):
                if len(cfg_recs) == 1:
                    cpl_aggs.append(cfg_recs[0])
                else:
                    sorted_cfg = sorted(cfg_recs, key=lambda x: (
                        get_repetition_identity(x) or 1,
                        validate_solve_time(x.get("solve_time")) or float("inf"),
                        validate_solve_time(x.get("wall_time")) or float("inf"),
                        str(x)
                    ))
                    agg = aggregate_repetition_group(sorted_cfg)
                    if agg is not None:
                        cpl_aggs.append(agg)
            if cpl_aggs:
                data_by_ds[ds]["cplex"] = sorted(cpl_aggs, key=_sat_rank)[0]

    ds_sorted = all_ds

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
    return df_display, summary, df_raw

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

    def _extract_seed(rec):
        if not isinstance(rec, dict):
            return None
        cfg = rec.get("config", {}) if isinstance(rec.get("config"), dict) else {}
        s_rec = rec.get("relabel_seed")
        s_cfg = cfg.get("relabel_seed")
        val = s_rec if s_rec is not None else s_cfg
        return strict_int(val, min_val=0)

    family_recs = [r for r in records if (is_valid_record_dict(r) and _extract_seed(r) is not None)]
    if not family_recs:
        print("[WARNING] Skipping table_ablation_family: no records with relabel_seed found.")
        return None

    family_recs = _ensure_aggregated_records(family_recs, strict=True)

    pairs_map = {}
    for r in family_recs:
        base_key = get_pairing_identity(r)
        if base_key is None or base_key[5] is None:
            continue
        var = get_ablation_variant(r, style="family")
        if var == "other":
            continue

        existing = pairs_map.setdefault(base_key, {}).get(var)
        if existing is not None:
            pairs_map[base_key][var] = pick_better_record(existing, r, strict=True)
        else:
            pairs_map[base_key][var] = r

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
            1 for base_key, vmap in pairs_map.items()
            if var_target in vmap and var_ref in vmap
            and is_optimal_status(vmap[var_target])
            and is_optimal_status(vmap[var_ref])
            and extract_repetition_solve_time(vmap[var_target]) is not None
            and extract_repetition_solve_time(vmap[var_target]) > 0
            and extract_repetition_solve_time(vmap[var_ref]) is not None
            and extract_repetition_solve_time(vmap[var_ref]) > 0
        )
        if n_pairs < 1:
            print(f"[WARNING] Skipping table_ablation_family: comparison '{var_target} vs {var_ref}' has 0 valid pairs (>=1 required).")
            return None

    rows = []
    for comp_group, var_label, var_target, var_ref in comparisons:
        ratios = []
        faster_count = 0
        total_count = 0
        for base_key, vmap in pairs_map.items():
            if var_target in vmap and var_ref in vmap:
                rec_t = vmap[var_target]
                rec_r = vmap[var_ref]
                if is_optimal_status(rec_t) and is_optimal_status(rec_r):
                    t_target = extract_repetition_solve_time(rec_t)
                    t_ref = extract_repetition_solve_time(rec_r)
                    if (
                        t_target is not None and math.isfinite(t_target) and t_target > 0
                        and t_ref is not None and math.isfinite(t_ref) and t_ref > 0
                    ):
                        ratio = t_target / t_ref
                        if math.isfinite(ratio) and ratio > 0:
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
    datasets = sorted(list(set(base_key[0] for base_key in pairs_map.keys())))
    can_generate_pg = True
    if not datasets:
        can_generate_pg = False
    else:
        for ds in datasets:
            for comp_group, var_label, var_target, var_ref in comparisons:
                count = sum(
                    1 for base_key, vmap in pairs_map.items()
                    if base_key[0] == ds and var_target in vmap and var_ref in vmap
                    and is_optimal_status(vmap[var_target])
                    and is_optimal_status(vmap[var_ref])
                    and extract_repetition_solve_time(vmap[var_target]) is not None
                    and extract_repetition_solve_time(vmap[var_target]) > 0
                    and extract_repetition_solve_time(vmap[var_ref]) is not None
                    and extract_repetition_solve_time(vmap[var_ref]) > 0
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
                for base_key, vmap in pairs_map.items():
                    if base_key[0] == ds and var_target in vmap and var_ref in vmap:
                        rec_t = vmap[var_target]
                        rec_r = vmap[var_ref]
                        if is_optimal_status(rec_t) and is_optimal_status(rec_r):
                            t_target = extract_repetition_solve_time(rec_t)
                            t_ref = extract_repetition_solve_time(rec_r)
                            if (
                                t_target is not None and math.isfinite(t_target) and t_target > 0
                                and t_ref is not None and math.isfinite(t_ref) and t_ref > 0
                            ):
                                ratio = t_target / t_ref
                                if math.isfinite(ratio) and ratio > 0:
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

    def _extract_seed(rec):
        if not isinstance(rec, dict):
            return None
        cfg = rec.get("config", {}) if isinstance(rec.get("config"), dict) else {}
        s_rec = rec.get("relabel_seed")
        s_cfg = cfg.get("relabel_seed")
        val = s_rec if s_rec is not None else s_cfg
        return strict_int(val, min_val=0)

    family_recs = [r for r in records if (is_valid_record_dict(r) and _extract_seed(r) is not None)]
    if not family_recs:
        print("[WARNING] Skipping table_ablation_phases: no records with relabel_seed found.")
        return None

    if all(get_canonical_key(r) is not None for r in family_recs):
        family_recs = _ensure_aggregated_records(family_recs, strict=True)

    variants = ["none", "rotation", "rotation_reflection", "heuristic_ub", "all"]
    sizes = sorted(list(set(r.get("N", 0) for r in family_recs if r.get("N"))))
    if not sizes:
        print("[WARNING] Skipping table_ablation_phases: no sizes found.")
        return None

    def _clean_iter_times(r):
        pit = r.get("per_iter_times")
        if not isinstance(pit, (list, tuple)) or len(pit) == 0:
            return None
        cleaned = []
        for x in pit:
            xv = validate_solve_time(x)
            if xv is None or not math.isfinite(xv):
                return None
            cleaned.append(xv)
        return cleaned

    # Condition: each (size, variant) must have >= 1 valid OPTIMAL record with valid per_iter_times
    can_generate_phases = True
    for n in sizes:
        for var in variants:
            count = sum(
                1 for r in family_recs
                if r.get("N") == n and get_ablation_variant(r, style="family") == var and is_optimal_status(r)
                and _clean_iter_times(r) is not None
            )
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
                if r.get("N") == n and get_ablation_variant(r, style="family") == var and is_optimal_status(r):
                    cleaned_iter = _clean_iter_times(r)
                    if cleaned_iter is not None and len(cleaned_iter) > 0:
                        last_t = cleaned_iter[-1]
                        prev_t = sum(cleaned_iter[:-1])
                        last_times.append(last_t)
                        prev_times.append(prev_t)
                        st = extract_repetition_solve_time(r)
                        if st is not None and math.isfinite(st) and st > 0:
                            ratio = last_t / st
                            if math.isfinite(ratio) and ratio >= 0:
                                ratios.append(ratio)

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
        st = normalize_status(r.get("status")) or (str(r.get("status")) if r.get("status") is not None else "-")
        ub_val = validate_best_ub(r.get("best_ub"))
        ub_str = str(ub_val) if ub_val is not None else "-"

        solve_t_val = validate_solve_time(r.get("solve_time"))
        solve_t_str = f"{solve_t_val:.4f}" if solve_t_val is not None else "-"

        wall_t_val = validate_solve_time(r.get("wall_time"))
        wall_t_str = f"{wall_t_val:.4f}" if wall_t_val is not None else "-"

        n_iters = r.get("num_iterations", len(r.get("per_iter_times", [])))

        lb_deg = get_lb_degree(ds)
        best_lb_raw = r.get("best_lb")
        best_lb_val = validate_best_ub(best_lb_raw)
        lb_solver = best_lb_val if best_lb_val is not None else "-"
        lb = max(lb_deg, best_lb_val) if best_lb_val is not None else lb_deg

        if st == "OPTIMAL":
            ub_minus_lb_str = "0"
        elif st == "TIMEOUT":
            if ub_val is not None and lb is not None:
                ub_minus_lb_str = str(ub_val - lb)
            else:
                ub_minus_lb_str = "-"
        else:
            ub_minus_lb_str = "-"

        rows.append({
            "dataset": ds,
            "N": r.get("N", 0),
            "|E|": r.get("num_edges", 0),
            "status": st,
            "best_ub": ub_str,
            "lb_degree": lb_deg,
            "lb_solver": lb_solver,
            "lb": lb,
            "ub_minus_lb": ub_minus_lb_str,
            "solve_time": solve_t_str,
            "wall_time": wall_t_str,
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

    agg_records = _ensure_aggregated_records(records, strict=True)
    pairs_map = {}
    for r in agg_records:
        if not is_valid_record_dict(r):
            continue
        canon = get_canonical_key(r)
        if canon is None:
            continue
        ds, m, sol, flags, seed = canon
        if seed is None:
            continue
        if m == "cpsat":
            existing = pairs_map.setdefault((ds, seed), {}).get("cpsat")
            if existing is not None:
                pairs_map[(ds, seed)]["cpsat"] = pick_better_record(existing, r, strict=True)
            else:
                pairs_map[(ds, seed)]["cpsat"] = r
        elif m == "sat" and sol == "cadical195":
            if len(flags) >= 3 and flags[0] and flags[1] and flags[2]:
                existing = pairs_map.setdefault((ds, seed), {}).get("sat")
                if existing is not None:
                    pairs_map[(ds, seed)]["sat"] = pick_better_record(existing, r, strict=True)
                else:
                    pairs_map[(ds, seed)]["sat"] = r

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
                    t_cp = extract_repetition_solve_time(rec_cp)
                    t_sat = extract_repetition_solve_time(rec_sat)
                    if (
                        t_cp is not None and math.isfinite(t_cp) and t_cp > 0
                        and t_sat is not None and math.isfinite(t_sat) and t_sat > 0
                    ):
                        ratio = t_cp / t_sat
                        if math.isfinite(ratio) and ratio > 0:
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

    agg_records = _ensure_aggregated_records(records, strict=True)
    for r in agg_records:
        if not is_valid_record_dict(r):
            continue
        canon = get_canonical_key(r)
        if canon is None:
            continue
        ds, m, sol, flags, seed = canon
        if seed is not None:
            continue
        if ds not in target_datasets:
            continue
        meta[ds] = (r.get("N", 0), r.get("num_edges", 0))

        if m == "cpsat":
            if ds in cpsat_recs:
                cpsat_recs[ds] = pick_better_record(cpsat_recs[ds], r, strict=True)
            else:
                cpsat_recs[ds] = r
        elif m == "sat" and sol == "cadical195":
            if len(flags) >= 3 and flags[0] and flags[1] and flags[2]:
                if ds in sat_recs:
                    sat_recs[ds] = pick_better_record(sat_recs[ds], r, strict=True)
                else:
                    sat_recs[ds] = r

    if not cpsat_recs and not sat_recs:
        return

    rows = []
    for ds in target_datasets:
        n, e = meta.get(ds, (0, 0))
        lb_deg = get_lb_degree(ds)

        rec_sat = sat_recs.get(ds, {})
        sat_st = normalize_status(rec_sat.get("status")) or (str(rec_sat.get("status")) if rec_sat.get("status") is not None else "-")
        sat_ub_val = validate_best_ub(rec_sat.get("best_ub"))
        sat_ub = sat_ub_val if sat_ub_val is not None else "-"
        sat_t = validate_solve_time(rec_sat.get("solve_time"))
        sat_time = f"{sat_t:.4f}" if sat_t is not None else "-"

        rec_cp = cpsat_recs.get(ds, {})
        cp_st = normalize_status(rec_cp.get("status")) or (str(rec_cp.get("status")) if rec_cp.get("status") is not None else "-")
        cp_ub_val = validate_best_ub(rec_cp.get("best_ub"))
        cp_ub = cp_ub_val if cp_ub_val is not None else "-"
        raw_cp_lb = rec_cp.get("best_lb")
        val_cp_lb = validate_best_ub(raw_cp_lb)
        cpsat_solver_lb = val_cp_lb if val_cp_lb is not None else "-"
        if val_cp_lb is not None:
            lb_combined = max(lb_deg, val_cp_lb)
        else:
            lb_combined = lb_deg if (cp_st != "-" or sat_st != "-") else "-"
        cp_t = validate_solve_time(rec_cp.get("solve_time"))
        cp_time = f"{cp_t:.4f}" if cp_t is not None else "-"

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

    agg_records = _ensure_aggregated_records(records, strict=True)
    for r in agg_records:
        if not is_valid_record_dict(r):
            continue
        canon = get_canonical_key(r)
        if canon is None:
            continue
        ds, m, sol, flags, seed = canon
        if ds not in target_datasets:
            continue
        if seed not in seeds:
            continue

        if m == "cpsat":
            if (ds, seed) in cpsat_map:
                cpsat_map[(ds, seed)] = pick_better_record(cpsat_map[(ds, seed)], r, strict=True)
            else:
                cpsat_map[(ds, seed)] = r
        elif m == "sat" and sol == "cadical195":
            if len(flags) >= 3 and flags[0] and flags[1] and flags[2]:
                if (ds, seed) in sat_map:
                    sat_map[(ds, seed)] = pick_better_record(sat_map[(ds, seed)], r, strict=True)
                else:
                    sat_map[(ds, seed)] = r

    if not sat_map and not cpsat_map:
        print("[WARNING] Skipping table_methods_large_seeds: no records found.")
        return None

    rows = []
    for ds in target_datasets:
        for seed in seeds:
            rec_sat = sat_map.get((ds, seed), {})
            sat_st = normalize_status(rec_sat.get("status")) or (str(rec_sat.get("status")) if rec_sat.get("status") is not None else "-")
            sat_ub_val = validate_best_ub(rec_sat.get("best_ub"))
            sat_ub = sat_ub_val if sat_ub_val is not None else "-"
            sat_t = validate_solve_time(rec_sat.get("solve_time"))
            sat_time = f"{sat_t:.4f}" if sat_t is not None else "-"

            rec_cp = cpsat_map.get((ds, seed), {})
            cp_st = normalize_status(rec_cp.get("status")) or (str(rec_cp.get("status")) if rec_cp.get("status") is not None else "-")
            cp_ub_val = validate_best_ub(rec_cp.get("best_ub"))
            cp_ub = cp_ub_val if cp_ub_val is not None else "-"
            cp_t = validate_solve_time(rec_cp.get("solve_time"))
            cp_time = f"{cp_t:.4f}" if cp_t is not None else "-"

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
