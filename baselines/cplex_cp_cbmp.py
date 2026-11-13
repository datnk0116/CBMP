"""
IBM CPLEX CP Optimizer (Constraint Programming) Baseline for CBMP.
Uses docplex.cp to model and solve the Cyclic Bandwidth Minimization Problem.
"""

import os
import sys
import time
import json
import math
import argparse
from pathlib import Path
from typing import Optional, Union, List, Tuple, Any, Dict
import numpy as np

# Ensure repo root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_utils import (
    resolve_dataset_path,
    validate_and_normalize_graph,
    load_graph_from_mtx,
    validate_solution,
    compute_cyclic_bandwidth,
)
from cbmp import SolveResult

def is_cplex_cp_available() -> Tuple[bool, str]:
    """Check if docplex.cp and underlying cpoptimizer executable are available."""
    try:
        import docplex.cp
        import shutil
        from docplex.cp.config import context
        execfile = getattr(getattr(getattr(context, "solver", None), "local", None), "execfile", "cpoptimizer")
        if not shutil.which(execfile) and not os.path.isfile(execfile):
            return False, f"CPLEX CP Optimizer engine ('{execfile}') is not found in system PATH"
        from docplex.cp.model import CpoModel
        mdl = CpoModel()
        x = mdl.integer_var(0, 1)
        mdl.add(x == 1)
        res = mdl.solve(TimeLimit=1, LogVerbosity="Quiet")
        return True, "CPLEX CP Optimizer is available"
    except Exception as e:
        return False, f"CPLEX CP Optimizer unavailable: {e}"

def build_cplex_cp_model(N: int, edges: list, sb_rotation: bool = True):
    """
    Build docplex.cp model for CBMP.
    Returns: (mdl, f, k)
    """
    from docplex.cp.model import CpoModel
    mdl = CpoModel(name="cbmp_cplex_cp")

    # Vertex labels f[u] in {1, ..., N}
    f = [mdl.integer_var(1, N, name=f"f_{u}") for u in range(N)]
    mdl.add(mdl.all_diff(f))

    # Rotation symmetry breaking: pin vertex 0 to position 1
    if sb_rotation and N >= 1:
        mdl.add(f[0] == 1)

    # Objective k in [0, N // 2]
    k = mdl.integer_var(0, N // 2, name="k")

    for u, v in edges:
        diff = mdl.abs(f[u] - f[v])
        cyc_d = mdl.min(diff, N - diff)
        mdl.add(cyc_d <= k)

    mdl.minimize(k)
    return mdl, f, k

def _validate_constraint_count(val):
    """
    Accepts only genuine non-negative integer constraint counts.
    Rejects booleans, negative numbers, NaN, infinity, malformed strings, and invalid values.
    """
    if val is None or isinstance(val, bool):
        return None
    try:
        if isinstance(val, (int, float, str)):
            f_val = float(val)
            if math.isnan(f_val) or math.isinf(f_val):
                return None
            if f_val < 0:
                return None
            if f_val.is_integer():
                return int(f_val)
    except (ValueError, TypeError):
        pass
    return None

def solve_cplex_cp(
    dataset: Optional[Union[str, int]] = None,
    N: Optional[int] = None,
    edges: Optional[List[Tuple[int, int]]] = None,
    timeout: float = 600.0,
    verbose: bool = True,
    sb_rotation: bool = True,
) -> SolveResult:
    """
    Solve CBMP using IBM CPLEX CP Optimizer.
    """
    if isinstance(dataset, int) and not isinstance(dataset, bool):
        # Positional call: solve(N, edges, ...) or solve(6, edges=[...])
        if isinstance(N, (list, tuple, set)):
            edges = N
            N = dataset
            dataset = None
        elif isinstance(N, int) and not isinstance(N, bool):
            if N != dataset:
                raise ValueError(f"Conflicting values for N: {dataset} vs {N}")
            N = dataset
            dataset = None
        else:
            if edges is None and N is not None:
                edges = N
            N = dataset
            dataset = None

    if N is not None and edges is not None:
        N, edges = validate_and_normalize_graph(N, edges)
        dataset_name = dataset if isinstance(dataset, str) else "custom"
    elif dataset is not None:
        N, edges, dataset_name = load_graph_from_mtx(dataset)
    else:
        raise ValueError("Either (N, edges) or dataset must be provided to solve_cplex_cp.")

    stats: Dict[str, Any] = {
        "dataset": dataset_name,
        "N": N,
        "num_edges": len(edges),
        "config": {
            "time_limit": timeout,
            "sb_rotation": sb_rotation,
        },
        "solver": "cplex_cp",
        "status": "RUNNING",
        "best_ub": -1,
        "best_lb": None,
        "encode_time": 0.0,
        "solve_time": 0.0,
        "per_iter_times": [],
        "num_iterations": 0,
        "num_vars": 0,
        "num_clauses": 0,
    }

    if not edges:
        stats["status"] = "OPTIMAL"
        stats["best_ub"] = 0
        stats["best_lb"] = 0
        stats["gap"] = 0.0
        stats["assignment"] = list(range(N))
        if verbose:
            print("minimum bandwidth: 0")
        return SolveResult(0, stats)

    available, msg = is_cplex_cp_available()
    if not available:
        status = "LICENSE_ERROR" if "license" in msg.lower() else "UNAVAILABLE"
        stats["status"] = status
        stats["error"] = msg
        stats["assignment"] = None
        if verbose:
            print(f"[{status}] {msg}")
        return SolveResult(-1, stats)

    t_enc_start = time.perf_counter()
    mdl, f, k = build_cplex_cp_model(N, edges, sb_rotation=sb_rotation)
    t_enc_end = time.perf_counter()
    stats["encode_time"] = t_enc_end - t_enc_start
    stats["num_vars"] = N + 1
    # SAT clause count is not applicable to CP Optimizer; report None for num_clauses
    # to avoid mixing CP constraints with SAT clauses in comparative tables.
    stats["num_clauses"] = None
    stats["estimated_constraints"] = len(edges) + 1 + (1 if sb_rotation else 0)
    # Model expressions in Docplex CP contain nested AST subexpressions rather than pure
    # top-level constraints. Do not expose unverified expression counts as exact constraint counts.
    # num_constraints will be populated after solve only if the solver exposes genuine statistics.
    stats["num_constraints"] = None

    t_sol_start = time.perf_counter()
    try:
        sol = mdl.solve(
            TimeLimit=float(timeout) if timeout else None,
            LogVerbosity="Normal" if verbose else "Quiet",
        )
    except Exception as e:
        err_msg = str(e)
        stats["status"] = "LICENSE_ERROR" if "license" in err_msg.lower() else "ERROR"
        stats["error"] = err_msg
        stats["assignment"] = None
        return SolveResult(-1, stats)

    solve_time = time.perf_counter() - t_sol_start
    stats["solve_time"] = solve_time
    stats["per_iter_times"] = [solve_time]
    stats["num_iterations"] = 1

    # Attempt genuine solver-info sources independently
    num_constraints = None
    if sol and hasattr(sol, "get_solver_info"):
        try:
            raw_c = sol.get_solver_info("NumberOfConstraints")
            num_constraints = _validate_constraint_count(raw_c)
        except Exception:
            num_constraints = None

    if num_constraints is None and sol and hasattr(sol, "get_solver_infos"):
        try:
            infos = sol.get_solver_infos()
            raw_c = None
            if hasattr(infos, "get_number_of_constraints"):
                raw_c = infos.get_number_of_constraints()
            elif hasattr(infos, "get"):
                raw_c = infos.get("NumberOfConstraints")
            num_constraints = _validate_constraint_count(raw_c)
        except Exception:
            num_constraints = None

    stats["num_constraints"] = num_constraints

    def _safe_sol_call(obj, method_name, default=None):
        """Safely invokes a solver result method, guarding against missing methods or exceptions."""
        if obj is None:
            return default
        try:
            fn = getattr(obj, method_name, None)
            if callable(fn):
                res = fn()
                return res if res is not None else default
        except Exception:
            return default
        return default

    def _safe_bool_call(obj, method_name) -> bool:
        """
        Safely invokes a boolean solver result method.
        Accepts only genuine boolean types (type(value) is bool).
        Rejects None, strings (even 'true'/'false'), integers (0, 1), and other objects.
        """
        val = _safe_sol_call(obj, method_name, default=None)
        if type(val) is bool:
            return val
        return False

    raw_sol_status = _safe_sol_call(sol, "get_solve_status", default=None)
    raw_stop_cause = _safe_sol_call(sol, "get_stop_cause", default=None)
    raw_fail_status = _safe_sol_call(sol, "get_fail_status", default=None)
    raw_search_status = _safe_sol_call(sol, "get_search_status", default=None)
    has_solution = _safe_bool_call(sol, "is_solution")

    sol_status = raw_sol_status.strip() if isinstance(raw_sol_status, str) else ""
    stop_cause = raw_stop_cause.strip() if isinstance(raw_stop_cause, str) else ""
    fail_status = raw_fail_status.strip() if isinstance(raw_fail_status, str) else ""
    search_status = raw_search_status.strip() if isinstance(raw_search_status, str) else ""

    # Preserve raw solver status fields for auditability
    stats["solve_status"] = str(raw_sol_status) if raw_sol_status is not None else ""
    stats["stop_cause"] = str(raw_stop_cause) if raw_stop_cause is not None else ""
    stats["fail_status"] = str(raw_fail_status) if raw_fail_status is not None else ""
    stats["search_status"] = str(raw_search_status) if raw_search_status is not None else ""

    # Extract lower bound if exposed
    best_lb = None
    try:
        if sol and hasattr(sol, "get_objective_bound"):
            raw_lb = sol.get_objective_bound()
            if raw_lb is not None and not math.isinf(raw_lb) and not math.isnan(raw_lb):
                best_lb = int(math.ceil(raw_lb))
        elif sol and hasattr(sol, "get_objective_bounds"):
            raw_bounds = sol.get_objective_bounds()
            if raw_bounds and raw_bounds[0] is not None:
                b = raw_bounds[0]
                if not math.isinf(b) and not math.isnan(b):
                    best_lb = int(math.ceil(b))
    except Exception:
        best_lb = None

    valid_solution = False
    actual_bw = None
    assignment = None

    if has_solution:
        try:
            assignment = [int(sol[f[u]]) - 1 for u in range(N)]
            reported_bw = int(round(sol[k]))
            valid, actual_bw, err_msg = validate_solution(
                N, edges, assignment, reported_objective=reported_bw
            )
            if valid and actual_bw is not None:
                valid_solution = True
            else:
                stats["validation_error"] = err_msg
                assignment = None
        except Exception as e:
            valid_solution = False
            stats["validation_error"] = str(e)
            assignment = None

    # Determine termination cause categories
    is_solution_optimal = _safe_bool_call(sol, "is_solution_optimal")
    is_optimal = (sol_status.lower() == "optimal") or is_solution_optimal

    clean_stop_cause = stop_cause.lower().replace(" ", "").replace("_", "")
    clean_sol_status = sol_status.lower().replace(" ", "").replace("_", "")
    clean_fail_status = fail_status.lower().replace(" ", "").replace("_", "")

    # Explicit evidence of time limit from stop cause, status, solver infos, or solver log
    is_timelimit = (
        "timelimit" in clean_stop_cause
        or "timelimit" in clean_sol_status
        or "timelimit" in clean_fail_status
    )
    if not is_timelimit and sol:
        try:
            if hasattr(sol, "get_solver_info"):
                for k in ("SearchStopCause", "EffectiveSearchStopCause", "TimeLimit", "StopCause"):
                    val = str(sol.get_solver_info(k, "") or "").lower()
                    if "timelimit" in val:
                        is_timelimit = True
                        break
        except Exception:
            pass
    if not is_timelimit and sol:
        try:
            if hasattr(sol, "get_solver_log"):
                log_txt = str(sol.get_solver_log() or "").lower()
                if (
                    "time limit reached" in log_txt
                    or "stopped by time limit" in log_txt
                    or "timelimit reached" in log_txt
                    or "! time limit reached" in log_txt
                ):
                    is_timelimit = True
        except Exception:
            pass

    is_abort_stop = (
        "abort" in clean_stop_cause
        or "abort" in clean_fail_status
        or clean_sol_status in ("jobaborted", "aborted")
    )
    is_fail_stop = (
        clean_sol_status in ("jobfailed", "failed")
        or "exception" in clean_fail_status
    )
    is_infeasible = clean_sol_status in ("infeasible",)

    if is_optimal:
        if valid_solution:
            status = "OPTIMAL"
            best_ub = actual_bw
            stats["assignment"] = assignment
        else:
            status = "ERROR"
            best_ub = -1
            stats["assignment"] = None
    elif valid_solution:
        # Valid incumbent found without optimality proof
        best_ub = actual_bw
        stats["assignment"] = assignment
        if is_timelimit:
            status = "TIMEOUT"
        elif is_abort_stop:
            status = "INTERRUPTED"
        else:
            # Stopped by node limit, iteration limit, or other search limit with valid incumbent preserved
            status = "FEASIBLE"
    elif has_solution and not valid_solution:
        # Solution claimed but failed validation
        status = "ERROR"
        best_ub = -1
        stats["assignment"] = None
    else:
        # No solution found
        best_ub = -1
        stats["assignment"] = None
        if is_infeasible:
            status = "INFEASIBLE"
        elif is_timelimit:
            status = "TIMEOUT"
        elif is_abort_stop:
            status = "INTERRUPTED"
        else:
            # Generic search limits without solution and without time-limit evidence,
            # or unclassified/failed/unknown termination -> ERROR
            status = "ERROR"
            stats["error"] = f"CP Optimizer terminated with status '{sol_status}', stop cause '{stop_cause}' without solution"

    if status == "OPTIMAL" and best_lb is None:
        best_lb = best_ub

    gap = None
    if best_ub >= 0 and best_lb is not None and best_lb >= 0:
        if best_ub == 0:
            gap = 0.0
        else:
            gap = float(best_ub - best_lb) / float(best_ub)

    stats["gap"] = gap
    stats["best_ub"] = best_ub
    stats["best_lb"] = best_lb
    stats["status"] = status
    return SolveResult(best_ub, stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve CBMP using IBM CPLEX CP Optimizer baseline")
    parser.add_argument("dataset", help="name of .mtx file in datasets/ or direct path")
    parser.add_argument("--timeout", type=float, default=600.0, help="time limit in seconds")
    parser.add_argument("--sb-rotation", action=argparse.BooleanOptionalAction, default=True, help="enable rotation symmetry breaking (default: True)")
    parser.add_argument("--json-out", type=str, default=None, help="path to append statistics as JSON line")
    args = parser.parse_args()

    res = solve_cplex_cp(
        dataset=args.dataset,
        timeout=args.timeout,
        verbose=True,
        sb_rotation=args.sb_rotation,
    )

    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(res.stats) + "\n")
