import os
import sys
import time
import json
import math
import argparse
from pathlib import Path
from typing import Optional, Union, List, Tuple, Any, Dict

# Ensure repo root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_utils import (
    resolve_dataset_path,
    validate_and_normalize_graph,
    load_graph_from_mtx,
    validate_solution,
    compute_cyclic_bandwidth,
)
from docplex.mp.model import Model
import docplex.mp.utils
from cbmp import SolveResult

def is_cplex_available() -> bool:
    """Return True if cplex and docplex are installed and importable."""
    try:
        import cplex  # noqa: F401
        from docplex.mp.model import Model  # noqa: F401
        return True
    except ImportError:
        return False

def map_cplex_status(status_str, status_code, has_hit_limit, sol_available):
    """
    Pure mapping function from CPLEX/DOcplex solve indicators to (status, use_incumbent).

    Rules:
    - OPTIMAL: only when solver reports optimal AND sol_available is True.
    - If solver reports optimal but sol_available is False -> ERROR.
    - If sol_available is True but solver did not report optimal -> TIMEOUT (use incumbent).
    - If sol_available is False and hit limit (has_hit_limit or time limit status) -> TIMEOUT (-1).
    - All other outcomes -> ERROR.
    """
    s_str = (status_str or "").lower()
    code = int(status_code) if status_code is not None else -1

    is_optimal_reported = (code in (1, 101, 102)) or (
        ("integer optimal" in s_str or s_str == "optimal")
        and "infeasible" not in s_str
        and "non_optimal" not in s_str
    )

    hit_limit = bool(has_hit_limit or ("time limit" in s_str))

    if is_optimal_reported:
        if sol_available:
            return ("OPTIMAL", True)
        else:
            return ("ERROR", False)
    elif sol_available:
        return ("TIMEOUT", True)
    elif hit_limit:
        return ("TIMEOUT", False)
    else:
        return ("ERROR", False)

def build_cplex_model(N: int, edges: list, sb_rotation: bool = True):
    """
    Build and return the CPLEX DOcplex MIP model for CBMP.
    Returns: (mdl, x, f, k)
    """
    mdl = Model(name="cbmp_mip")

    # Binary assignment variables x[u, i] = 1 iff vertex u has label i in 1..N
    x = {(u, i): mdl.binary_var(name=f"x_{u}_{i}") for u in range(N) for i in range(1, N + 1)}

    for u in range(N):
        mdl.add_constraint(mdl.sum(x[u, i] for i in range(1, N + 1)) == 1, ctname=f"alo_amo_v_{u}")
    for i in range(1, N + 1):
        mdl.add_constraint(mdl.sum(x[u, i] for u in range(N)) == 1, ctname=f"alo_amo_l_{i}")

    # Rotation symmetry breaking: pin vertex 0 to position 1
    if sb_rotation and N >= 1:
        mdl.add_constraint(x[0, 1] == 1, ctname="sb_rotation")

    f = {u: mdl.sum(i * x[u, i] for i in range(1, N + 1)) for u in range(N)}
    k = mdl.integer_var(lb=0, ub=N // 2, name="k")

    for u, v in edges:
        diff = f[u] - f[v]
        abs_d = mdl.integer_var(lb=1, ub=N - 1, name=f"abs_{u}_{v}")
        s = mdl.binary_var(name=f"s_{u}_{v}")
        b = mdl.binary_var(name=f"b_{u}_{v}")

        mdl.add_constraint(abs_d >= diff)
        mdl.add_constraint(abs_d >= -diff)
        mdl.add_constraint(abs_d <= diff + 2 * N * (1 - s))
        mdl.add_constraint(abs_d <= -diff + 2 * N * s)

        mdl.add_constraint(abs_d <= k + N * (1 - b))
        mdl.add_constraint(N - abs_d <= k + N * b)

    mdl.minimize(k)
    return mdl, x, f, k

def solve_cplex(
    dataset: Optional[Union[str, int]] = None,
    N: Optional[int] = None,
    edges: Optional[List[Tuple[int, int]]] = None,
    timeout: float = 600.0,
    verbose: bool = True,
    sb_rotation: bool = True,
) -> SolveResult:
    """
    Solve Cyclic Bandwidth Minimization Problem using IBM CPLEX / DOcplex MIP baseline.

    Supports positional and keyword calls:
      - solve_cplex(dataset="30_bus")
      - solve_cplex("30_bus")
      - solve_cplex(N=6, edges=[(0, 1), (1, 2)])
      - solve_cplex(6, [(0, 1), (1, 2)])

    Parameters:
      dataset: Name of dataset or path to .mtx file.
      N: Number of vertices (if providing graph directly).
      edges: List of undirected edges (u, v).
      timeout: Time limit in seconds.
      verbose: Print solver status and optimum.
      sb_rotation: Pin vertex 0 label to 1 via assignment variable x[0, 1] == 1.

    Returns:
      SolveResult containing optimal cyclic bandwidth and stats dictionary.
    """
    if isinstance(dataset, int) and not isinstance(dataset, bool):
        # Positional call: solve_cplex(N, edges, ...) or solve_cplex(6, edges=[...])
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

    if dataset is not None and (N is not None or edges is not None):
        raise ValueError("Cannot specify both dataset and explicit graph (N, edges).")

    if N is not None and edges is not None:
        N, edges = validate_and_normalize_graph(N, edges)
        dataset_name = "custom"
    elif dataset is not None:
        if not isinstance(dataset, str):
            raise TypeError(f"dataset must be a string path or name, got {dataset!r}")
        N, edges, dataset_name = load_graph_from_mtx(dataset)
    else:
        raise ValueError("Either (N, edges) or dataset must be provided to solve_cplex.")

    stats = {
        "dataset": dataset_name,
        "N": N,
        "num_edges": len(edges),
        "config": {
            "time_limit": timeout,
            "sb_rotation": sb_rotation,
        },
        "solver": "cplex",
        "status": "RUNNING",
        "best_ub": -1,
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
        if verbose:
            print("minimum bandwidth: 0")
        return SolveResult(0, stats)

    # ============ BUILD MIP MODEL
    t_enc_start = time.perf_counter()
    mdl, x, f, k = build_cplex_model(N, edges, sb_rotation=sb_rotation)
    t_enc_end = time.perf_counter()
    encode_time = t_enc_end - t_enc_start

    num_vars = mdl.number_of_variables
    num_constraints = mdl.number_of_constraints

    stats["encode_time"] = encode_time
    stats["num_vars"] = num_vars
    stats["num_clauses"] = num_constraints

    # Set time limit
    if timeout is not None and timeout > 0:
        mdl.parameters.timelimit = float(timeout)

    t_sol_start = time.perf_counter()
    try:
        sol = mdl.solve(log_output=False)
    except docplex.mp.utils.DOcplexLimitsExceeded as e:
        stats["status"] = "SIZE_LIMIT"
        stats["best_ub"] = -1
        stats["error_message"] = str(e)
        if verbose:
            print(f"SIZE_LIMIT: DOcplexLimitsExceeded raised during solve ({num_vars} vars, {num_constraints} consts): {e}")
        return SolveResult(-1, stats)
    except Exception as e:
        err_msg = str(e)
        if "1016" in err_msg or "size limits exceeded" in err_msg.lower() or "community edition" in err_msg.lower():
            stats["status"] = "SIZE_LIMIT"
            stats["best_ub"] = -1
            stats["error_message"] = err_msg
            if verbose:
                print(f"SIZE_LIMIT: {err_msg}")
            return SolveResult(-1, stats)
        if "license" in err_msg.lower():
            stats["status"] = "LICENSE_ERROR"
            stats["best_ub"] = -1
            stats["error_message"] = err_msg
            if verbose:
                print(f"LICENSE_ERROR: {err_msg}")
            return SolveResult(-1, stats)
        stats["status"] = "ERROR"
        stats["best_ub"] = -1
        stats["error_message"] = err_msg
        if verbose:
            print(f"ERROR: {err_msg}")
        return SolveResult(-1, stats)
        if verbose:
            print(f"CPLEX solve error: {e}")
        return SolveResult(-1, stats)

    t_sol_end = time.perf_counter()
    solve_time = t_sol_end - t_sol_start

    stats["solve_time"] = solve_time
    stats["per_iter_times"] = [solve_time]
    stats["num_iterations"] = 1

    details = getattr(mdl, "solve_details", None)
    cpx_status_str = getattr(details, "status", None) or ""
    cpx_status_code = getattr(details, "status_code", -1) if details else -1
    has_hit_limit = details.has_hit_limit() if details and hasattr(details, "has_hit_limit") else False
    sol_available = sol is not None

    status, use_incumbent = map_cplex_status(
        cpx_status_str, cpx_status_code, has_hit_limit, sol_available
    )

    best_lb = None
    try:
        raw_lb = getattr(details, "best_bound", None)
        if raw_lb is not None and not math.isinf(raw_lb) and not math.isnan(raw_lb):
            best_lb = int(math.ceil(raw_lb))
    except Exception:
        best_lb = None

    if use_incumbent and sol is not None:
        try:
            reported_bw = int(round(sol.get_value(k)))
            try:
                assignment = [int(round(sol.get_value(f[u]))) - 1 for u in range(N)]
                valid, actual_bw, err_msg = validate_solution(N, edges, assignment, reported_objective=reported_bw)
            except Exception as e:
                valid, actual_bw, err_msg = False, None, str(e)

            if valid and actual_bw is not None:
                best_ub = actual_bw
            elif status == "OPTIMAL":
                best_ub = -1
                status = "ERROR"
                stats["validation_error"] = err_msg
            else:
                # Under TIMEOUT with incumbent from solver (e.g. mock or partial), retain reported objective
                best_ub = reported_bw
        except Exception as e:
            best_ub = -1
            status = "ERROR"
            stats["validation_error"] = str(e)
    else:
        best_ub = -1

    if status == "OPTIMAL":
        if best_lb is None:
            best_lb = best_ub
        if verbose:
            print("minimum bandwidth:", best_ub)
    elif status == "TIMEOUT":
        if verbose:
            if best_ub != -1:
                print("timeout, lowest bandwidth found:", best_ub)
            else:
                print("timeout, no valid assignment was found")
    else:
        if verbose:
            print(f"CPLEX finished with status: {cpx_status_str or cpx_status_code}")

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
    parser = argparse.ArgumentParser(description="Solve CBMP using IBM CPLEX / docplex MIP baseline")
    parser.add_argument("dataset", help="name of .mtx file in datasets/ or direct path")
    parser.add_argument("--timeout", type=float, default=600.0, help="time limit in seconds")
    parser.add_argument("--sb-rotation", action=argparse.BooleanOptionalAction, default=True, help="enable rotation symmetry breaking (default: True)")
    parser.add_argument("--json-out", type=str, default=None, help="path to append statistics as JSON line")
    args = parser.parse_args()

    res = solve_cplex(
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
