"""
Gurobi MIP Baseline for CBMP.
Uses gurobipy to model and solve the Cyclic Bandwidth Minimization Problem.
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

def is_gurobi_available() -> Tuple[bool, str]:
    """Check if gurobipy is installed and licensed."""
    try:
        import gurobipy as gp
        # Test license creation
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
        return True, "Gurobi solver is available"
    except ImportError:
        return False, "Gurobi solver is not installed ('gurobipy' module not found)"
    except Exception as e:
        return False, f"Gurobi solver unavailable (license error): {e}"

def build_gurobi_model(N: int, edges: list, sb_rotation: bool = True):
    """
    Build Gurobi MIP model for CBMP.
    Returns: (model, x, f, k)
    Raises:
        ImportError: if gurobipy is not installed or cannot be imported.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as e:
        raise ImportError(f"Gurobi solver is not installed ('gurobipy' module not found): {e}") from e

    model = gp.Model("cbmp_gurobi")

    # Binary assignment variables x[u, i] in {0, 1}
    x = {}
    for u in range(N):
        for i in range(1, N + 1):
            x[u, i] = model.addVar(vtype=GRB.BINARY, name=f"x_{u}_{i}")

    # Position variables f[u] = sum(i * x[u, i])
    f = {}
    for u in range(N):
        f[u] = model.addVar(lb=1, ub=N, vtype=GRB.INTEGER, name=f"f_{u}")
        model.addConstr(f[u] == gp.quicksum(i * x[u, i] for i in range(1, N + 1)))

    # Exactly one position per vertex
    for u in range(N):
        model.addConstr(gp.quicksum(x[u, i] for i in range(1, N + 1)) == 1)

    # Exactly one vertex per position
    for i in range(1, N + 1):
        model.addConstr(gp.quicksum(x[u, i] for u in range(N)) == 1)

    # Rotation symmetry breaking: pin vertex 0 to position 1
    if sb_rotation and N >= 1:
        model.addConstr(x[0, 1] == 1)

    # Objective k in [0, N // 2]
    k = model.addVar(lb=0, ub=N // 2, vtype=GRB.INTEGER, name="k")

    for u, v in edges:
        diff = f[u] - f[v]
        abs_d = model.addVar(lb=1, ub=N - 1, vtype=GRB.INTEGER, name=f"abs_{u}_{v}")
        s = model.addVar(vtype=GRB.BINARY, name=f"s_{u}_{v}")
        b = model.addVar(vtype=GRB.BINARY, name=f"b_{u}_{v}")

        model.addConstr(abs_d >= diff)
        model.addConstr(abs_d >= -diff)
        model.addConstr(abs_d <= diff + 2 * N * (1 - s))
        model.addConstr(abs_d <= -diff + 2 * N * s)

        model.addConstr(abs_d <= k + N * (1 - b))
        model.addConstr(N - abs_d <= k + N * b)

    model.setObjective(k, GRB.MINIMIZE)
    return model, x, f, k

def solve_gurobi(
    dataset: Optional[Union[str, int]] = None,
    N: Optional[int] = None,
    edges: Optional[List[Tuple[int, int]]] = None,
    timeout: float = 600.0,
    verbose: bool = True,
    sb_rotation: bool = True,
) -> SolveResult:
    """
    Solve CBMP using Gurobi MIP baseline.
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
        raise ValueError("Either (N, edges) or dataset must be provided to solve_gurobi.")

    stats: Dict[str, Any] = {
        "dataset": dataset_name,
        "N": N,
        "num_edges": len(edges),
        "config": {
            "time_limit": timeout,
            "sb_rotation": sb_rotation,
        },
        "solver": "gurobi",
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

    available, msg = is_gurobi_available()
    if not available:
        status = "LICENSE_ERROR" if "license" in msg.lower() else "UNAVAILABLE"
        stats["status"] = status
        stats["error"] = msg
        stats["assignment"] = None
        if verbose:
            print(f"[{status}] {msg}")
        return SolveResult(-1, stats)

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as e:
        stats["status"] = "UNAVAILABLE"
        stats["error"] = str(e)
        stats["assignment"] = None
        return SolveResult(-1, stats)

    t_enc_start = time.perf_counter()
    model, x, f, k = build_gurobi_model(N, edges, sb_rotation=sb_rotation)
    t_enc_end = time.perf_counter()

    stats["encode_time"] = t_enc_end - t_enc_start
    stats["num_vars"] = model.NumVars
    stats["num_clauses"] = model.NumConstrs

    if timeout is not None and timeout > 0:
        model.setParam("TimeLimit", float(timeout))
    if not verbose:
        model.setParam("OutputFlag", 0)

    try:
        t_sol_start = time.perf_counter()
        model.optimize()
        solve_time = time.perf_counter() - t_sol_start
    except Exception as e:
        err_msg = str(e)
        stats["status"] = "LICENSE_ERROR" if "license" in err_msg.lower() else "ERROR"
        stats["error"] = err_msg
        stats["assignment"] = None
        return SolveResult(-1, stats)

    stats["solve_time"] = solve_time
    stats["per_iter_times"] = [solve_time]
    stats["num_iterations"] = 1

    best_lb = None
    try:
        raw_lb = model.ObjBound
        if raw_lb is not None and not math.isinf(raw_lb) and not math.isnan(raw_lb):
            best_lb = int(math.ceil(raw_lb))
    except Exception:
        best_lb = None

    sol_count = getattr(model, "SolCount", 0)
    grb_status = getattr(model, "Status", -1)
    stats["grb_status"] = grb_status

    valid_solution = False
    actual_bw = None
    assignment = None

    if sol_count > 0:
        try:
            assignment = [int(round(f[u].X)) - 1 for u in range(N)]
            reported_bw = int(round(k.X))
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

    if grb_status == GRB.OPTIMAL:
        if valid_solution:
            status = "OPTIMAL"
            best_ub = actual_bw
            stats["assignment"] = assignment
            stats["termination_reason"] = "optimal"
        else:
            status = "ERROR"
            best_ub = -1
            stats["assignment"] = None
            stats["termination_reason"] = "validation_error"
    elif valid_solution:
        # Feasible incumbent found, but optimality not proven
        best_ub = actual_bw
        stats["assignment"] = assignment
        if grb_status == GRB.TIME_LIMIT:
            status = "TIMEOUT"
            stats["termination_reason"] = "time_limit"
        elif grb_status == GRB.INTERRUPTED:
            status = "INTERRUPTED"
            stats["termination_reason"] = "interrupted"
        elif grb_status in (GRB.NODE_LIMIT, GRB.ITERATION_LIMIT, GRB.SOLUTION_LIMIT, GRB.SUBOPTIMAL):
            status = "FEASIBLE"
            stats["termination_reason"] = "search_limit"
        else:
            status = "FEASIBLE"
            stats["termination_reason"] = "feasible_incumbent"
    elif sol_count > 0 and not valid_solution:
        # Candidate solution rejected by independent validation or objective mismatch
        status = "ERROR"
        best_ub = -1
        stats["assignment"] = None
        stats["termination_reason"] = "validation_error"
    else:
        # No solution (sol_count == 0)
        best_ub = -1
        stats["assignment"] = None
        if grb_status == GRB.TIME_LIMIT:
            status = "TIMEOUT"
            stats["termination_reason"] = "time_limit"
        elif grb_status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            status = "INFEASIBLE"
            stats["termination_reason"] = "infeasible"
        elif grb_status == GRB.INTERRUPTED:
            status = "INTERRUPTED"
            stats["termination_reason"] = "interrupted"
        elif grb_status == GRB.NUMERIC:
            status = "NUMERIC_ERROR"
            stats["termination_reason"] = "numeric_error"
        elif grb_status in (GRB.NODE_LIMIT, GRB.ITERATION_LIMIT, GRB.SOLUTION_LIMIT):
            status = "ERROR"
            stats["termination_reason"] = "search_limit"
            stats["error"] = f"Gurobi reached search limit (status {grb_status}) without finding a feasible solution"
        else:
            status = "ERROR"
            stats["termination_reason"] = "unknown_error"
            stats["error"] = f"Gurobi terminated with unclassified status {grb_status} without solution"

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
    parser = argparse.ArgumentParser(description="Solve CBMP using Gurobi MIP baseline")
    parser.add_argument("dataset", help="name of .mtx file in datasets/ or direct path")
    parser.add_argument("--timeout", type=float, default=600.0, help="time limit in seconds")
    parser.add_argument("--sb-rotation", action=argparse.BooleanOptionalAction, default=True, help="enable rotation symmetry breaking (default: True)")
    parser.add_argument("--json-out", type=str, default=None, help="path to append statistics as JSON line")
    args = parser.parse_args()

    res = solve_gurobi(
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

