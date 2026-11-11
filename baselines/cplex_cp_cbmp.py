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

from graph_utils import resolve_dataset_path, validate_and_normalize_graph, load_graph_from_mtx
from cbmp import SolveResult, compute_cyclic_bandwidth

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
        if verbose:
            print("minimum bandwidth: 0")
        return SolveResult(0, stats)

    available, msg = is_cplex_cp_available()
    if not available:
        stats["status"] = "UNAVAILABLE"
        stats["error"] = msg
        if verbose:
            print(f"[UNAVAILABLE] {msg}")
        return SolveResult(-1, stats)

    t_enc_start = time.perf_counter()
    mdl, f, k = build_cplex_cp_model(N, edges, sb_rotation=sb_rotation)
    t_enc_end = time.perf_counter()
    stats["encode_time"] = t_enc_end - t_enc_start
    stats["num_vars"] = N + 1
    stats["num_clauses"] = len(edges) + 1 + (1 if sb_rotation else 0)

    t_sol_start = time.perf_counter()
    try:
        sol = mdl.solve(
            TimeLimit=float(timeout) if timeout else None,
            LogVerbosity="Normal" if verbose else "Quiet",
        )
    except Exception as e:
        stats["status"] = "ERROR"
        stats["error"] = str(e)
        return SolveResult(-1, stats)

    solve_time = time.perf_counter() - t_sol_start
    stats["solve_time"] = solve_time
    stats["per_iter_times"] = [solve_time]
    stats["num_iterations"] = 1

    if sol is not None and sol.is_solution():
        try:
            assignment = [int(sol[f[u]]) - 1 for u in range(N)]
            is_valid_perm = len(assignment) == N and sorted(assignment) == list(range(N))
            if is_valid_perm:
                actual_bw = compute_cyclic_bandwidth(N, edges, assignment)
                reported_bw = int(round(sol[k]))
                best_ub = actual_bw
                status = "OPTIMAL" if sol.get_solve_status() == "Optimal" else "TIMEOUT"
            else:
                best_ub = -1
                status = "ERROR"
        except Exception:
            best_ub = -1
            status = "ERROR"
    else:
        best_ub = -1
        status = "TIMEOUT"

    stats["best_ub"] = best_ub
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
