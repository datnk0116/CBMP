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
from ortools.sat.python import cp_model
from cbmp import SolveResult, compute_cyclic_bandwidth

def build_cpsat_model(N: int, edges: list, sb_rotation: bool = True):
    """
    Build and return the CP-SAT model for CBMP.
    Returns: (model, f, k, diff_vars, b_vars)
    """
    model = cp_model.CpModel()

    # Variables f[u] in [1, N]
    f = [model.NewIntVar(1, N, f"f_{u}") for u in range(N)]
    model.AddAllDifferent(f)

    # Rotation symmetry breaking: pin vertex 0 to position 1
    if sb_rotation and N >= 1:
        model.Add(f[0] == 1)

    # Objective k in [0, N // 2]
    k = model.NewIntVar(0, N // 2, "k")

    diff_vars = {}
    b_vars = {}
    for u, v in edges:
        diff = model.NewIntVar(1, N - 1, f"diff_{u}_{v}")
        model.AddAbsEquality(diff, f[u] - f[v])
        b = model.NewBoolVar(f"b_{u}_{v}")

        # if b: diff <= k; if not b: diff >= N - k
        model.Add(diff <= k).OnlyEnforceIf(b)
        model.Add(diff >= N - k).OnlyEnforceIf(b.Not())
        diff_vars[(u, v)] = diff
        b_vars[(u, v)] = b

    model.Minimize(k)
    return model, f, k, diff_vars, b_vars

def solve_cpsat(
    dataset: Optional[Union[str, int]] = None,
    N: Optional[int] = None,
    edges: Optional[List[Tuple[int, int]]] = None,
    timeout: float = 600.0,
    num_workers: int = 1,
    relabel_seed: Optional[int] = None,
    verbose: bool = True,
    sb_rotation: bool = True,
) -> SolveResult:
    """
    Solve Cyclic Bandwidth Minimization Problem using Google OR-Tools CP-SAT.

    Supports positional and keyword calls:
      - solve_cpsat(dataset="30_bus")
      - solve_cpsat("30_bus")
      - solve_cpsat(N=6, edges=[(0, 1), (1, 2)])
      - solve_cpsat(6, [(0, 1), (1, 2)])

    Parameters:
      dataset: Name of dataset or path to .mtx file.
      N: Number of vertices (if providing graph directly).
      edges: List of undirected edges (u, v).
      timeout: Time limit in seconds.
      num_workers: Number of search worker threads.
      relabel_seed: Seed for random vertex permutation.
      verbose: Print solver status and optimum.
      sb_rotation: Pin vertex 0 label to 1 to break rotational symmetry.

    Returns:
      SolveResult containing optimal cyclic bandwidth and stats dictionary.
    """
    t_start = time.perf_counter()

    if isinstance(dataset, int) and not isinstance(dataset, bool):
        # Positional call: solve_cpsat(N, edges, ...) or solve_cpsat(6, edges=[...])
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
        raise ValueError("Either (N, edges) or dataset must be provided to solve_cpsat.")

    if relabel_seed is not None:
        rng = np.random.default_rng(relabel_seed)
        perm = rng.permutation(N)
        edges = sorted(list(set((min(int(perm[u]), int(perm[v])), max(int(perm[u]), int(perm[v]))) for u, v in edges if u != v)))

    stats = {
        "dataset": dataset_name,
        "N": N,
        "num_edges": len(edges),
        "relabel_seed": relabel_seed,
        "config": {
            "time_limit": timeout,
            "num_workers": num_workers,
            "relabel_seed": relabel_seed,
            "sb_rotation": sb_rotation,
        },
        "solver": "cp_sat",
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

    # ============ BUILD MODEL
    t_enc_start = time.perf_counter()
    model, f, k, diff_vars, b_vars = build_cpsat_model(N, edges, sb_rotation=sb_rotation)
    t_enc_end = time.perf_counter()
    encode_time = t_enc_end - t_enc_start

    num_vars = len(model.proto.variables)
    num_constraints = len(model.proto.constraints)

    stats["encode_time"] = encode_time
    stats["num_vars"] = num_vars
    stats["num_clauses"] = num_constraints

    # ============ SOLVE
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = int(num_workers)
    if timeout is not None and timeout > 0:
        solver.parameters.max_time_in_seconds = float(timeout)

    t_sol_start = time.perf_counter()
    cp_status = solver.Solve(model)
    t_sol_end = time.perf_counter()
    solve_time = t_sol_end - t_sol_start

    stats["solve_time"] = solve_time
    stats["per_iter_times"] = [solve_time]
    stats["num_iterations"] = 1

    best_lb = None
    try:
        raw_lb = solver.BestObjectiveBound()
        if raw_lb is not None and not math.isinf(raw_lb) and not math.isnan(raw_lb):
            best_lb = int(math.ceil(raw_lb))
    except Exception:
        best_lb = None

    if cp_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        reported_bw = int(round(solver.ObjectiveValue()))
        try:
            assignment = [int(solver.Value(f[u])) - 1 for u in range(N)]
            valid, actual_bw, err_msg = validate_solution(
                N, edges, assignment, reported_objective=reported_bw if cp_status == cp_model.OPTIMAL else None
            )
        except Exception as e:
            valid, actual_bw, err_msg = False, None, str(e)

        if valid and actual_bw is not None:
            best_ub = actual_bw
            stats["assignment"] = assignment
            if cp_status == cp_model.OPTIMAL:
                status = "OPTIMAL"
                if best_lb is None:
                    best_lb = best_ub
            else:
                status = "FEASIBLE"
        else:
            best_ub = -1
            stats["assignment"] = None
            status = "ERROR"
            stats["validation_error"] = err_msg

        if status == "OPTIMAL" and verbose:
            print("minimum bandwidth:", best_ub)
        elif status == "FEASIBLE" and verbose:
            print("feasible solution found, bandwidth:", best_ub)
    elif cp_status == cp_model.UNKNOWN:
        best_ub = -1
        stats["assignment"] = None
        status = "TIMEOUT"
        if verbose:
            print("timeout, no valid assignment was found")
    elif cp_status == cp_model.INFEASIBLE:
        best_ub = -1
        stats["assignment"] = None
        status = "INFEASIBLE"
        if verbose:
            print("model proved infeasible")
    else:
        best_ub = -1
        stats["assignment"] = None
        status = "ERROR"
        if verbose:
            print(f"CP-SAT finished with status: {solver.StatusName(cp_status)}")

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
    parser = argparse.ArgumentParser(description="Solve CBMP using OR-Tools CP-SAT baseline")
    parser.add_argument("dataset", help="name of .mtx file in datasets/ or direct path")
    parser.add_argument("--timeout", type=float, default=600.0, help="time limit in seconds")
    parser.add_argument("--num-workers", type=int, default=1, help="number of worker threads (default: 1)")
    parser.add_argument("--relabel-seed", type=int, default=None, help="seed for random vertex relabeling")
    parser.add_argument("--sb-rotation", action=argparse.BooleanOptionalAction, default=True, help="enable rotation symmetry breaking (default: True)")
    parser.add_argument("--json-out", type=str, default=None, help="path to append statistics as JSON line")
    args = parser.parse_args()

    res = solve_cpsat(
        dataset=args.dataset,
        timeout=args.timeout,
        num_workers=args.num_workers,
        relabel_seed=args.relabel_seed,
        verbose=True,
        sb_rotation=args.sb_rotation,
    )

    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(res.stats) + "\n")

