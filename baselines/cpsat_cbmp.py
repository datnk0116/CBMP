import os
import sys
import time
import json
import math
import argparse
from pathlib import Path
import numpy as np

# Ensure repo root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.io import mmread
from ortools.sat.python import cp_model
from cbmp import SolveResult

def solve_cpsat(
    dataset=None,
    N=None,
    edges=None,
    timeout=600.0,
    num_workers=1,
    relabel_seed=None,
    verbose=True,
):
    t_start = time.perf_counter()

    if isinstance(dataset, int):
        # solve_cpsat(N, edges, ...)
        N = dataset
        edges = N_or_edges if 'N_or_edges' in locals() else edges

    if N is not None and edges is not None:
        dataset_name = dataset if isinstance(dataset, str) else "custom"
    else:
        dataset_name = dataset if isinstance(dataset, str) else "unknown"
        filepath = f"datasets/{dataset}.mtx" if not dataset.endswith(".mtx") else dataset
        GRAPH = mmread(filepath)
        N = GRAPH.shape[0]
        rows, cols = GRAPH.nonzero()
        edges = [(u, v) for u, v in zip(rows, cols) if u < v]

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
        if verbose:
            print("minimum bandwidth: 0")
        return SolveResult(0, stats)

    # ============ BUILD MODEL
    t_enc_start = time.perf_counter()
    model = cp_model.CpModel()

    # Variables f[u] in [1, N]
    f = [model.NewIntVar(1, N, f"f_{u}") for u in range(N)]
    model.AddAllDifferent(f)

    # Objective k in [0, N // 2]
    k = model.NewIntVar(0, N // 2, "k")

    for u, v in edges:
        diff = model.NewIntVar(1, N - 1, f"diff_{u}_{v}")
        model.AddAbsEquality(diff, f[u] - f[v])
        b = model.NewBoolVar(f"b_{u}_{v}")

        # if b: diff <= k; if not b: diff >= N - k
        model.Add(diff <= k).OnlyEnforceIf(b)
        model.Add(diff >= N - k).OnlyEnforceIf(b.Not())

    model.Minimize(k)
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

    if cp_status == cp_model.OPTIMAL:
        best_ub = int(solver.ObjectiveValue())
        status = "OPTIMAL"
        if best_lb is None:
            best_lb = best_ub
        if verbose:
            print("minimum bandwidth:", best_ub)
    elif cp_status == cp_model.FEASIBLE:
        best_ub = int(solver.ObjectiveValue())
        status = "TIMEOUT"
        if verbose:
            print("timeout, lowest bandwidth found:", best_ub)
    elif cp_status == cp_model.UNKNOWN:
        best_ub = -1
        status = "TIMEOUT"
        if verbose:
            print("timeout, no valid assignment was found")
    else:
        best_ub = -1
        status = "ERROR"
        if verbose:
            print(f"CP-SAT finished with status: {solver.StatusName(cp_status)}")

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
    parser.add_argument("--json-out", type=str, default=None, help="path to append statistics as JSON line")
    args = parser.parse_args()

    res = solve_cpsat(
        dataset=args.dataset,
        timeout=args.timeout,
        num_workers=args.num_workers,
        relabel_seed=args.relabel_seed,
        verbose=True,
    )

    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(res.stats) + "\n")

