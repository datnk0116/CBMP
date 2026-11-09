import os
import sys
import time
import json
import argparse
from pathlib import Path

# Ensure repo root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.io import mmread
from docplex.mp.model import Model
import docplex.mp.utils
from cbmp import SolveResult

def solve_cplex(
    dataset=None,
    N=None,
    edges=None,
    timeout=600.0,
    verbose=True,
):
    if isinstance(dataset, int):
        N = dataset

    if N is not None and edges is not None:
        dataset_name = dataset if isinstance(dataset, str) else "custom"
    else:
        dataset_name = dataset if isinstance(dataset, str) else "unknown"
        filepath = f"datasets/{dataset}.mtx" if not dataset.endswith(".mtx") else dataset
        GRAPH = mmread(filepath)
        N = GRAPH.shape[0]
        rows, cols = GRAPH.nonzero()
        edges = [(u, v) for u, v in zip(rows, cols) if u < v]

    stats = {
        "dataset": dataset_name,
        "N": N,
        "num_edges": len(edges),
        "config": {
            "time_limit": timeout,
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
    mdl = Model(name="cbmp_mip")

    # Binary assignment variables x[u, i] = 1 iff vertex u has label i in 1..N
    x = {(u, i): mdl.binary_var(name=f"x_{u}_{i}") for u in range(N) for i in range(1, N + 1)}

    for u in range(N):
        mdl.add_constraint(mdl.sum(x[u, i] for i in range(1, N + 1)) == 1, ctname=f"alo_amo_v_{u}")
    for i in range(1, N + 1):
        mdl.add_constraint(mdl.sum(x[u, i] for u in range(N)) == 1, ctname=f"alo_amo_l_{i}")

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
    t_enc_end = time.perf_counter()
    encode_time = t_enc_end - t_enc_start

    num_vars = mdl.number_of_variables
    num_constraints = mdl.number_of_constraints

    stats["encode_time"] = encode_time
    stats["num_vars"] = num_vars
    stats["num_clauses"] = num_constraints

    # Community Edition check: 1000 variables and 1000 constraints limit
    if num_vars > 1000 or num_constraints > 1000:
        stats["status"] = "SIZE_LIMIT"
        stats["best_ub"] = -1
        if verbose:
            print(f"SIZE_LIMIT: Problem exceeds CPLEX Community Edition limits ({num_vars} vars, {num_constraints} consts)")
        return SolveResult(-1, stats)

    # Set time limit
    if timeout is not None and timeout > 0:
        mdl.parameters.timelimit = float(timeout)

    t_sol_start = time.perf_counter()
    try:
        sol = mdl.solve(log_output=False)
    except docplex.mp.utils.DOcplexLimitsExceeded:
        stats["status"] = "SIZE_LIMIT"
        stats["best_ub"] = -1
        if verbose:
            print(f"SIZE_LIMIT: DOcplexLimitsExceeded raised during solve ({num_vars} vars, {num_constraints} consts)")
        return SolveResult(-1, stats)
    except Exception as e:
        stats["status"] = "ERROR"
        stats["best_ub"] = -1
        if verbose:
            print(f"CPLEX solve error: {e}")
        return SolveResult(-1, stats)

    t_sol_end = time.perf_counter()
    solve_time = t_sol_end - t_sol_start

    stats["solve_time"] = solve_time
    stats["per_iter_times"] = [solve_time]
    stats["num_iterations"] = 1

    details = getattr(mdl, "solve_details", None)
    cpx_status_str = (getattr(details, "status", None) or "").lower() if details else ""
    cpx_status_code = getattr(details, "status_code", -1) if details else -1
    has_hit_limit = details.has_hit_limit() if details and hasattr(details, "has_hit_limit") else False

    # Check if CPLEX reported optimal
    # docplex status strings: "integer optimal solution", "optimal", "integer optimal, tolerance"
    # CPLEX status codes: 1 (optimal), 101 (integer optimal), 102 (integer optimal with tolerance)
    is_optimal = cpx_status_code in (1, 101, 102) or (
        ("integer optimal" in cpx_status_str or cpx_status_str == "optimal")
        and "infeasible" not in cpx_status_str
    )

    if is_optimal:
        best_ub = int(round(sol.get_value(k))) if sol is not None else -1
        status = "OPTIMAL"
        if verbose:
            print("minimum bandwidth:", best_ub)
    elif sol is not None:
        # Incumbent exists but optimality was not proven within limit -> TIMEOUT
        best_ub = int(round(sol.get_value(k)))
        status = "TIMEOUT"
        if verbose:
            print("timeout, lowest bandwidth found:", best_ub)
    elif has_hit_limit or "time limit" in cpx_status_str:
        best_ub = -1
        status = "TIMEOUT"
        if verbose:
            print("timeout, no valid assignment was found")
    else:
        best_ub = -1
        status = "ERROR"
        if verbose:
            print(f"CPLEX finished with status: {cpx_status_str or cpx_status_code}")

    stats["best_ub"] = best_ub
    stats["status"] = status

    return SolveResult(best_ub, stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve CBMP using IBM CPLEX / docplex MIP baseline")
    parser.add_argument("dataset", help="name of .mtx file in datasets/ or direct path")
    parser.add_argument("--timeout", type=float, default=600.0, help="time limit in seconds")
    parser.add_argument("--json-out", type=str, default=None, help="path to append statistics as JSON line")
    args = parser.parse_args()

    res = solve_cplex(
        dataset=args.dataset,
        timeout=args.timeout,
        verbose=True,
    )

    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(res.stats) + "\n")
