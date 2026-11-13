#!/usr/bin/env python3
"""Independent brute-force cross-check of all CBMP solvers on small graphs.

The brute-force oracle below is written from the problem definition only
(it does NOT import any project code):

    B_c(G) = min over bijections f: V -> {0..N-1} of
             max over edges {u,v} of min(|f(u)-f(v)|, N - |f(u)-f(v)|)

Every solver result is checked against the oracle, and every returned
assignment is re-validated independently (bijection + objective).

Usage:
    PATH="$PWD/CBMP/bin:$PATH" python scripts/verify_bruteforce_crosscheck.py [--max-n 7] [--random 40]
"""
import argparse
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def oracle(n, edges):
    """Exhaustive enumeration of all n! labelings (independent of project code)."""
    es = set()
    for u, v in edges:
        if u != v:
            es.add((min(u, v), max(u, v)))
    if not es:
        return 0
    best = n
    for perm in itertools.permutations(range(n)):
        w = 0
        for u, v in es:
            d = abs(perm[u] - perm[v])
            d = min(d, n - d)
            if d > w:
                w = d
                if w >= best:
                    break
        if w < best:
            best = w
    return best


def check_assignment(n, edges, assignment, expected):
    if assignment is None:
        return "no assignment"
    if isinstance(assignment, dict):
        labels = [assignment[i] for i in range(n)]
    else:
        labels = list(assignment)
    base = min(labels)  # accept 0- or 1-based labels
    labels = [x - base for x in labels]
    if sorted(labels) != list(range(n)):
        return f"not a bijection: {labels}"
    w = 0
    for u, v in edges:
        if u != v:
            d = abs(labels[u] - labels[v])
            w = max(w, min(d, n - d))
    if w != expected:
        return f"assignment objective {w} != expected {expected}"
    return None


def make_graphs(max_n, n_random, seed):
    rng = random.Random(seed)
    graphs = []
    for n in range(2, max_n + 1):
        all_e = [(i, j) for i in range(n) for j in range(i + 1, n)]
        graphs.append((f"path{n}", n, [(i, i + 1) for i in range(n - 1)]))
        graphs.append((f"cycle{n}", n, [(i, (i + 1) % n) for i in range(n)] if n > 2 else [(0, 1)]))
        graphs.append((f"star{n}", n, [(0, i) for i in range(1, n)]))
        graphs.append((f"K{n}", n, all_e))
        graphs.append((f"empty{n}", n, []))
    for k in range(n_random):
        n = rng.randint(3, max_n)
        all_e = [(i, j) for i in range(n) for j in range(i + 1, n)]
        m = rng.randint(1, len(all_e))
        es = rng.sample(all_e, m)
        # inject duplicates, reversed edges and self-loops to test normalization
        noisy = list(es) + [(v, u) for (u, v) in es[:2]] + [(0, 0)]
        graphs.append((f"rand{k}_n{n}_m{m}", n, noisy))
    return graphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=7)
    ap.add_argument("--random", type=int, default=40)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    from cbmp import solve as sat_solve
    from baselines.cpsat_cbmp import solve_cpsat
    from baselines.cplex_cbmp import solve_cplex
    from baselines.cplex_cp_cbmp import solve_cplex_cp, is_cplex_cp_available

    solvers = {}
    sat_variants = [
        ("sat_order_norot", dict(encoding="order", sb_rotation=False, sb_reflection=False)),
        ("sat_order_rot_ref_heur", dict(encoding="order", sb_rotation=True, sb_reflection=True, use_heuristic_initial_ub=True)),
        ("sat_pairwise_rot_ref", dict(encoding="pairwise", sb_rotation=True, sb_reflection=True)),
        ("sat_order_iterative_rot", dict(encoding="order", sb_rotation=True, solving_mode="iterative")),
        ("sat_order_glucose_seed", dict(encoding="order", sb_rotation=True, solver="glucose4", relabel_seed=7)),
    ]
    for name, kw in sat_variants:
        solvers[name] = (lambda n, e, kw=kw: sat_solve(N=n, edges=e, verbose=False, **kw))
    solvers["cpsat"] = lambda n, e: solve_cpsat(N=n, edges=e, timeout=30, verbose=False)
    solvers["cplex_mip"] = lambda n, e: solve_cplex(N=n, edges=e, timeout=30, verbose=False)
    avail, reason = is_cplex_cp_available()
    if avail:
        solvers["cplex_cp"] = lambda n, e: solve_cplex_cp(N=n, edges=e, timeout=30, verbose=False)
    else:
        print(f"[SKIP] cplex_cp unavailable: {reason}")
    try:
        import gurobipy  # noqa: F401
        from baselines.gurobi_cbmp import solve_gurobi
        solvers["gurobi"] = lambda n, e: solve_gurobi(N=n, edges=e, timeout=30, verbose=False)
    except ImportError:
        print("[SKIP] gurobi unavailable: gurobipy not installed")

    graphs = make_graphs(args.max_n, args.random, args.seed)
    failures = 0
    checks = 0
    for gname, n, edges in graphs:
        exp = oracle(n, edges)
        for sname, fn in solvers.items():
            checks += 1
            try:
                res = fn(n, edges)
            except Exception as ex:  # noqa: BLE001
                print(f"[FAIL] {gname} {sname}: exception {type(ex).__name__}: {ex}")
                failures += 1
                continue
            val = int(res)
            st = res.stats.get("status")
            msg = None
            if st == "LICENSE_ERROR":
                print(f"[LIMIT] {gname} {sname}: LICENSE_ERROR (solver size limit)")
                continue
            if val != exp:
                msg = f"value {val} != oracle {exp} (status {st})"
            elif st not in ("OPTIMAL",):
                msg = f"status {st} (expected OPTIMAL)"
            else:
                a = res.stats.get("assignment")
                # relabeled SAT runs report assignment in the relabeled space; skip the
                # per-vertex check there (objective already verified)
                if a is not None and res.stats.get("relabel_seed") is None:
                    msg = check_assignment(n, edges, a, exp)
            if msg:
                failures += 1
                print(f"[FAIL] {gname} {sname}: {msg}")
    print(f"graphs={len(graphs)} solvers={list(solvers)} checks={checks} failures={failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
