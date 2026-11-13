import os
import sys
from pathlib import Path

# Ensure repo root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import random
import pytest
from sat_encodings import (
    injectivity_surjectivity_X,
    monotonic_G,
    lower_upper_bound_G,
    biconditional_X_and_G,
    build_pairwise_base_formula,
    break_rotation_symmetry,
    break_reflection_symmetry_order,
    break_reflection_symmetry_pairwise,
    break_reflection_symmetry,
    cyclic_edge,
    cyclic_edge_pairwise,
)
from pysat.solvers import Cadical195
from tests.brute_force import cyclic_bandwidth_bruteforce

def count_models_on_X(N, edges, UB, sb_rot, sb_ref):
    """
    Enumerate all distinct assignments of vertex label variables X_u,i (u in 0..N-1)
    satisfying the CBMP SAT formula for upper bound UB.
    Auxiliary variables (order encoding G, commander variables) are projected out
    by adding blocking clauses over only the X variables.
    """
    if not edges or UB <= 0:
        return 0

    x_start = 1
    g_start = x_start + N * N
    a_start = g_start + (N + 1) * N
    X = [[(x_start + u * N + i) for i in range(N)] for u in range(N)]
    G = [[(g_start + u * (N + 1) + i) for i in range(N + 1)] for u in range(N)]

    clauses = []
    c, _ = injectivity_surjectivity_X(N, X, a_start)
    clauses.extend(c)
    clauses.extend(monotonic_G(N, G))
    clauses.extend(lower_upper_bound_G(N, G))
    clauses.extend(biconditional_X_and_G(N, X, G))

    if sb_rot:
        clauses.extend(break_rotation_symmetry(X, 0))
        if sb_ref:
            clauses.extend(break_reflection_symmetry(N, X, G))

    for u, v in edges:
        clauses.extend(cyclic_edge(N, X, G, u, v, UB))

    solver = Cadical195()
    solver.append_formula(clauses)

    count = 0
    while solver.solve():
        model = set(solver.get_model())
        blocking = []
        for u in range(N):
            for i in range(N):
                var = X[u][i]
                if var in model:
                    blocking.append(-var)
                    break
        count += 1
        solver.add_clause(blocking)

    solver.delete()
    return count


def count_models_on_X_pairwise(N, edges, UB, sb_rot, sb_ref, eo_strategy="commander"):
    """
    Enumerate all distinct assignments of vertex label variables X_u,i (u in 0..N-1)
    satisfying the Pairwise-encoded CBMP SAT formula for upper bound UB.
    Auxiliary commander EO variables are projected out by blocking only X variables.
    """
    if not edges or UB <= 0:
        return 0

    clauses, X, _ = build_pairwise_base_formula(N, eo_strategy=eo_strategy)

    if sb_rot:
        clauses.extend(break_rotation_symmetry(X, 0))
        if sb_ref:
            clauses.extend(break_reflection_symmetry_pairwise(N, X))

    for u, v in edges:
        clauses.extend(cyclic_edge_pairwise(N, X, u, v, UB))

    solver = Cadical195()
    solver.append_formula(clauses)

    count = 0
    while solver.solve():
        model = set(solver.get_model())
        blocking = []
        for u in range(N):
            for i in range(N):
                var = X[u][i]
                if var in model:
                    blocking.append(-var)
                    break
        count += 1
        solver.add_clause(blocking)

    solver.delete()
    return count

def generate_benchmark_graphs():
    """
    Generates a suite of graphs with N <= 9 including tightly-constrained topologies:
    1. 3x3 Grid (N=9, |E|=12)
    2. Cycles with 1 and 2 chords (N=7, 8)
    3. Binary tree (N=7, |E|=6)
    4. Random graphs with |E| approx 1.5N for N in 5..9
    5. Standard paths (P_N) and cycles (C_N) for N in 4..8
    """
    graphs = []

    # 1. 3x3 Grid (N=9)
    grid_edges = []
    for r in range(3):
        for c in range(3):
            u = r * 3 + c
            if r + 1 < 3:
                grid_edges.append((u, (r + 1) * 3 + c))
            if c + 1 < 3:
                grid_edges.append((u, r * 3 + c + 1))
    graphs.append(("Grid_3x3", 9, grid_edges))

    # 2. Cycle plus 1-2 chords (N=7, 8)
    c7 = [(i, i + 1) for i in range(6)] + [(0, 6)]
    graphs.append(("Cycle7_1chord", 7, c7 + [(0, 3)]))
    graphs.append(("Cycle7_2chords", 7, c7 + [(0, 3), (1, 5)]))

    c8 = [(i, i + 1) for i in range(7)] + [(0, 7)]
    graphs.append(("Cycle8_1chord", 8, c8 + [(0, 4)]))
    graphs.append(("Cycle8_2chords", 8, c8 + [(0, 4), (2, 6)]))

    # 3. Binary tree N=7
    graphs.append(("BinTree_N7", 7, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]))

    # 4. Random graphs with |E| approx 1.5N (N=5..9)
    rng = random.Random(42)
    for N in range(5, 10):
        m = round(1.5 * N)
        all_possible = [(u, v) for u in range(N) for v in range(u + 1, N)]
        edges = sorted(rng.sample(all_possible, m))
        graphs.append((f"Rnd_1.5N_N{N}", N, edges))

    # 5. Standard path graphs (P_N) for N=4..8
    for N in range(4, 9):
        graphs.append((f"P_{N}", N, [(i, i + 1) for i in range(N - 1)]))

    # 6. Standard cycle graphs (C_N) for N=4..8
    for N in range(4, 9):
        graphs.append((f"C_{N}", N, [(i, i + 1) for i in range(N - 1)] + [(0, N - 1)]))

    return graphs

def run_solution_counting(output_csv="results/symmetry_counts.csv", print_table=True, encoding="order"):
    enc_norm = (encoding or "order").strip().lower()
    if enc_norm not in ("order", "pairwise"):
        raise ValueError(f"Unsupported encoding for solution counting: {encoding}")
    count_fn = count_models_on_X if enc_norm == "order" else count_models_on_X_pairwise

    graphs = generate_benchmark_graphs()
    results = []

    header = [
        "graph",
        "N",
        "num_edges",
        "UB",
        "bound_type",
        "trivial",
        "none_count",
        "rot_count",
        "rot_ref_count",
        "ratio_rot",
        "ratio_ref",
        "expected_rot",
        "expected_ref",
        "status",
        "notes",
    ]

    for name, N, edges in graphs:
        opt = cyclic_bandwidth_bruteforce(N, edges)

        # Evaluate at both UB = opt (must be UNSAT, 0 models) and UB = opt + 1 (SAT)
        evaluations = [
            ("opt", opt),
            ("opt+1", opt + 1),
        ]

        for bound_type, UB in evaluations:
            is_trivial = UB > (N // 2)

            c_none = count_fn(N, edges, UB, sb_rot=False, sb_ref=False)
            c_rot = count_fn(N, edges, UB, sb_rot=True, sb_ref=False)
            c_ref = count_fn(N, edges, UB, sb_rot=True, sb_ref=True)

            if bound_type == "opt":
                # Expect 0 models because distance strictly < opt is impossible
                if c_none == 0 and c_rot == 0 and c_ref == 0:
                    status = "UNSAT_OK"
                    notes = "0 solutions (strictly < opt is UNSAT, proving optimality)"
                else:
                    status = "POTENTIAL_BUG_NOT_UNSAT"
                    notes = f"Expected 0 models at UB=opt, but got none={c_none}, rot={c_rot}, ref={c_ref}"
                ratio_rot_str = "N/A"
                ratio_ref_str = "N/A"
                expected_rot_str = "N/A"
                expected_ref_str = "N/A"
            else:
                ratio_rot = (c_none / c_rot) if c_rot > 0 else 0.0
                ratio_ref = (c_rot / c_ref) if c_ref > 0 else 0.0
                expected_rot = float(N)
                expected_ref = 2.0

                rot_ok = abs(ratio_rot - expected_rot) < 1e-4
                ref_ok = abs(ratio_ref - expected_ref) < 1e-4

                if rot_ok and ref_ok:
                    status = "MATCH_THEORY"
                    notes = "Exact theoretical reduction (N for rotation, 2 for reflection)"
                elif not rot_ok:
                    status = "POTENTIAL_BUG_ROT"
                    notes = f"Rotation ratio {ratio_rot:.2f} deviates from N={N}"
                else:
                    status = "POTENTIAL_BUG_REF"
                    notes = f"Reflection ratio {ratio_ref:.2f} deviates from expected 2.0"

                ratio_rot_str = f"{ratio_rot:.2f}"
                ratio_ref_str = f"{ratio_ref:.2f}"
                expected_rot_str = f"{expected_rot:.1f}"
                expected_ref_str = f"{expected_ref:.1f}"

            row = {
                "graph": name,
                "N": N,
                "num_edges": len(edges),
                "UB": UB,
                "bound_type": bound_type,
                "trivial": is_trivial,
                "none_count": c_none,
                "rot_count": c_rot,
                "rot_ref_count": c_ref,
                "ratio_rot": ratio_rot_str,
                "ratio_ref": ratio_ref_str,
                "expected_rot": expected_rot_str,
                "expected_ref": expected_ref_str,
                "status": status,
                "notes": notes,
            }
            results.append(row)

    if print_table:
        # Separate non-trivial rows (main table) and trivial rows (auxiliary table)
        non_trivial_rows = [r for r in results if not r["trivial"]]
        trivial_rows = [r for r in results if r["trivial"]]

        print("\n" + "=" * 120)
        print("MAIN RESULTS TABLE: NON-TRIVIAL BOUNDS (UB <= N // 2)")
        print("=" * 120)
        print(f"{'Graph':<16} | {'N':>2} | {'|E|':>3} | {'Type':<5} | {'UB':>2} | {'Triv':<5} | {'None':>6} | {'Rot':>6} | {'Rot+Ref':>7} | {'Ratio(Rot)':>10} | {'Ratio(Ref)':>10} | {'Status':<14}")
        print("-" * 120)
        for r in non_trivial_rows:
            print(f"{r['graph']:<16} | {r['N']:>2} | {r['num_edges']:>3} | {r['bound_type']:<5} | {r['UB']:>2} | {str(r['trivial']):<5} | {r['none_count']:>6} | {r['rot_count']:>6} | {r['rot_ref_count']:>7} | {r['ratio_rot']:>10} | {r['ratio_ref']:>10} | {r['status']:<14}")

        if trivial_rows:
            print("\n" + "=" * 120)
            print("SUPPLEMENTARY RESULTS TABLE: TRIVIAL BOUNDS (UB > N // 2, all permutations valid)")
            print("=" * 120)
            print(f"{'Graph':<16} | {'N':>2} | {'|E|':>3} | {'Type':<5} | {'UB':>2} | {'Triv':<5} | {'None':>6} | {'Rot':>6} | {'Rot+Ref':>7} | {'Ratio(Rot)':>10} | {'Ratio(Ref)':>10} | {'Status':<14}")
            print("-" * 120)
            for r in trivial_rows:
                print(f"{r['graph']:<16} | {r['N']:>2} | {r['num_edges']:>3} | {r['bound_type']:<5} | {r['UB']:>2} | {str(r['trivial']):<5} | {r['none_count']:>6} | {r['rot_count']:>6} | {r['rot_ref_count']:>7} | {r['ratio_rot']:>10} | {r['ratio_ref']:>10} | {r['status']:<14}")

        print("=" * 120 + "\n")

    # Write CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)

    if print_table:
        print(f"Results successfully written to: {output_csv}")

    return results

def test_count_solutions_match_theory_and_unsat(tmp_path):
    """
    Pytest verification (Order encoding):
    1. At UB = opt: exactly 0 models (UNSAT).
    2. At UB = opt + 1: rotation divides by N, reflection divides by 2.
    """
    results = run_solution_counting(output_csv=str(tmp_path / "symmetry_counts_order.csv"), print_table=False, encoding="order")
    for r in results:
        if r["bound_type"] == "opt":
            assert r["status"] == "UNSAT_OK", (
                f"Graph {r['graph']} failed UNSAT at UB={r['UB']}: "
                f"expected 0 models, got none={r['none_count']}, rot={r['rot_count']}, ref={r['rot_ref_count']}"
            )
        else:
            assert r["status"] == "MATCH_THEORY", (
                f"Graph {r['graph']} failed symmetry reduction: "
                f"ratio_rot={r['ratio_rot']} (expected {r['expected_rot']}), "
                f"ratio_ref={r['ratio_ref']} (expected {r['expected_ref']})"
            )


def test_count_solutions_pairwise_matches_order_and_theory(tmp_path):
    """
    A6: Pairwise encoding projected model counts on X must match Order encoding
    exact counts across none / rotation / rotation+reflection and achieve
    exact N (rotation) and 2 (reflection) reduction factors.
    """
    order_results = run_solution_counting(
        output_csv=str(tmp_path / "symmetry_counts_order.csv"),
        print_table=False,
        encoding="order",
    )
    pairwise_results = run_solution_counting(
        output_csv=str(tmp_path / "symmetry_counts_pairwise.csv"),
        print_table=False,
        encoding="pairwise",
    )
    assert len(pairwise_results) == len(order_results)

    for ord_r, pw_r in zip(order_results, pairwise_results):
        assert pw_r["graph"] == ord_r["graph"]
        assert pw_r["bound_type"] == ord_r["bound_type"]
        if pw_r["bound_type"] == "opt":
            assert pw_r["status"] == "UNSAT_OK"
        else:
            assert pw_r["status"] == "MATCH_THEORY", (
                f"Pairwise SB reduction failed on {pw_r['graph']}: {pw_r}"
            )
        assert pw_r["none_count"] == ord_r["none_count"]
        assert pw_r["rot_count"] == ord_r["rot_count"]
        assert pw_r["rot_ref_count"] == ord_r["rot_ref_count"]

if __name__ == "__main__":
    run_solution_counting()
