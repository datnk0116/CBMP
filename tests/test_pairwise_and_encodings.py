"""
Comprehensive test suite for Pairwise Encoding, Order Encoding,
Cross-Encoding Equivalence, Incremental vs Iterative Solving,
EO/AMO Strategies, and Baseline Integrations.
"""

import math
import pytest
from pysat.solvers import Solver

from cbmp import solve, compute_cyclic_bandwidth
from tests.brute_force import cyclic_bandwidth_bruteforce
from sat_encodings import (
    build_pairwise_base_formula,
    build_order_base_formula,
    cyclic_edge_pairwise,
    cyclic_edge_order,
    break_rotation_symmetry,
    break_reflection_symmetry_pairwise,
    break_reflection_symmetry_order,
    cyclic_distance,
)
from alo_amo_eo_encodings import (
    encode_exactly_one,
    encode_at_most_one,
    EO_STRATEGIES,
    AMO_STRATEGIES,
)
from baselines.cplex_cp_cbmp import solve_cplex_cp, is_cplex_cp_available, build_cplex_cp_model
from baselines.gurobi_cbmp import solve_gurobi, is_gurobi_available, build_gurobi_model
from experiments.run_experiments import get_canonical_key


# ============================================================================
# 1. ALO / AMO / EO STRATEGY TRUTH-TABLE & MODEL COUNTING TESTS
# ============================================================================

@pytest.mark.parametrize("strategy", ["commander", "pairwise", "binary", "sequential", "product"])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 7, 8])
def test_all_eo_strategies_model_counts(strategy, m):
    """
    Every Exactly-One strategy on m variables must yield exactly m satisfying
    assignments (the m one-hot assignments), rejecting all others.
    """
    vars_X = list(range(1, m + 1))
    a_base = m + 1
    clauses, final_a = encode_exactly_one(vars_X, a_base, strategy=strategy)

    projected_solutions = set()
    with Solver(name="glucose4", bootstrap_with=clauses) as solver:
        for model in solver.enum_models():
            sol = tuple(1 if x in model else 0 for x in vars_X)
            projected_solutions.add(sol)

    assert len(projected_solutions) == m, f"Strategy {strategy} on m={m} gave {len(projected_solutions)} solutions"
    for sol in projected_solutions:
        assert sum(sol) == 1


@pytest.mark.parametrize("strategy", ["commander", "pairwise", "binary", "sequential", "product"])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
def test_all_amo_strategies_model_counts(strategy, m):
    """
    Every At-Most-One strategy on m variables must yield exactly m + 1 satisfying
    assignments (m one-hot assignments plus the all-zero assignment).
    """
    vars_X = list(range(1, m + 1))
    a_base = m + 1
    clauses, final_a = encode_at_most_one(vars_X, a_base, strategy=strategy)

    # Add tautologies for each variable in vars_X so solver knows unconstrained variables exist
    tautologies = [[x, -x] for x in vars_X]

    projected_solutions = set()
    with Solver(name="glucose4", bootstrap_with=clauses + tautologies) as solver:
        for model in solver.enum_models():
            sol = tuple(1 if x in model else 0 for x in vars_X)
            projected_solutions.add(sol)

    assert len(projected_solutions) == m + 1, f"Strategy {strategy} on m={m} gave {len(projected_solutions)} solutions"
    for sol in projected_solutions:
        assert sum(sol) <= 1


# ============================================================================
# 2. PAIRWISE ENCODING VS BRUTE FORCE GROUND TRUTH
# ============================================================================

SMALL_TEST_GRAPHS = [
    ("path_4", 4, [(0, 1), (1, 2), (2, 3)]),
    ("cycle_4", 4, [(0, 1), (1, 2), (2, 3), (0, 3)]),
    ("clique_4", 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
    ("path_5", 5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    ("cycle_5", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]),
    ("star_5", 5, [(0, 1), (0, 2), (0, 3), (0, 4)]),
    ("path_6", 6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
    ("cycle_6", 6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]),
    ("rnd_6", 6, [(0, 1), (1, 3), (2, 4), (3, 5), (0, 4)]),
]

@pytest.mark.parametrize("name, N, edges", SMALL_TEST_GRAPHS)
@pytest.mark.parametrize("sb_rot, sb_ref", [(False, False), (True, False), (True, True)])
def test_pairwise_matches_bruteforce(name, N, edges, sb_rot, sb_ref):
    expected = cyclic_bandwidth_bruteforce(N, edges)
    actual = solve(
        N=N,
        edges=edges,
        encoding="pairwise",
        sb_rotation=sb_rot,
        sb_reflection=sb_ref,
        verbose=False,
    )
    assert actual == expected
    assert actual.stats["status"] == "OPTIMAL"
    assert actual.stats["best_ub"] == expected


# ============================================================================
# 3. CROSS-ENCODING EQUIVALENCE: SAT_Pairwise(G, K) == SAT_Order(G, K)
# ============================================================================

@pytest.mark.parametrize("name, N, edges", SMALL_TEST_GRAPHS[:5])
def test_cross_encoding_satisfiability_all_bounds(name, N, edges):
    """
    For every candidate bound UB in [1, N // 2 + 1], Pairwise and Order
    encodings must have identical satisfiability.
    """
    for ub in range(1, N // 2 + 2):
        # 1. Build Pairwise formula for target UB
        pw_clauses, X_pw, _ = build_pairwise_base_formula(N)
        for u, v in edges:
            pw_clauses.extend(cyclic_edge_pairwise(N, X_pw, u, v, ub))

        with Solver(name="cadical195", bootstrap_with=pw_clauses) as s_pw:
            sat_pw = s_pw.solve()

        # 2. Build Order formula for target UB
        ord_clauses, X_ord, G_ord, _ = build_order_base_formula(N)
        for u, v in edges:
            ord_clauses.extend(cyclic_edge_order(N, X_ord, G_ord, u, v, ub))

        with Solver(name="cadical195", bootstrap_with=ord_clauses) as s_ord:
            sat_ord = s_ord.solve()

        assert sat_pw == sat_ord, f"Discrepancy on {name} at UB={ub}: pw={sat_pw}, ord={sat_ord}"


# ============================================================================
# 4. INCREMENTAL VS ITERATIVE SOLVING EQUIVALENCE
# ============================================================================

@pytest.mark.parametrize("name, N, edges", SMALL_TEST_GRAPHS[:6])
@pytest.mark.parametrize("encoding", ["order", "pairwise"])
def test_incremental_vs_iterative_equivalence(name, N, edges, encoding):
    expected = cyclic_bandwidth_bruteforce(N, edges)

    r_inc = solve(
        N=N,
        edges=edges,
        encoding=encoding,
        solving_mode="incremental",
        sb_rotation=True,
        sb_reflection=True,
        verbose=False,
    )
    r_iter = solve(
        N=N,
        edges=edges,
        encoding=encoding,
        solving_mode="iterative",
        sb_rotation=True,
        sb_reflection=True,
        verbose=False,
    )

    assert r_inc == expected
    assert r_iter == expected
    assert r_inc.stats["status"] == "OPTIMAL"
    assert r_iter.stats["status"] == "OPTIMAL"


# ============================================================================
# 5. CRITICAL CYCLIC WRAP-AROUND CASE (CYCLE C_5)
# ============================================================================

def test_wrap_around_critical_case_c5():
    """
    On C_5, linear bandwidth is 2 (e.g. vertices 0-1-2-3-4 with back-edge 0-4 has distance 4 linearly).
    However, optimal cyclic bandwidth is strictly 1:
      f(0)=0, f(1)=1, f(2)=2, f(3)=3, f(4)=4.
      Edges: (0,1)->1, (1,2)->1, (2,3)->1, (3,4)->1, (4,0)->min(4, 5-4)=1.
    Both Order and Pairwise encodings must return optimal cyclic bandwidth = 1.
    """
    N = 5
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    
    res_ord = solve(N=N, edges=edges, encoding="order", verbose=False)
    res_pw = solve(N=N, edges=edges, encoding="pairwise", verbose=False)

    assert res_ord == 1
    assert res_pw == 1


# ============================================================================
# 6. BASELINE AVAILABILITY & UNAVAILABLE STATUS HANDLING
# ============================================================================

def test_cplex_cp_unavailable_reporting():
    """CPLEX CP must return UNAVAILABLE without raising unhandled exceptions when engine is missing."""
    avail, msg = is_cplex_cp_available()
    res = solve_cplex_cp(N=4, edges=[(0, 1)], verbose=False)
    if not avail:
        assert res.stats["status"] == "UNAVAILABLE"
        assert res.ub == -1
        assert "error" in res.stats


def test_gurobi_unavailable_reporting():
    """Gurobi must return UNAVAILABLE without raising unhandled exceptions when module is missing."""
    avail, msg = is_gurobi_available()
    res = solve_gurobi(N=4, edges=[(0, 1)], verbose=False)
    if not avail:
        assert res.stats["status"] == "UNAVAILABLE"
        assert res.ub == -1
        assert "error" in res.stats


# ============================================================================
# 7. MOCKED BASELINE EXECUTION & SOLUTION VALIDATION
# ============================================================================

def test_cplex_cp_model_construction():
    """Verify that build_cplex_cp_model builds valid docplex.cp model objects."""
    pytest.importorskip("docplex.cp")
    mdl, f, k = build_cplex_cp_model(4, [(0, 1), (1, 2)], sb_rotation=True)
    assert len(f) == 4
    assert k is not None


def test_gurobi_model_construction():
    """Verify that build_gurobi_model constructs valid MIP variables and constraints when available."""
    pytest.importorskip("gurobipy")
    from baselines.gurobi_cbmp import build_gurobi_model
    model, x, f, k = build_gurobi_model(4, [(0, 1)], sb_rotation=True)
    assert model is not None


# ============================================================================
# 8. EXHAUSTIVE EDGE CASES & GRAPH FAMILIES (N=1, N=2, N>=3, DISCONNECTED, COMPLETE)
# ============================================================================

EXHAUSTIVE_EDGE_CASES = [
    ("empty_1", 1, []),
    ("empty_2", 2, []),
    ("single_edge_2", 2, [(0, 1)]),
    ("empty_3", 3, []),
    ("path_3", 3, [(0, 1), (1, 2)]),
    ("triangle_3", 3, [(0, 1), (1, 2), (0, 2)]),
    ("empty_4", 4, []),
    ("disconnected_2k2_4", 4, [(0, 1), (2, 3)]),
    ("star_4", 4, [(0, 1), (0, 2), (0, 3)]),
    ("clique_4", 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
    ("disconnected_k3_k2_5", 5, [(0, 1), (1, 2), (0, 2), (3, 4)]),
    ("clique_5", 5, [(u, v) for u in range(5) for v in range(u + 1, 5)]),
]


@pytest.mark.parametrize("name, N, edges", EXHAUSTIVE_EDGE_CASES)
def test_exhaustive_order_and_pairwise_vs_bruteforce(name, N, edges):
    """Exhaustively verify both encodings against brute force on edge cases."""
    expected = cyclic_bandwidth_bruteforce(N, edges)

    res_ord, stats_ord = solve(N=N, edges=edges, encoding="order", verbose=False)
    res_pw, stats_pw = solve(N=N, edges=edges, encoding="pairwise", verbose=False)

    assert res_ord == expected, f"Order encoding failed on {name}: expected {expected}, got {res_ord}"
    assert res_pw == expected, f"Pairwise encoding failed on {name}: expected {expected}, got {res_pw}"
    assert stats_ord["status"] == "OPTIMAL"
    assert stats_pw["status"] == "OPTIMAL"
    assert stats_ord["best_ub"] == expected
    assert stats_pw["best_ub"] == expected


@pytest.mark.parametrize("name, N, edges", [
    ("empty_1", 1, []),
    ("empty_2", 2, []),
    ("edge_2", 2, [(0, 1)]),
    ("empty_3", 3, []),
    ("path_3", 3, [(0, 1), (1, 2)]),
    ("triangle_3", 3, [(0, 1), (1, 2), (0, 2)]),
    ("empty_4", 4, []),
    ("disconnected_2k2_4", 4, [(0, 1), (2, 3)]),
    ("isolated_plus_k3_4", 4, [(1, 2), (2, 3), (1, 3)]),
    ("path_4", 4, [(0, 1), (1, 2), (2, 3)]),
    ("cycle_4", 4, [(0, 1), (1, 2), (2, 3), (0, 3)]),
    ("star_4", 4, [(0, 1), (0, 2), (0, 3)]),
    ("clique_4", 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
    ("path_5", 5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    ("cycle_5", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]),
    ("star_5", 5, [(0, 1), (0, 2), (0, 3), (0, 4)]),
])
@pytest.mark.parametrize("encoding", ["order", "pairwise"])
@pytest.mark.parametrize("sb_rot, sb_ref", [
    (False, False),
    (True, False),
    (True, True),
])
def test_exhaustive_symmetry_breaking_combinations(name, N, edges, encoding, sb_rot, sb_ref):
    """
    Verify that symmetry-breaking combinations preserve exact optimal cyclic bandwidth
    across N=1, 2, >=3, empty, disconnected, path, cycle, star, and complete graphs,
    and returned assignments are valid permutations.
    """
    from graph_utils import validate_solution
    expected = cyclic_bandwidth_bruteforce(N, edges)
    res, stats = solve(
        N=N,
        edges=edges,
        encoding=encoding,
        sb_rotation=sb_rot,
        sb_reflection=sb_ref,
        verbose=False,
    )
    assert res == expected, f"Failed on {name} with {encoding} (rot={sb_rot}, ref={sb_ref}): expected {expected}, got {res}"
    assert stats["status"] == "OPTIMAL"
    valid, actual_bw, err = validate_solution(N, edges, stats["assignment"], reported_objective=res)
    assert valid is True, f"Assignment validation failed on {name}: {err}"
    assert actual_bw == expected


def test_reflection_only_enforces_prerequisite():
    """Verify solve() enforces that reflection symmetry breaking requires rotation symmetry breaking."""
    with pytest.raises(ValueError, match="requires --sb-rotation"):
        solve(N=3, edges=[(0, 1), (1, 2)], sb_rotation=False, sb_reflection=True, verbose=False)


@pytest.mark.parametrize("encoding", ["order", "pairwise"])
def test_rotation_anchor_present_in_formula_when_both_enabled(encoding):
    """
    Verify that when both rotation and reflection symmetry breaking are enabled,
    the generated formula explicitly contains the unit clause fixing f(0) = 0.
    """
    N = 4
    if encoding == "order":
        from sat_encodings import build_order_base_formula, break_rotation_symmetry, break_reflection_symmetry_order
        base_clauses, X, G, nv = build_order_base_formula(N)
        rot_clauses = break_rotation_symmetry(X, 0)
        ref_clauses = break_reflection_symmetry_order(N, X, G)
    else:
        from sat_encodings import build_pairwise_base_formula, break_rotation_symmetry, break_reflection_symmetry_pairwise
        base_clauses, X, nv = build_pairwise_base_formula(N)
        rot_clauses = break_rotation_symmetry(X, 0)
        ref_clauses = break_reflection_symmetry_pairwise(N, X)

    combined = base_clauses + rot_clauses + ref_clauses
    expected_unit = [X[0][0]]
    assert expected_unit in combined, f"Formula for {encoding} must contain rotation anchor unit clause {expected_unit}"


@pytest.mark.parametrize("encoding", ["order", "pairwise"])
@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_formula_level_reflection_semantics(encoding, N):
    """
    Verify semantic correctness of reflection symmetry breaking at formula level:
    - N < 3 must return empty list (safe boundary handling).
    - N >= 3 must enforce f(1) < f(N - 1).
    """
    if encoding == "order":
        from sat_encodings import build_order_base_formula, break_reflection_symmetry_order
        clauses, X, G, nv = build_order_base_formula(N)
        ref_clauses = break_reflection_symmetry_order(N, X, G)
        if N < 3:
            assert ref_clauses == [], f"Order reflection for N={N} must be empty"
        else:
            assert len(ref_clauses) == N
            for i, cl in enumerate(ref_clauses):
                assert cl == [-X[N - 1][i], -G[1][i]]
    else:
        from sat_encodings import build_pairwise_base_formula, break_reflection_symmetry_pairwise
        clauses, X, nv = build_pairwise_base_formula(N)
        ref_clauses = break_reflection_symmetry_pairwise(N, X)
        if N < 3:
            assert ref_clauses == [], f"Pairwise reflection for N={N} must be empty"
        else:
            expected_count = N * (N + 1) // 2
            assert len(ref_clauses) == expected_count
            for cl in ref_clauses:
                assert len(cl) == 2
                assert cl[0] < 0 and cl[1] < 0


@pytest.mark.parametrize("encoding", ["order", "pairwise"])
@pytest.mark.parametrize("sb_rot,sb_ref", [
    (False, False),
    (True, False),
    (True, True),
])
def test_formula_level_symmetry_combinations(encoding, sb_rot, sb_ref):
    """
    Verify formula construction across symmetry breaking combinations:
    - Unit clause [X[0][0]] is present if and only if sb_rotation=True.
    - Reflection clauses are present if and only if sb_rotation=True and sb_reflection=True (for N >= 3).
    - Ensures tests never assert reflection symmetry clauses when reflection is disabled.
    """
    N = 4
    if encoding == "order":
        from sat_encodings import (
            build_order_base_formula,
            break_rotation_symmetry,
            break_reflection_symmetry_order,
        )
        base_clauses, X, G, nv = build_order_base_formula(N)
        clauses = list(base_clauses)
        if sb_rot:
            clauses.extend(break_rotation_symmetry(X, 0))
            if sb_ref and N >= 3:
                clauses.extend(break_reflection_symmetry_order(N, X, G))
        ref_candidates = break_reflection_symmetry_order(N, X, G)
    else:
        from sat_encodings import (
            build_pairwise_base_formula,
            break_rotation_symmetry,
            break_reflection_symmetry_pairwise,
        )
        base_clauses, X, nv = build_pairwise_base_formula(N)
        clauses = list(base_clauses)
        if sb_rot:
            clauses.extend(break_rotation_symmetry(X, 0))
            if sb_ref and N >= 3:
                clauses.extend(break_reflection_symmetry_pairwise(N, X))
        ref_candidates = break_reflection_symmetry_pairwise(N, X)

    unit_anchor = [X[0][0]]
    if sb_rot:
        assert unit_anchor in clauses, f"Rotation anchor {unit_anchor} must be in formula when sb_rotation=True"
    else:
        assert unit_anchor not in clauses, f"Rotation anchor {unit_anchor} must not be in formula when sb_rotation=False"

    if sb_ref:
        for cl in ref_candidates:
            assert cl in clauses, f"Reflection clause {cl} must be in formula when sb_reflection=True"
    else:
        for cl in ref_candidates:
            assert cl not in clauses, f"Reflection clause {cl} must not be in formula when sb_reflection=False"


def enumerate_projected_sat_models(N, edges, UB, encoding="order", sb_rot=False, sb_ref=False):
    """
    Enumerate all distinct vertex label assignments f: V -> {0..N-1}
    satisfying the SAT formula for upper bound UB (cyclic bandwidth <= UB - 1).
    Projects onto original X variables only: auxiliary variables (ladder G, commander/AMO)
    are excluded by adding blocking clauses strictly over X variables.
    """
    from pysat.solvers import Cadical195

    if encoding == "order":
        from sat_encodings import (
            build_order_base_formula,
            cyclic_edge_order,
            break_rotation_symmetry,
            break_reflection_symmetry_order,
        )
        base_clauses, X, G, final_a = build_order_base_formula(N)
        clauses = list(base_clauses)
        if sb_rot:
            clauses.extend(break_rotation_symmetry(X, 0))
            if sb_ref and N >= 3:
                clauses.extend(break_reflection_symmetry_order(N, X, G))
        for u, v in edges:
            clauses.extend(cyclic_edge_order(N, X, G, u, v, UB))
    else:  # pairwise
        from sat_encodings import (
            build_pairwise_base_formula,
            cyclic_edge_pairwise,
            break_rotation_symmetry,
            break_reflection_symmetry_pairwise,
        )
        base_clauses, X, final_a = build_pairwise_base_formula(N)
        clauses = list(base_clauses)
        if sb_rot:
            clauses.extend(break_rotation_symmetry(X, 0))
            if sb_ref and N >= 3:
                clauses.extend(break_reflection_symmetry_pairwise(N, X))
        for u, v in edges:
            clauses.extend(cyclic_edge_pairwise(N, X, u, v, UB))

    solver = Cadical195()
    solver.append_formula(clauses)

    projected_assignments = set()
    while solver.solve():
        model = set(solver.get_model())
        assignment = [None] * N
        blocking = []
        for u in range(N):
            for i in range(N):
                var = X[u][i]
                if var in model:
                    assignment[u] = i
                    blocking.append(-var)
                    break
        assert None not in assignment, f"Incomplete assignment in SAT model: {assignment}"
        projected_assignments.add(tuple(assignment))
        solver.add_clause(blocking)

    solver.delete()
    return projected_assignments


@pytest.mark.parametrize("name, N, edges", [
    ("edge_2", 2, [(0, 1)]),
    ("path_3", 3, [(0, 1), (1, 2)]),
    ("path_4", 4, [(0, 1), (1, 2), (2, 3)]),
    ("cycle_4", 4, [(0, 1), (1, 2), (2, 3), (0, 3)]),
    ("star_4", 4, [(0, 1), (0, 2), (0, 3)]),
    ("disconnected_2k2_4", 4, [(0, 1), (2, 3)]),
])
@pytest.mark.parametrize("encoding", ["order", "pairwise"])
@pytest.mark.parametrize("sb_rot, sb_ref", [
    (False, False),
    (True, False),
    (True, True),
])
def test_projected_sat_models_match_canonical_orbit_representatives(name, N, edges, encoding, sb_rot, sb_ref):
    """
    Genuine SAT model enumeration and orbit verification:
    1. Enumerate all valid bijections V -> {0..N-1} via brute force achieving bandwidth <= opt.
    2. Partition solutions into rotational/dihedral equivalence classes under D_N.
    3. Construct expected canonical representatives (f(0)=0 for rot; f(0)=0 and f(1)<f(N-1) for dihedral).
    4. Enumerate SAT models, projecting strictly onto original X variables (excluding auxiliaries).
    5. Compare projected SAT assignments against expected canonical representatives:
       - Every equivalence class has exactly one representative.
       - No equivalence class is lost.
       - No invalid representative is introduced.
    """
    import itertools
    opt = cyclic_bandwidth_bruteforce(N, edges)
    UB = opt + 1  # SAT upper bound condition: bandwidth < UB <=> bandwidth <= opt

    all_perms = list(itertools.permutations(range(N)))
    feasible_perms = {
        p for p in all_perms if compute_cyclic_bandwidth(N, edges, list(p)) <= opt
    }
    assert len(feasible_perms) > 0

    # Determine expected brute-force representative set
    if not sb_rot and not sb_ref:
        expected_reps = feasible_perms
    elif sb_rot and not sb_ref:
        # Rotation only: fix f(0) = 0
        expected_reps = {p for p in feasible_perms if p[0] == 0}
        # Verify every feasible permutation's rotation orbit is represented
        for p in feasible_perms:
            shift = (N - p[0]) % N
            rot_p = tuple((x + shift) % N for x in p)
            assert rot_p in expected_reps
    else:  # sb_rot and sb_ref
        if N < 3:
            expected_reps = {p for p in feasible_perms if p[0] == 0}
        else:
            expected_reps = {p for p in feasible_perms if p[0] == 0 and p[1] < p[N - 1]}
            # Verify every feasible permutation's dihedral orbit is represented
            for p in feasible_perms:
                shift = (N - p[0]) % N
                r = tuple((x + shift) % N for x in p)
                if r[1] > r[N - 1]:
                    r = tuple((N - x) % N for x in r)
                assert r in expected_reps

    # Enumerate actual projected SAT models
    actual_sat_models = enumerate_projected_sat_models(
        N=N, edges=edges, UB=UB, encoding=encoding, sb_rot=sb_rot, sb_ref=sb_ref
    )

    # Bijective set comparison between projected SAT models and expected representatives
    assert actual_sat_models == expected_reps, (
        f"SAT projected models mismatch on {name} ({encoding}, rot={sb_rot}, ref={sb_ref}):\n"
        f"  Missing: {expected_reps - actual_sat_models}\n"
        f"  Unexpected: {actual_sat_models - expected_reps}"
    )



@pytest.mark.parametrize("name, N, edges", [
    ("empty_4", 4, []),
    ("star_4", 4, [(0, 1), (0, 2), (0, 3)]),
    ("path_5", 5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    ("clique_4", 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
])
@pytest.mark.parametrize("encoding", ["order", "pairwise"])
@pytest.mark.parametrize("heuristic_ub", [True, False])
def test_exhaustive_heuristic_ub_enabled_disabled(name, N, edges, encoding, heuristic_ub):
    """Verify solver correctness with heuristic initial UB enabled vs disabled."""
    expected = cyclic_bandwidth_bruteforce(N, edges)
    res, stats = solve(
        N=N,
        edges=edges,
        encoding=encoding,
        use_heuristic_initial_ub=heuristic_ub,
        verbose=False,
    )
    assert res == expected
    assert stats["status"] == "OPTIMAL"


def test_canonical_key_distinguishes_solver_and_encoding_records():
    """Verify experiment canonical keys never collide between distinct formulations."""
    base_task = {
        "dataset": "30_bus",
        "method": "sat",
        "solver": "cadical195",
        "sb_rotation": True,
        "sb_reflection": True,
        "heuristic_ub": True,
        "relabel_seed": None,
    }
    key_order = get_canonical_key(dict(base_task, encoding="order"))
    key_pairwise = get_canonical_key(dict(base_task, encoding="pairwise"))
    key_iterative = get_canonical_key(dict(base_task, encoding="order", solving_mode="iterative"))
    key_product_eo = get_canonical_key(dict(base_task, encoding="order", eo_strategy="product"))

    all_keys = [key_order, key_pairwise, key_iterative, key_product_eo]
    assert len(set(all_keys)) == 4, "Every distinct encoding/mode/eo configuration must produce a unique canonical key"

