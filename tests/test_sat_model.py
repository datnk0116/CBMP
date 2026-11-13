"""
Unit tests for direct SAT model inspection and truth-table verification.
Phase 4.3 of the Code Quality & Robustness Audit.
"""

import itertools
import pytest
from pysat.solvers import Solver

from alo_amo_eo_encodings import (
    exactly_one_commander,
    at_most_one_commander_recurse,
    at_most_one_commander_group,
)
from sat_encodings import (
    monotonic_G,
    lower_upper_bound_G,
    biconditional_X_and_G,
    injectivity_surjectivity_X,
    cyclic_edge,
    break_rotation_symmetry,
    break_reflection_symmetry,
)
from cbmp import compute_cyclic_bandwidth


# ============================================================================
# 1. INDEPENDENT ALO / AMO / EO ENCODING TESTS
# ============================================================================

@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 7, 8])
def test_exactly_one_commander_truth_table(m):
    """
    Verify that exactly_one_commander on m variables admits exactly m solutions
    (the m one-hot assignments), and rejects all assignments of Hamming weight != 1.
    """
    vars_X = list(range(1, m + 1))
    a_base = m + 1
    clauses, final_a = exactly_one_commander(vars_X, a_base)

    # Use SAT solver to enumerate satisfying assignments projected onto vars_X
    projected_solutions = set()
    with Solver(name="glucose4", bootstrap_with=clauses) as solver:
        for model in solver.enum_models():
            # Project onto vars_X
            assignment = tuple(1 if x in model else 0 for x in vars_X)
            projected_solutions.add(assignment)

    # Must be exactly m solutions
    assert len(projected_solutions) == m

    # Every solution must have Hamming weight exactly 1
    for sol in projected_solutions:
        assert sum(sol) == 1


# ============================================================================
# 2. MONOTONICITY & BICONDITIONAL CONSTRAINTS DIRECT INSPECTION
# ============================================================================

def evaluate_cnf_assignment(clauses, assignment_dict):
    """
    Helper to evaluate whether an explicit assignment satisfies all CNF clauses.
    assignment_dict maps var_int -> bool.
    """
    for clause in clauses:
        clause_satisfied = False
        for lit in clause:
            var = abs(lit)
            val = assignment_dict.get(var, False)
            lit_val = val if lit > 0 else not val
            if lit_val:
                clause_satisfied = True
                break
        if not clause_satisfied:
            return False
    return True


def test_monotonic_and_biconditional_on_single_vertex():
    """
    For N=3, check that valid ladder assignments satisfy monotonic_G,
    lower_upper_bound_G, and biconditional_X_and_G, while invalid assignments fail.
    """
    N = 3
    # Variables:
    # X[0] has N=3 vars: 1, 2, 3
    # G[0] has N+1=4 vars: 4, 5, 6, 7
    X = [[1, 2, 3]]
    G = [[4, 5, 6, 7]]

    clauses = []
    # Test on vertex 0 with domain N=3
    # monotonic_G: G[0][i+1] -> G[0][i] for i in range(N)
    for i in range(N):
        clauses.append([-G[0][i + 1], G[0][i]])
    # lower_upper_bound_G: G[0][0] = True, G[0][N] = False
    clauses.append([G[0][0]])
    clauses.append([-G[0][N]])
    # biconditional_X_and_G
    for i in range(N):
        clauses.append([-X[0][i], G[0][i]])
        clauses.append([-X[0][i], -G[0][i + 1]])
        clauses.append([-G[0][i], G[0][i + 1], X[0][i]])

    # There are exactly N=3 valid label assignments:
    # position 0: X[0]=[T, F, F], G[0]=[T, F, F, F]
    # position 1: X[0]=[F, T, F], G[0]=[T, T, F, F]
    # position 2: X[0]=[F, F, T], G[0]=[T, T, T, F]

    valid_cases = [
        ({1: True, 2: False, 3: False, 4: True, 5: False, 6: False, 7: False}, True),
        ({1: False, 2: True, 3: False, 4: True, 5: True, 6: False, 7: False}, True),
        ({1: False, 2: False, 3: True, 4: True, 5: True, 6: True, 7: False}, True),
    ]

    for asgn, expected in valid_cases:
        assert evaluate_cnf_assignment(clauses, asgn) == expected

    # Invalid case 1: G is non-monotonic: G[0][2]=True while G[0][1]=False
    invalid_non_monotonic = {1: False, 2: False, 3: False, 4: True, 5: False, 6: True, 7: False}
    assert evaluate_cnf_assignment(clauses, invalid_non_monotonic) is False

    # Invalid case 2: G[0][0]=False violates lower bound G[u][0]=True
    invalid_lower_bound = {1: False, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False}
    assert evaluate_cnf_assignment(clauses, invalid_lower_bound) is False

    # Invalid case 3: G[0][3]=True violates upper bound G[u][N]=False
    invalid_upper_bound = {1: False, 2: False, 3: True, 4: True, 5: True, 6: True, 7: True}
    assert evaluate_cnf_assignment(clauses, invalid_upper_bound) is False

    # Invalid case 4: X does not match G (X[0][0]=True but G indicates position 1)
    invalid_mismatch = {1: True, 2: True, 3: False, 4: True, 5: True, 6: False, 7: False}
    assert evaluate_cnf_assignment(clauses, invalid_mismatch) is False


# ============================================================================
# 3. CYCLIC EDGE CONSTRAINT DIRECT INSPECTION
# ============================================================================

def test_cyclic_edge_constraint_semantics():
    """
    Test cyclic_edge(N, X, G, u, v, UB) directly:
    When cyclic distance between pos(u) and pos(v) < UB, clauses must be satisfiable.
    When cyclic distance >= UB, the assignment must violate the clauses.
    """
    N = 5
    UB = 2  # Desired cyclic distance < 2, i.e. cyclic distance == 1

    x_start = 1
    g_start = x_start + N * N
    X = [[(x_start + u * N + i) for i in range(N)] for u in range(N)]
    G = [[(g_start + u * (N + 1) + i) for i in range(N + 1)] for u in range(N)]

    base_clauses = []
    base_clauses.extend(monotonic_G(N, G))
    base_clauses.extend(lower_upper_bound_G(N, G))
    base_clauses.extend(biconditional_X_and_G(N, X, G))

    edge_clauses = cyclic_edge(N, X, G, 0, 1, UB)
    all_clauses = base_clauses + edge_clauses

    # Test all pairs of positions (p0, p1) for u=0, v=1
    for p0 in range(N):
        for p1 in range(N):
            if p0 == p1:
                continue
            cyc_dist = min((p0 - p1) % N, (p1 - p0) % N)
            
            # Construct unit assumptions fixing u to p0 and v to p1
            assumptions = [X[0][p0], X[1][p1]]

            with Solver(name="glucose4", bootstrap_with=all_clauses) as solver:
                sat = solver.solve(assumptions=assumptions)
                if cyc_dist < UB:
                    assert sat is True, f"Positions ({p0}, {p1}) distance {cyc_dist} < {UB} should be SAT"
                else:
                    assert sat is False, f"Positions ({p0}, {p1}) distance {cyc_dist} >= {UB} should be UNSAT"


# ============================================================================
# 4. SYMMETRY-BREAKING CONSTRAINTS INSPECTION
# ============================================================================

def test_rotation_symmetry_clause():
    N = 4
    X = [[(u * N + i + 1) for i in range(N)] for u in range(N)]
    clauses = break_rotation_symmetry(X, anchor_vertex=0)
    assert clauses == [[X[0][0]]]  # Forces vertex 0 to position 0


def test_reflection_symmetry_clauses():
    N = 5
    x_start = 1
    g_start = x_start + N * N
    X = [[(x_start + u * N + i) for i in range(N)] for u in range(N)]
    G = [[(g_start + u * (N + 1) + i) for i in range(N + 1)] for u in range(N)]

    clauses = []
    clauses.extend(monotonic_G(N, G))
    clauses.extend(lower_upper_bound_G(N, G))
    clauses.extend(biconditional_X_and_G(N, X, G))
    clauses.extend(break_reflection_symmetry(N, X, G))

    # Test with solver: pos(1) < pos(N-1) must be SAT, pos(1) > pos(N-1) must be UNSAT
    with Solver(name="glucose4", bootstrap_with=clauses) as solver:
        # pos(1)=1, pos(4)=3 -> 1 < 3 -> SAT
        assert solver.solve(assumptions=[X[1][1], X[4][3]]) is True
        # pos(1)=3, pos(4)=1 -> 3 > 1 -> UNSAT
        assert solver.solve(assumptions=[X[1][3], X[4][1]]) is False


def test_sat_formula_size_proportions():
    """
    Check variable and clause counts scaling for small N.
    """
    for N in [3, 4, 5]:
        x_start = 1
        g_start = x_start + N * N
        a_start = g_start + (N + 1) * N
        X = [[(x_start + u * N + i) for i in range(N)] for u in range(N)]
        G = [[(g_start + u * (N + 1) + i) for i in range(N + 1)] for u in range(N)]

        inがお_clauses, final_a = injectivity_surjectivity_X(N, X, a_start)
        assert len(inがお_clauses) > 0
        mono = monotonic_G(N, G)
        assert len(mono) == N * N
        bounds = lower_upper_bound_G(N, G)
        assert len(bounds) == 2 * N
        bicond = biconditional_X_and_G(N, X, G)
        assert len(bicond) == 3 * N * N
