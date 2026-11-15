"""Contract tests for cyclic rotation/reflection symmetry-breaking clauses.

These tests check the logical meaning of the clauses independently of a SAT
solver.  They protect the equivalence of the Order and Pairwise formulations.
"""

from sat_encodings import (
    break_reflection_symmetry_order,
    break_reflection_symmetry_pairwise,
    break_rotation_symmetry,
)


def _satisfies(clauses, assignment):
    """Evaluate a CNF formula under a complete variable -> bool assignment."""
    return all(any(assignment[abs(lit)] == (lit > 0) for lit in clause) for clause in clauses)


def test_rotation_breaking_fixes_anchor_to_zero():
    n = 7
    x = [[1 + u * n + i for i in range(n)] for u in range(n)]
    clauses = break_rotation_symmetry(x, anchor_vertex=0)

    # The unit clause must select label zero for the anchor vertex.
    assert clauses == [[x[0][0]]]


def test_reflection_breaking_matches_strict_order_in_both_encodings():
    """Both encodings must retain exactly assignments with f(1) < f(N-1)."""
    for n in range(3, 9):
        x = [[1 + u * n + i for i in range(n)] for u in range(n)]
        g_start = 1 + n * n
        g = [[g_start + u * (n + 1) + i for i in range(n + 1)] for u in range(n)]

        order_clauses = break_reflection_symmetry_order(n, x, g)
        pairwise_clauses = break_reflection_symmetry_pairwise(n, x)

        for f1 in range(n):
            for f_last in range(n):
                expected = f1 < f_last

                # Complete only the variables used by the reflection clauses.
                assignment = {}
                for i in range(n):
                    assignment[x[n - 1][i]] = (i == f_last)
                    assignment[g[1][i]] = (f1 >= i)

                assert _satisfies(order_clauses, assignment) is expected, (
                    f"Order encoding mismatch for N={n}, f(1)={f1}, "
                    f"f(N-1)={f_last}"
                )

                pair_assignment = {
                    x[1][i]: (i == f1) for i in range(n)
                }
                pair_assignment.update({
                    x[n - 1][i]: (i == f_last) for i in range(n)
                })
                assert _satisfies(pairwise_clauses, pair_assignment) is expected, (
                    f"Pairwise encoding mismatch for N={n}, f(1)={f1}, "
                    f"f(N-1)={f_last}"
                )


def test_reflection_breaking_is_disabled_for_degenerate_orders():
    """Reflection has no distinct orientation to choose for N < 3."""
    for n in (1, 2):
        x = [[1 + u * n + i for i in range(n)] for u in range(n)]
        g_start = 1 + n * n
        g = [[g_start + u * (n + 1) + i for i in range(n + 1)] for u in range(n)]

        assert break_reflection_symmetry_order(n, x, g) == []
        assert break_reflection_symmetry_pairwise(n, x) == []
