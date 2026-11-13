"""
SAT Encodings for Cyclic Bandwidth Minimization Problem (CBMP).

Implements both:
  1. Order Encoding: Uses ladder variables G[u][i] <=> (f(u) >= i) alongside one-hot X[u][i].
  2. Pairwise Encoding: Uses direct one-hot variables X[u][i] with pairwise forbidden-distance clauses.
"""

from typing import List, Tuple, Sequence, Optional
from alo_amo_eo_encodings import (
    encode_exactly_one,
    encode_at_most_one,
    exactly_one_commander,
    at_least_one,
    at_most_one_pairwise,
)

# ============================================================================
# COMMON DISTANCE HELPER
# ============================================================================

def cyclic_distance(pos1: int, pos2: int, N: int) -> int:
    """Compute cyclic distance on cycle of length N between 0-indexed positions."""
    diff = abs(pos1 - pos2)
    return min(diff, N - diff)


# ============================================================================
# 1. ORDER ENCODING
# ============================================================================
"""
G: upper bound of X
X[u][i] <-> (G[u][i] and -G[u][i+1])
"""

def monotonic_G(N: int, G: Sequence[Sequence[int]]) -> List[List[int]]:
    """
    Monotonicity constraint for ladder variables:
    G[u][i+1] -> G[u][i] == -G[u][i+1] v G[u][i]
    """
    clauses = []
    for u in range(N):
        for i in range(N):
            clauses.append([-G[u][i + 1], G[u][i]])
    return clauses

def lower_upper_bound_G(N: int, G: Sequence[Sequence[int]]) -> List[List[int]]:
    """
    Boundary conditions:
    G[u][0] = True   (f(u) >= 0 is always true)
    G[u][N] = False  (f(u) >= N is always false for 0-indexed values in 0..N-1)
    """
    clauses = []
    for u in range(N):
        clauses.append([G[u][0]])
        clauses.append([-G[u][N]])
    return clauses

def biconditional_X_and_G(
    N: int, X: Sequence[Sequence[int]], G: Sequence[Sequence[int]]
) -> List[List[int]]:
    """
    Biconditional link between exact value and order variables:
    X[u][i] <-> (G[u][i] and -G[u][i+1])
    """
    clauses = []
    for u in range(N):
        for i in range(N):
            # X -> G
            clauses.append([-X[u][i], G[u][i]])
            clauses.append([-X[u][i], -G[u][i + 1]])
            # G -> X
            clauses.append([-G[u][i], G[u][i + 1], X[u][i]])
    return clauses

def injectivity_surjectivity_X(
    N: int, X: Sequence[Sequence[int]], a_base: int, eo_strategy: str = "commander"
) -> Tuple[List[List[int]], int]:
    """
    Ensure every label position i is occupied by exactly one vertex u:
    EO(X[0][i], ..., X[N-1][i]).
    """
    clauses = []
    for i in range(N):
        new_clauses, a_base = encode_exactly_one(
            [X[u][i] for u in range(N)], a_base, strategy=eo_strategy
        )
        clauses.extend(new_clauses)
    return clauses, a_base

def cyclic_edge_order(
    N: int,
    X: Sequence[Sequence[int]],
    G: Sequence[Sequence[int]],
    u: int,
    v: int,
    UB: int,
) -> List[List[int]]:
    """
    Cyclic edge upper bound constraint for Order Encoding:
    cyclic_dist(f(u), f(v)) < UB, i.e., <= K where K = UB - 1.
    """
    K = UB - 1
    clauses = []

    for i in range(N):
        if i - K < 0:
            # Wraps left past 0 by (K - i) elements
            # allowed f(v): [0, i + K] or [N - (K - i), N - 1]
            # forbidden: [i + K + 1, N - (K - i) - 1]
            clauses.append([-X[u][i], -G[v][i + K + 1], G[v][N + i - K]])
        elif i + K > N - 1:
            # Wraps right past N - 1
            # allowed f(v): [0, K - (N - 1 - i) - 1] or [i - K, N - 1]
            # forbidden: [K - (N - 1 - i), i - K - 1]
            clauses.append([-X[u][i], -G[v][i + K - N + 1], G[v][i - K]])
        else:
            # No wrap. allowed f(v): [i - K, i + K]
            if i - K > 0:
                clauses.append([-X[u][i], G[v][i - K]])
            if i + K + 1 < N:
                clauses.append([-X[u][i], -G[v][i + K + 1]])

    return clauses

# Backward compatibility alias
cyclic_edge = cyclic_edge_order


# ============================================================================
# 2. PAIRWISE ENCODING
# ============================================================================

def cyclic_edge_pairwise(
    N: int, X: Sequence[Sequence[int]], u: int, v: int, UB: int
) -> List[List[int]]:
    """
    Direct pairwise forbidden-distance clauses:
    For an edge {u, v} and upper bound UB, prohibit all label pairs (i, j) with
    cyclic_distance(i, j, N) >= UB (i.e., > K = UB - 1).
    Clause: -X[u][i] v -X[v][j].
    """
    clauses = []
    for i in range(N):
        for j in range(N):
            if cyclic_distance(i, j, N) >= UB:
                clauses.append([-X[u][i], -X[v][j]])
    return clauses

def build_pairwise_base_formula(
    N: int, eo_strategy: str = "commander", a_start: Optional[int] = None
) -> Tuple[List[List[int]], List[List[int]], int]:
    """
    Build the permanent base clauses for Pairwise Encoding:
      - Variables X[u][i] for 0 <= u, i < N.
      - Exactly-One position per vertex: EO(X[u][0], ..., X[u][N-1]).
      - At-Most-One vertex per position: AMO(X[0][i], ..., X[N-1][i]).
    Returns: (clauses, X, next_free_var)
    """
    x_start = 1
    X = [[(x_start + u * N + i) for i in range(N)] for u in range(N)]
    a_base = a_start if a_start is not None else (1 + N * N)

    clauses: List[List[int]] = []

    # 1. Exactly one position per vertex
    for u in range(N):
        eo_clauses, a_base = encode_exactly_one(
            [X[u][i] for i in range(N)], a_base, strategy=eo_strategy
        )
        clauses.extend(eo_clauses)

    # 2. At most one vertex per position (together with vertex EO implies bijection)
    for i in range(N):
        amo_clauses, a_base = encode_at_most_one(
            [X[u][i] for u in range(N)], a_base, strategy=eo_strategy
        )
        clauses.extend(amo_clauses)

    return clauses, X, a_base

def build_order_base_formula(
    N: int, eo_strategy: str = "commander", a_start: Optional[int] = None
) -> Tuple[List[List[int]], List[List[int]], List[List[int]], int]:
    """
    Build the permanent base clauses for Order Encoding:
      - Variables X[u][i] and ladder G[u][i].
      - Monotonicity of G.
      - Lower and upper bounds of G.
      - Biconditional link X <-> G.
      - Injective mapping of labels.
    Returns: (clauses, X, G, next_free_var)
    """
    x_start = 1
    g_start = x_start + N * N
    a_base = a_start if a_start is not None else (g_start + (N + 1) * N)

    X = [[(x_start + u * N + i) for i in range(N)] for u in range(N)]
    G = [[(g_start + u * (N + 1) + i) for i in range(N + 1)] for u in range(N)]

    clauses: List[List[int]] = []
    inj_clauses, a_base = injectivity_surjectivity_X(N, X, a_base, eo_strategy=eo_strategy)
    clauses.extend(inj_clauses)
    clauses.extend(monotonic_G(N, G))
    clauses.extend(lower_upper_bound_G(N, G))
    clauses.extend(biconditional_X_and_G(N, X, G))

    return clauses, X, G, a_base


# ============================================================================
# 3. SYMMETRY BREAKING
# ============================================================================

def break_rotation_symmetry(X: Sequence[Sequence[int]], anchor_vertex: int = 0) -> List[List[int]]:
    """
    Fix f(anchor_vertex) = 0 to break rotational symmetry on Z_N.

    Indexing convention:
      Code uses 0-based vertices/labels (f(0) = 0), which corresponds to 1-based
      notation f(x_1) = 1 in the technical report.
    Valid for both Pairwise and Order encodings.
    """
    return [[X[anchor_vertex][0]]]

def break_reflection_symmetry_order(
    N: int, X: Sequence[Sequence[int]], G: Sequence[Sequence[int]]
) -> List[List[int]]:
    """
    After fixing f(0) = 0, enforce f(1) < f(N - 1) in Order Encoding.

    Indexing & size convention:
      - 0-based f(1) < f(N - 1) corresponds to 1-based f(x_2) < f(x_N) in the report.
      - Requires N >= 3: for N <= 2, vertices 1 and N - 1 coincide (or wrap), so
        enforcing f(1) < f(N - 1) is unsatisfiable; returns [] when N < 3.
    """
    if N < 3:
        return []
    clauses = []
    for i in range(N):
        clauses.append([-X[N - 1][i], -G[1][i]])
    return clauses

def break_reflection_symmetry_pairwise(
    N: int, X: Sequence[Sequence[int]]
) -> List[List[int]]:
    """
    After fixing f(0) = 0, enforce f(1) < f(N - 1) in Pairwise Encoding.
    For all i >= j, prohibit X[1][i] and X[N-1][j]: -X[1][i] v -X[N-1][j].

    Indexing & size convention:
      - 0-based f(1) < f(N - 1) corresponds to 1-based f(x_2) < f(x_N) in the report.
      - Requires N >= 3: returns [] when N < 3 to avoid contradicting f(1) = f(N - 1).
    """
    if N < 3:
        return []
    clauses = []
    for i in range(N):
        for j in range(i + 1):  # i >= j
            clauses.append([-X[1][i], -X[N - 1][j]])
    return clauses

# Backward compatibility alias
break_reflection_symmetry = break_reflection_symmetry_order
