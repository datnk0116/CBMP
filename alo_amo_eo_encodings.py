import math
from typing import List, Tuple, Sequence, Optional

# ============================================================================
# ALO (At-Least-One)
# ============================================================================

def at_least_one(X: Sequence[int]) -> List[List[int]]:
    """
    At-least-one constraint: at least one literal in X must be True.
    For empty X, returns a contradictory empty clause [[]].
    """
    if not X:
        return [[]]
    return [list(X)]


# ============================================================================
# 1. PAIRWISE AMO & EO
# ============================================================================

def at_most_one_pairwise(X: Sequence[int]) -> List[List[int]]:
    """
    Direct quadratic pairwise AMO: -x_i v -x_j for all i < j.
    Introduces 0 auxiliary variables.
    """
    clauses = []
    n = len(X)
    for i in range(n - 1):
        for j in range(i + 1, n):
            clauses.append([-X[i], -X[j]])
    return clauses

def exactly_one_pairwise(X: Sequence[int], a_base: int) -> Tuple[List[List[int]], int]:
    """
    Direct pairwise Exactly-One: ALO + pairwise AMO.
    Introduces 0 auxiliary variables, returns (clauses, a_base).
    """
    if len(X) == 0:
        return [[]], a_base
    if len(X) == 1:
        return [[X[0]]], a_base

    clauses = at_least_one(X)
    clauses.extend(at_most_one_pairwise(X))
    return clauses, a_base


# ============================================================================
# 2. COMMANDER AMO & EO
# ============================================================================

GROUP_WIDTH = 3

def at_most_one_commander_group(X: Sequence[int], commander: Optional[int]) -> List[List[int]]:
    pairwise_clauses = []

    if len(X) == 1:
        if commander is not None:
            pairwise_clauses.append([-X[0], commander])
            pairwise_clauses.append([-commander, X[0]])

    elif len(X) == 2:
        if commander is not None:
            pairwise_clauses.append([-X[0], commander])
            pairwise_clauses.append([-X[1], commander])
            pairwise_clauses.append([-commander, X[0], X[1]])
        pairwise_clauses.append([-X[0], -X[1]])

    elif len(X) >= 3:
        if commander is not None:
            for x in X:
                pairwise_clauses.append([-x, commander])
            pairwise_clauses.append([-commander] + list(X))
        for i in range(len(X) - 1):
            for j in range(i + 1, len(X)):
                pairwise_clauses.append([-X[i], -X[j]])

    return pairwise_clauses

def at_most_one_commander_recurse(
    X: Sequence[int], a_base: int, clauses: List[List[int]]
) -> Tuple[List[int], int]:
    if len(X) <= GROUP_WIDTH:
        clauses.extend(at_most_one_commander_group(X, None))
        return list(X), a_base

    commanders = []
    n_groups = math.ceil(len(X) / GROUP_WIDTH)

    for g in range(n_groups):
        commander = a_base
        commanders.append(commander)
        a_base += 1

        group = X[g * GROUP_WIDTH : (g + 1) * GROUP_WIDTH]
        clauses.extend(at_most_one_commander_group(list(group), commander))

    top_commanders, a_base = at_most_one_commander_recurse(commanders, a_base, clauses)
    return top_commanders, a_base

def exactly_one_commander(X: Sequence[int], a_base: int) -> Tuple[List[List[int]], int]:
    """
    Commander Exactly-One:
    Partitions variables into groups of width <= 3 with commander variables,
    recursing until top-level commanders have ALO enforced.
    """
    if len(X) == 0:
        return [[]], a_base
    if len(X) == 1:
        return [[X[0]]], a_base

    clauses: List[List[int]] = []
    top_commanders, a_base = at_most_one_commander_recurse(X, a_base, clauses)
    clauses.append(list(top_commanders))
    return clauses, a_base

def at_most_one_commander(X: Sequence[int], a_base: int) -> Tuple[List[List[int]], int]:
    """
    Commander At-Most-One without the top-level ALO clause.
    """
    if len(X) <= 1:
        return [], a_base
    clauses: List[List[int]] = []
    _, a_base = at_most_one_commander_recurse(X, a_base, clauses)
    return clauses, a_base


# ============================================================================
# 3. BINARY (BIT) AMO & EO
# ============================================================================

def at_most_one_binary(X: Sequence[int], a_base: int) -> Tuple[List[List[int]], int]:
    """
    Binary / logarithmic AMO encoding:
    Uses ceil(log2(m)) auxiliary bit variables b_0, ..., b_{k-1}.
    Each variable x_i implies the binary representation of i.
    """
    m = len(X)
    if m <= 1:
        return [], a_base

    k = math.ceil(math.log2(m))
    bit_vars = [a_base + j for j in range(k)]
    a_base += k

    clauses: List[List[int]] = []
    for i, lit in enumerate(X):
        for j in range(k):
            bit = (i >> j) & 1
            if bit == 1:
                clauses.append([-lit, bit_vars[j]])
            else:
                clauses.append([-lit, -bit_vars[j]])

    # For unused codes v from m to 2^k - 1, forbid the combination
    for v in range(m, 1 << k):
        forbid = []
        for j in range(k):
            bit = (v >> j) & 1
            if bit == 1:
                forbid.append(-bit_vars[j])
            else:
                forbid.append(bit_vars[j])
        clauses.append(forbid)

    return clauses, a_base

def exactly_one_binary(X: Sequence[int], a_base: int) -> Tuple[List[List[int]], int]:
    """
    Binary Exactly-One: ALO + binary AMO.
    """
    if len(X) == 0:
        return [[]], a_base
    if len(X) == 1:
        return [[X[0]]], a_base

    clauses = at_least_one(X)
    amo_clauses, a_base = at_most_one_binary(X, a_base)
    clauses.extend(amo_clauses)
    return clauses, a_base


# ============================================================================
# 4. SEQUENTIAL COUNTER AMO & EO (Sinz 2005)
# ============================================================================

def at_most_one_sequential(X: Sequence[int], a_base: int) -> Tuple[List[List[int]], int]:
    """
    Sequential counter AMO encoding (Sinz 2005):
    Uses m - 1 auxiliary register variables s_0, ..., s_{m-2}.
    """
    m = len(X)
    if m <= 1:
        return [], a_base

    s_vars = [a_base + i for i in range(m - 1)]
    a_base += (m - 1)

    clauses: List[List[int]] = []
    # x_0 -> s_0
    clauses.append([-X[0], s_vars[0]])

    for i in range(1, m - 1):
        # x_i -> s_i
        clauses.append([-X[i], s_vars[i]])
        # s_{i-1} -> s_i
        clauses.append([-s_vars[i - 1], s_vars[i]])
        # not (x_i and s_{i-1})
        clauses.append([-X[i], -s_vars[i - 1]])

    # not (x_{m-1} and s_{m-2})
    clauses.append([-X[m - 1], -s_vars[m - 2]])

    return clauses, a_base

def exactly_one_sequential(X: Sequence[int], a_base: int) -> Tuple[List[List[int]], int]:
    """
    Sequential Counter Exactly-One: ALO + sequential counter AMO.
    """
    if len(X) == 0:
        return [[]], a_base
    if len(X) == 1:
        return [[X[0]]], a_base

    clauses = at_least_one(X)
    amo_clauses, a_base = at_most_one_sequential(X, a_base)
    clauses.extend(amo_clauses)
    return clauses, a_base


# ============================================================================
# 5. PRODUCT AMO & EO (Chen 2010)
# ============================================================================

def at_most_one_product(X: Sequence[int], a_base: int) -> Tuple[List[List[int]], int]:
    """
    Product AMO encoding (Chen 2010):
    Arranges variables in a p x q grid, with row and column indicators.
    """
    m = len(X)
    if m <= 3:
        # Fall back to pairwise for small sets where grid overhead is unnecessary
        return at_most_one_pairwise(X), a_base

    p = int(math.isqrt(m))
    q = math.ceil(m / p)

    row_vars = [a_base + i for i in range(p)]
    col_vars = [a_base + p + j for j in range(q)]
    a_base += (p + q)

    clauses: List[List[int]] = []
    for k, lit in enumerate(X):
        u = k // q
        v = k % q
        if u < p and v < q:
            clauses.append([-lit, row_vars[u]])
            clauses.append([-lit, col_vars[v]])

    # AMO on row variables
    clauses.extend(at_most_one_pairwise(row_vars))
    # AMO on column variables
    clauses.extend(at_most_one_pairwise(col_vars))

    return clauses, a_base

def exactly_one_product(X: Sequence[int], a_base: int) -> Tuple[List[List[int]], int]:
    """
    Product Exactly-One: ALO + product AMO.
    """
    if len(X) == 0:
        return [[]], a_base
    if len(X) == 1:
        return [[X[0]]], a_base

    clauses = at_least_one(X)
    amo_clauses, a_base = at_most_one_product(X, a_base)
    clauses.extend(amo_clauses)
    return clauses, a_base


# ============================================================================
# 6. UNIFIED DISPATCHERS
# ============================================================================

EO_STRATEGIES = {
    "commander": exactly_one_commander,
    "pairwise": exactly_one_pairwise,
    "binary": exactly_one_binary,
    "sequential": exactly_one_sequential,
    "product": exactly_one_product,
}

AMO_STRATEGIES = {
    "commander": at_most_one_commander,
    "pairwise": lambda X, a: (at_most_one_pairwise(X), a),
    "binary": at_most_one_binary,
    "sequential": at_most_one_sequential,
    "product": at_most_one_product,
}

def encode_exactly_one(
    X: Sequence[int], a_base: int, strategy: str = "commander"
) -> Tuple[List[List[int]], int]:
    strat = strategy.lower().strip()
    if strat not in EO_STRATEGIES:
        avail = ", ".join(sorted(EO_STRATEGIES.keys()))
        raise ValueError(f"Unknown EO strategy '{strategy}'. Available: {avail}")
    return EO_STRATEGIES[strat](X, a_base)

def encode_at_most_one(
    X: Sequence[int], a_base: int, strategy: str = "commander"
) -> Tuple[List[List[int]], int]:
    strat = strategy.lower().strip()
    if strat not in AMO_STRATEGIES:
        avail = ", ".join(sorted(AMO_STRATEGIES.keys()))
        raise ValueError(f"Unknown AMO strategy '{strategy}'. Available: {avail}")
    return AMO_STRATEGIES[strat](X, a_base)
