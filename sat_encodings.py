from alo_amo_eo_encodings import *

# ============================== ORDER ENCODING ========================================
"""
G: upper bound of X
X[u][i] <-> (G[u][i] and -G[u][i+1])
"""
# === monotonicity
def monotonic_G(N, G):
    """
    G[u][i] -> G[u][i-1]
    G[u][i+1] -> G[u][i] == -G[u][i] -> -G[u][i+1]
    """
    clauses = []
    for u in range(N):
        for i in range(N):
            clauses.append([-G[u][i + 1], G[u][i]]) # G[u][i+1] -> G[u][i]
    return clauses

# === G lower, upper bounds
def lower_upper_bound_G(N, G):
    """
    G[u][1] = true  : for every u
    G[u][N] = false : for every u
    """
    clauses = []
    for u in range(N):
        clauses.append([G[u][0]])
        clauses.append([-G[u][N]])

    return clauses

# === X_and_G
def biconditional_X_and_G(N, X, G):
    """
    X[u][i] <-> (G[u][i] and -G[u][i+1])
        * X[u][i] -> (G[u][i] and -G[u][i+1])
        * (G[u][i] and -G[u][i+1]) -> X[u][i]
    """
    clauses = []
    for u in range(N):
        for i in range(N):
            # X to G
            clauses.append([-X[u][i], G[u][i]])
            clauses.append([-X[u][i], -G[u][i + 1]])

            # G to X
            clauses.append([-G[u][i], G[u][i + 1], X[u][i]])
    return clauses


# === injectivity and surjectivity X
def injectivity_surjectivity_X(N, X, a_base):
    """
    x_u != x_v: X[u][i] -> -X[v][i] == -X[u][i] or -X[v][i]  ==  -X[v][i] or -X[u][i] == X[v][i] -> -X[u][i]  
    """
    clauses = []
    for i in range(N):
        new_clauses, a_base = exactly_one_commander([X[u][i] for u in range(N)], a_base)
        clauses.extend(new_clauses)
    return clauses, a_base


# === cyclic edges
def cyclic_edge(N, X, G, u, v, UB):
    """
    cyclic dist(pos(u), pos(v)) < UB
    """
    K = UB - 1 # desired upperbound

    clauses = []

    # fix f(x_u) = i for each i
    for i in range(N):
        if i - K < 0:
            # wraps left past 0 by K - i elements
            #   allowed f(v): [0, i + K] or [N - (K - i), N - 1]
            #   forbidden: [i + K + 1, N - (K - i) - 1]
            clauses.append([-X[u][i], -G[v][i + K + 1], G[v][N + i - K]])
        
        elif i + K > N - 1:
            # wraps right past N-1 by K - (N - 1 - i) elements
            #   allowed f(v): [0, K - (N - 1 - i) - 1] or [i - K, N - 1]
            #   forbidden: [K - (N - 1 - i), i - K - 1]
            clauses.append([-X[u][i], -G[v][i + K - N + 1], G[v][i - K]])
        
        else:
            # no wrap. allowed f(v): [i-K, i+K]
            if i - K > 0:
                clauses.append([-X[u][i], G[v][i - K]])        # f(v) >= i-K
            if i + K + 1 < N:
                clauses.append([-X[u][i], -G[v][i + K + 1]])   # f(v) <= i+K
    
    return clauses


# === CBMP symmetry breaking
def break_rotation_symmetry(X, anchor_vertex):
    """
    fix f(anchor_vertex) = 0 to break rotational symmetry.
    """
    return [[X[anchor_vertex][0]]]


def break_reflection_symmetry(N, X, G):
    """
    assume x_1 is fixed at label 0, enforce f(x_2) < f(x_N).
    Only valid for N >= 3 where vertex 1 and vertex N-1 are distinct.
    """
    if N < 3:
        return []
    clauses = []
    for i in range(N):
        clauses.append([-X[N-1][i], -G[1][i]])

    return clauses


# ===================================== UNUSUED ========================================
# # === LT
# def lt(G, v, k):
#     """
#     if X[v] < k, then -G[v][k]
#     """
#     return [-G[v][k]]

# # === GTE
# def gte(G, v, k):
#     """
#     if X[v] >= k, then G[v][k]
#     """
#     return [G[v][k]]

# def injectivity_X(N, X):
#     """
#     x_u != x_v: X[u][i] -> -X[v][i] == -X[u][i] or -X[v][i]  ==  -X[v][i] or -X[u][i] == X[v][i] -> -X[u][i]  
#     """
#     clauses = []
#     for i in range(N):
#         for u in range(N - 1):
#             for v in range(u + 1, N):
#                 clauses.append([-X[u][i], -X[v][i]])
#     return clauses

# # === injectivity G
# def injectivity_G(N, G):
#     """
#     (G[u][i] and -G[u][i+1]) -> -(G[v][i] and -G[v][i+1])  
#     """
#     clauses = []
#     for i in range(N):
#         for u in range(N - 1):
#             for v in range(u + 1, N):
#                 clauses.append([-G[u][i], G[u][i + 1], -G[v][i], G[v][i + 1]])
#     return clauses




# # ========================= NAIVE EXACT VALUE LT, GT ENCODINGS =========================
# # need |v - u| < ub, so: if u = j, then j - ub < v < j + ub
# # can act as EO
# def less_than(N, nv, ui, k): # ui: u = i
#     clauses = []
#     # v < i + ub -----> v <= i + ub - 1
#     for j in range(1, k):
#         clauses.append([-ui, nv + (j - 1)])     # (u = i) --> v = j, j < k
        
#     for j in range(k, N + 1):
#         clauses.append([-ui, -(nv + (j - 1))])  # (u = i) --> -(v = j), j >= k

#     nv += N
#     return clauses, nv

# def greater_than(N, nv, ui, k): # ui: u = i
#     clauses = []
#     # v > i - ub -----> v >= i - ub + 1
#     for j in range(1, k + 1):
#         clauses.append([-ui, -(nv + (j - 1))])  # (u = i) --> -(v = j), j <= k 
            
#     for j in range(k + 1, N + 1):
#         clauses.append([-ui, (nv + (j - 1))])   # (u = i) --> v = j, j > k

#     nv += N
#     return clauses, nv
