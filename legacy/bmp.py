from scipy.io import mmread
from sat_encodings import *
from pysat.solvers import Cadical300

def main(dataset):
    # ============ READ INPUT
    GRAPH = mmread(f"datasets/662_bus/{dataset}") 
    N = GRAPH.shape[0]  # num vertices

    rows, cols = GRAPH.nonzero()  # edge if: matrix cell has nonzero val
    edges = [(u, v) for u, v in zip(rows, cols) if u < v] # ignoring duplicate pairs, and diagonal

    # ============ BUILD VARIABLES
    x_start = 1
    g_start = x_start + N * N
    a_start = g_start + (N + 1) * N
    X = [[(x_start + u * N + i) for i in range(N)] for u in range(N)]
    G = [[(g_start + u * (N + 1) + i) for i in range(N + 1)] for u in range(N)]

    # ============ BUILD BASE CLAUSES
    clauses = []

    new_clauses, a_start = injectivity_surjectivity_X(N, X, a_start)
    clauses.extend(new_clauses)
    clauses.extend(monotonic_G(N, G))
    clauses.extend(lower_upper_bound_G(N, G)) # this one forces all vertices to be at least 1 (G[u][1])
    clauses.extend(biconditional_X_and_G(N, X, G))

    # ============ INCREMENTAL SAT
    # === INIT
    UB = N # trivial bound

    solver = Cadical300()
    solver.append_formula(clauses)

    # === ITERATE
    while True:
        # === UPPER BOUND MINIMIZATION CLAUSES
        for edge in edges:
            u, v = edge

            for i in range(N - UB):
                # === |f(u) - f(v)| < UB
                # cho p = i + UB, bootstrap v up until u = p and vice versa
                solver.add_clause([-G[u][i + UB], G[v][i + 1]]) # u >= p ---> v > p - UB
                solver.add_clause([-G[v][i + UB], G[u][i + 1]]) # v >= p ---> u > p - UB
                

        result = solver.solve()
        if not result: break
        
        model = solver.get_model()

        # === EXTRACT BANDWIDTH
        values = [0 for i in range(N)]

        for lit in model[:N * N]:
            if lit > 0:
                u_, i_ = divmod(lit - 1, N) # divmod(lit - 1, N) returns ((lit - 1) // N, (lit - 1) % N) 
                values[u_] = i_

        bandwidth = 0
        for edge in edges:
            u, v = edge

            bandwidth = max(bandwidth, abs(values[u] - values[v])) 

        UB = bandwidth
        print(UB)

    solver.delete()

from sys import argv
if __name__ == "__main__":
    dataset = argv[1]

    main(dataset)
