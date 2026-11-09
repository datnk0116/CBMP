from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from sat_encodings import *
import pysat.solvers
import numpy as np
import time

SOLVER_MAP = {
    "cadical195": getattr(pysat.solvers, "Cadical195", None),
    "glucose4": getattr(pysat.solvers, "Glucose4", None),
    "maplechrono": getattr(pysat.solvers, "MapleChrono", None),
    "minisat22": getattr(pysat.solvers, "Minisat22", None),
    "lingeling": getattr(pysat.solvers, "Lingeling", None),
}

AVAILABLE_SOLVERS = {}
for name, cls in SOLVER_MAP.items():
    if cls is not None:
        try:
            inst = cls()
            inst.delete()
            AVAILABLE_SOLVERS[name] = cls
        except Exception:
            pass

def get_solver_class(solver_name):
    if not isinstance(solver_name, str):
        if callable(solver_name):
            return solver_name
        raise ValueError(f"Invalid solver: {solver_name}")
    key = solver_name.lower().strip()
    if key not in AVAILABLE_SOLVERS:
        avail_str = ", ".join(sorted(AVAILABLE_SOLVERS.keys()))
        raise ValueError(f"Unknown or unavailable solver '{solver_name}'. Available solvers: {avail_str}")
    return AVAILABLE_SOLVERS[key]

class SolveResult(int):
    def __new__(cls, ub, stats):
        obj = super().__new__(cls, ub)
        obj.stats = stats
        obj.ub = ub
        return obj

    def __iter__(self):
        return iter((self.ub, self.stats))

    def __repr__(self):
        return str(self.ub)

def compute_cyclic_bandwidth(N, edges, assignment):
    if not edges:
        return 0
    return max(
        min(
            abs(assignment[u] - assignment[v]), 
            N - abs(assignment[u] - assignment[v])
        ) for u, v in edges
    )

def cuthill_mckee_layout(GRAPH, N):
    perm = reverse_cuthill_mckee(GRAPH.tocsr(), symmetric_mode=False)      # perm[k] = vertex placed at position k
    assignment = np.empty(N, dtype=int)
    assignment[perm] = np.arange(N)                                        # assignment[u] = position of vertex u
    return assignment

def solve(
    dataset=None,
    sb_rotation=False,
    sb_reflection=False,
    use_heuristic_initial_ub=False,
    best_ub=None,
    N=None,
    edges=None,
    verbose=True,
    use_symmetry_break=None,
    solver="cadical195",
    shared_stats=None,
):
    # ============ RESOLVE BACKWARD COMPATIBILITY & ARGUMENTS
    if use_symmetry_break is not None:
        sb_rotation = bool(use_symmetry_break)
        sb_reflection = bool(use_symmetry_break)

    if isinstance(dataset, int):
        # Direct positional call: solve(N, edges, ...)
        N = dataset
        edges = sb_rotation
        if isinstance(sb_reflection, bool) and isinstance(use_heuristic_initial_ub, bool) and isinstance(best_ub, bool):
            # solve(N, edges, sb_rotation, sb_reflection, use_heuristic_initial_ub)
            sb_rotation = sb_reflection
            sb_reflection = use_heuristic_initial_ub
            use_heuristic_initial_ub = best_ub
            best_ub = None
        elif isinstance(sb_reflection, bool) and isinstance(use_heuristic_initial_ub, bool):
            # solve(N, edges, use_symmetry_break, use_heuristic_initial_ub)
            sym = sb_reflection
            sb_rotation = sym
            sb_reflection = sym
            best_ub = None
        else:
            sb_rotation = False
            sb_reflection = False
        verbose = False
    elif best_ub is None and hasattr(use_heuristic_initial_ub, 'value'):
        # Legacy positional call: solve(dataset, use_symmetry_break, use_heuristic_initial_ub, best_ub)
        best_ub = use_heuristic_initial_ub
        use_heuristic_initial_ub = sb_reflection
        sym = sb_rotation
        sb_rotation = sym
        sb_reflection = sym

    if sb_reflection and not sb_rotation:
        raise ValueError("--sb-reflection requires --sb-rotation (reflection symmetry breaking assumes rotation symmetry breaking is active)")

    solver_cls = get_solver_class(solver)
    solver_name = solver if isinstance(solver, str) else getattr(solver_cls, "__name__", "custom")

    dataset_name = dataset if isinstance(dataset, str) else "custom"

    if N is not None and edges is not None:
        # Direct (N, edges) input
        if edges:
            rows = [u for u, v in edges] + [v for u, v in edges]
            cols = [v for u, v in edges] + [u for u, v in edges]
            data = [1] * len(rows)
            GRAPH = csr_matrix((data, (rows, cols)), shape=(N, N))
        else:
            GRAPH = csr_matrix((N, N), dtype=int)
    else:
        # Read from file
        GRAPH = mmread(f"datasets/{dataset}.mtx") 
        N = GRAPH.shape[0]  # num vertices

        rows, cols = GRAPH.nonzero()  # edge if: matrix cell has nonzero val
        edges = [(u, v) for u, v in zip(rows, cols) if u < v] # ignoring duplicate pairs, and diagonal

    stats = {
        "dataset": dataset_name,
        "N": N,
        "num_edges": len(edges),
        "config": {
            "sb_rotation": sb_rotation,
            "sb_reflection": sb_reflection,
            "heuristic_ub": use_heuristic_initial_ub,
        },
        "solver": solver_name,
        "status": "RUNNING",
        "best_ub": -1,
        "encode_time": 0.0,
        "solve_time": 0.0,
        "per_iter_times": [],
        "num_iterations": 0,
        "num_vars": 0,
        "num_clauses": 0,
    }
    if shared_stats is not None:
        shared_stats.update(stats)

    if not edges:
        if best_ub is not None:
            best_ub.value = 0
        if verbose:
            print(0)
        stats["best_ub"] = 0
        stats["status"] = "OPTIMAL"
        if shared_stats is not None:
            shared_stats.update(stats)
        return SolveResult(0, stats)

    # ============ BUILD VARIABLES & BASE CLAUSES
    t_enc_start = time.perf_counter()

    x_start = 1
    g_start = x_start + N * N
    a_start = g_start + (N + 1) * N
    X = [[(x_start + u * N + i) for i in range(N)] for u in range(N)]
    G = [[(g_start + u * (N + 1) + i) for i in range(N + 1)] for u in range(N)]

    clauses = []
    injectivity_surjectivity_clauses, final_a = injectivity_surjectivity_X(N, X, a_start)

    clauses.extend(injectivity_surjectivity_clauses)
    clauses.extend(monotonic_G(N, G))
    clauses.extend(lower_upper_bound_G(N, G)) # this one forces all vertices to be at least 1 (G[u][0])
    clauses.extend(biconditional_X_and_G(N, X, G))
    
    # === CBMP symmetry breaking
    if sb_rotation:
        ANCHOR_VERTEX = 0
        clauses.extend(break_rotation_symmetry(X, ANCHOR_VERTEX))
        if sb_reflection and N >= 3:
            clauses.extend(break_reflection_symmetry(N, X, G))

    t_enc_end = time.perf_counter()
    encode_time = t_enc_end - t_enc_start

    num_vars = final_a - 1
    total_clauses = len(clauses)

    stats["encode_time"] = encode_time
    stats["num_vars"] = num_vars
    stats["num_clauses"] = total_clauses
    if shared_stats is not None:
        shared_stats["encode_time"] = encode_time
        shared_stats["num_vars"] = num_vars
        shared_stats["num_clauses"] = total_clauses

    # ============ 2. HEURISTIC INITIAL UB & VARIABLE SEPARATION
    # UB: bound used for SAT clauses encoding
    # verified_best_ub: bandwidth of an actual verified assignment (starts at -1)
    t_heur_start = time.perf_counter()
    verified_best_ub = -1
    if use_heuristic_initial_ub:
        HEURISTIC_ASSIGNMENT = cuthill_mckee_layout(GRAPH, N)
        # Verify assignment is a valid permutation of 0..N-1
        try:
            is_valid = len(HEURISTIC_ASSIGNMENT) == N and sorted(HEURISTIC_ASSIGNMENT) == list(range(N))
        except Exception:
            is_valid = False

        if is_valid:
            bandwidth = int(compute_cyclic_bandwidth(N, edges, HEURISTIC_ASSIGNMENT))
            UB = bandwidth
            verified_best_ub = bandwidth
        else:
            UB = N // 2 + 1
    else:
        UB = N // 2 + 1  # trivial bound, with cyclic values, dist cant exceed N//2
    t_heur_end = time.perf_counter()
    heuristic_time = t_heur_end - t_heur_start

    stats["heuristic_time"] = heuristic_time
    stats["initial_ub"] = UB
    stats["best_ub"] = verified_best_ub

    # Only write to best_ub.value when an actual verified assignment exists
    if best_ub is not None and verified_best_ub != -1:
        best_ub.value = verified_best_ub
    if shared_stats is not None:
        shared_stats["heuristic_time"] = heuristic_time
        shared_stats["initial_ub"] = UB
        shared_stats["best_ub"] = verified_best_ub

    sat_solver = solver_cls()
    sat_solver.append_formula(clauses)

    per_iter_times = []

    # ============ 3. ITERATIVE SAT
    try:
        while True:
            # === UPPER BOUND MINIMIZATION CLAUSES
            for edge in edges:
                u, v = edge
                edge_clauses = cyclic_edge(N, X, G, u, v, UB)
                sat_solver.append_formula(edge_clauses)
                total_clauses += len(edge_clauses)

            stats["num_clauses"] = total_clauses
            if shared_stats is not None:
                shared_stats["num_clauses"] = total_clauses

            t_it_start = time.perf_counter()
            result = sat_solver.solve()
            t_it_end = time.perf_counter()
            iter_duration = t_it_end - t_it_start
            per_iter_times.append(iter_duration)
            accumulated_solve_time = sum(per_iter_times)

            if result:
                model = sat_solver.get_model()

                # === EXTRACT BANDWIDTH RESULT
                assignment = [0 for _ in range(N)]

                for lit in model[:N * N]:           # only check label variables (X_u)
                    if lit > 0:
                        u_, i_ = divmod(lit - 1, N) # divmod(lit - 1, N) returns ((lit - 1) // N, (lit - 1) % N) 
                        assignment[u_] = i_

                # Verify assignment is a valid permutation of 0..N-1
                if (
                    isinstance(assignment, (list, tuple))
                    and len(assignment) == N
                    and sorted(assignment) == list(range(N))
                ):
                    bandwidth = compute_cyclic_bandwidth(N, edges, assignment)
                    assert bandwidth < UB  # catch bugs

                    UB = bandwidth
                    verified_best_ub = bandwidth
                    if verbose:
                        print(UB)

            # Update stats after each iteration
            stats["per_iter_times"] = list(per_iter_times)
            stats["num_iterations"] = len(per_iter_times)
            stats["best_ub"] = verified_best_ub
            stats["solve_time"] = accumulated_solve_time
            if best_ub is not None and verified_best_ub != -1:
                best_ub.value = verified_best_ub

            if shared_stats is not None:
                shared_stats["per_iter_times"] = list(per_iter_times)
                shared_stats["num_iterations"] = len(per_iter_times)
                shared_stats["best_ub"] = verified_best_ub
                shared_stats["solve_time"] = accumulated_solve_time

            if not result:
                break

        if verified_best_ub != -1:
            stats["status"] = "OPTIMAL"
            if shared_stats is not None:
                shared_stats["status"] = "OPTIMAL"
        else:
            stats["status"] = "UNSAT"
            if shared_stats is not None:
                shared_stats["status"] = "UNSAT"
    except Exception as e:
        stats["status"] = "ERROR"
        if shared_stats is not None:
            shared_stats["status"] = "ERROR"
        raise e
    finally:
        sat_solver.delete()

    return SolveResult(verified_best_ub, stats)
