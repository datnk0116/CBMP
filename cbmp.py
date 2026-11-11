import time
from typing import Optional, Dict, Any, Tuple, List, Union
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from sat_encodings import (
    build_order_base_formula,
    build_pairwise_base_formula,
    cyclic_edge_order,
    cyclic_edge_pairwise,
    break_rotation_symmetry,
    break_reflection_symmetry_order,
    break_reflection_symmetry_pairwise,
)
from graph_utils import resolve_dataset_path, validate_and_normalize_graph, load_graph_from_mtx
import pysat.solvers
import numpy as np

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
        yield self.ub
        yield self.stats

def compute_cyclic_bandwidth(N, edges, assignment):
    """
    Given a layout permutation assignment[u] = position of vertex u in 0..N-1,
    compute max over (u, v) in edges of cyclic_dist(assignment[u], assignment[v]).
    """
    if not edges:
        return 0
    max_d = 0
    for u, v in edges:
        p1 = assignment[u]
        p2 = assignment[v]
        diff = abs(p1 - p2)
        d = min(diff, N - diff)
        if d > max_d:
            max_d = d
    return max_d

def cyclic_distance(p1, p2, N):
    diff = abs(p1 - p2)
    return min(diff, N - diff)

def cuthill_mckee_layout(GRAPH, N):
    perm = reverse_cuthill_mckee(GRAPH.tocsr(), symmetric_mode=False)      # perm[k] = vertex placed at position k
    assignment = np.empty(N, dtype=int)
    assignment[perm] = np.arange(N)                                        # assignment[u] = position of vertex u
    return assignment

def solve(
    dataset: Optional[Union[str, int]] = None,
    sb_rotation: bool = False,
    sb_reflection: bool = False,
    use_heuristic_initial_ub: bool = False,
    best_ub: Any = None,
    N: Optional[int] = None,
    edges: Optional[List[Tuple[int, int]]] = None,
    verbose: bool = True,
    use_symmetry_break: Optional[bool] = None,
    solver: str = "cadical195",
    shared_stats: Optional[Dict[str, Any]] = None,
    relabel_seed: Optional[int] = None,
    timeout: Optional[float] = None,
    encoding: str = "order",
    solving_mode: str = "incremental",
    eo_strategy: str = "commander",
) -> SolveResult:
    """
    Solve Cyclic Bandwidth Minimization Problem using SAT solving.

    Parameters:
      dataset: Name of dataset in datasets/ or direct path to .mtx file.
      sb_rotation: Enable rotation symmetry breaking (fix anchor vertex label to 0).
      sb_reflection: Enable reflection symmetry breaking (requires sb_rotation).
      use_heuristic_initial_ub: Compute initial UB via Reverse Cuthill-McKee layout.
      best_ub: Shared multiprocessing.Value or None for incumbent tracking.
      N: Number of vertices (if providing graph explicitly).
      edges: List of undirected edges (u, v) with 0 <= u, v < N.
      verbose: Whether to print progress during search.
      use_symmetry_break: Legacy alias enabling both rotation and reflection SB.
      solver: SAT solver backend ("cadical195", "glucose4", "maplechrono", "minisat22", "lingeling").
      shared_stats: Shared dictionary for inter-process statistics reporting.
      relabel_seed: Random seed for vertex relabeling experiments.
      timeout: Optional time limit (kept for API compatibility with baseline solvers).
      encoding: SAT encoding type ("order" or "pairwise", default: "order").
      solving_mode: Search strategy ("incremental" or "iterative", default: "incremental").
      eo_strategy: Exactly-one encoding strategy ("commander", "pairwise", "binary", "sequential", "product").

    Returns:
      SolveResult containing optimal cyclic bandwidth and execution statistics dict.
    """
    # ============ RESOLVE BACKWARD COMPATIBILITY & ARGUMENTS
    if use_symmetry_break is not None:
        sb_rotation = bool(use_symmetry_break)
        sb_reflection = bool(use_symmetry_break)

    if isinstance(dataset, int) and not isinstance(dataset, bool):
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

    enc_type = str(encoding).lower().strip()
    if enc_type not in ("order", "pairwise"):
        raise ValueError(f"Unknown encoding '{encoding}'. Available: 'order', 'pairwise'")

    mode = str(solving_mode).lower().strip()
    if mode not in ("incremental", "iterative"):
        raise ValueError(f"Unknown solving_mode '{solving_mode}'. Available: 'incremental', 'iterative'")

    solver_cls = get_solver_class(solver)
    solver_name = solver if isinstance(solver, str) else getattr(solver_cls, "__name__", "custom")

    if N is not None and edges is not None:
        N, edges = validate_and_normalize_graph(N, edges)
        dataset_name = dataset if isinstance(dataset, str) else "custom"
    elif dataset is not None and not isinstance(dataset, int):
        N, edges, dataset_name = load_graph_from_mtx(dataset)
    else:
        raise ValueError("Either (N, edges) or dataset must be provided to solve.")

    if relabel_seed is not None:
        rng = np.random.default_rng(relabel_seed)
        pi = rng.permutation(N)
        new_edges = []
        for u, v in edges:
            pu, pv = int(pi[u]), int(pi[v])
            if pu != pv:
                new_edges.append((min(pu, pv), max(pu, pv)))
        edges = sorted(list(set(new_edges)))

    if edges:
        rows = [u for u, v in edges] + [v for u, v in edges]
        cols = [v for u, v in edges] + [u for u, v in edges]
        data = [1] * len(rows)
        GRAPH = csr_matrix((data, (rows, cols)), shape=(N, N))
    else:
        GRAPH = csr_matrix((N, N), dtype=int)

    stats = {
        "dataset": dataset_name,
        "N": N,
        "num_edges": len(edges),
        "relabel_seed": relabel_seed,
        "config": {
            "sb_rotation": sb_rotation,
            "sb_reflection": sb_reflection,
            "heuristic_ub": use_heuristic_initial_ub,
            "encoding": enc_type,
            "solving_mode": mode,
            "eo_strategy": eo_strategy,
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

    # ============ 1. BUILD VARIABLES & BASE FORMULA
    t_enc_start = time.perf_counter()

    if enc_type == "order":
        base_clauses, X, G, final_a = build_order_base_formula(
            N, eo_strategy=eo_strategy
        )
        if sb_rotation:
            base_clauses.extend(break_rotation_symmetry(X, 0))
            if sb_reflection and N >= 3:
                base_clauses.extend(break_reflection_symmetry_order(N, X, G))
    else:  # pairwise
        base_clauses, X, final_a = build_pairwise_base_formula(
            N, eo_strategy=eo_strategy
        )
        G = None
        if sb_rotation:
            base_clauses.extend(break_rotation_symmetry(X, 0))
            if sb_reflection and N >= 3:
                base_clauses.extend(break_reflection_symmetry_pairwise(N, X))

    t_enc_end = time.perf_counter()
    encode_time = t_enc_end - t_enc_start

    num_vars = final_a - 1
    total_clauses = len(base_clauses)

    stats["encode_time"] = encode_time
    stats["num_vars"] = num_vars
    stats["num_clauses"] = total_clauses
    if shared_stats is not None:
        shared_stats["encode_time"] = encode_time
        shared_stats["num_vars"] = num_vars
        shared_stats["num_clauses"] = total_clauses

    # ============ 2. HEURISTIC INITIAL UB
    t_heur_start = time.perf_counter()
    verified_best_ub = -1
    if use_heuristic_initial_ub:
        HEURISTIC_ASSIGNMENT = cuthill_mckee_layout(GRAPH, N)
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
        UB = N // 2 + 1  # trivial bound, cyclic dist <= N//2

    t_heur_end = time.perf_counter()
    heuristic_time = t_heur_end - t_heur_start

    stats["heuristic_time"] = heuristic_time
    stats["initial_ub"] = UB
    stats["best_ub"] = verified_best_ub

    if best_ub is not None and verified_best_ub != -1:
        best_ub.value = verified_best_ub
    if shared_stats is not None:
        shared_stats["heuristic_time"] = heuristic_time
        shared_stats["initial_ub"] = UB
        shared_stats["best_ub"] = verified_best_ub

    # Helper function to generate edge constraints for candidate UB
    def get_edge_clauses_for_ub(target_ub):
        clauses = []
        for u, v in edges:
            if enc_type == "order":
                clauses.extend(cyclic_edge_order(N, X, G, u, v, target_ub))
            else:
                clauses.extend(cyclic_edge_pairwise(N, X, u, v, target_ub))
        return clauses

    per_iter_times = []

    # ============ 3. SAT SOLVING LOOP
    if mode == "incremental":
        sat_solver = solver_cls()
        sat_solver.append_formula(base_clauses)
        try:
            while True:
                edge_clauses = get_edge_clauses_for_ub(UB)
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
                    assignment = [0 for _ in range(N)]
                    for lit in model[:N * N]:
                        if lit > 0:
                            u_, i_ = divmod(lit - 1, N)
                            assignment[u_] = i_

                    if (
                        isinstance(assignment, (list, tuple))
                        and len(assignment) == N
                        and sorted(assignment) == list(range(N))
                    ):
                        bandwidth = compute_cyclic_bandwidth(N, edges, assignment)
                        assert bandwidth < UB
                        UB = bandwidth
                        verified_best_ub = bandwidth
                        if verbose:
                            print(UB)

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

    else:  # iterative (repeated SAT with independent instances)
        try:
            while True:
                sat_solver = solver_cls()
                try:
                    iter_formula = list(base_clauses)
                    edge_clauses = get_edge_clauses_for_ub(UB)
                    iter_formula.extend(edge_clauses)
                    sat_solver.append_formula(iter_formula)

                    total_clauses = len(iter_formula)
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
                        assignment = [0 for _ in range(N)]
                        for lit in model[:N * N]:
                            if lit > 0:
                                u_, i_ = divmod(lit - 1, N)
                                assignment[u_] = i_

                        if (
                            isinstance(assignment, (list, tuple))
                            and len(assignment) == N
                            and sorted(assignment) == list(range(N))
                        ):
                            bandwidth = compute_cyclic_bandwidth(N, edges, assignment)
                            assert bandwidth < UB
                            UB = bandwidth
                            verified_best_ub = bandwidth
                            if verbose:
                                print(UB)

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
                finally:
                    sat_solver.delete()

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

    return SolveResult(verified_best_ub, stats)
