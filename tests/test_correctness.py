import random
import pytest
from cbmp import solve
from tests.brute_force import cyclic_bandwidth_bruteforce

def generate_graphs():
    graphs = []

    # 1. Path graphs (P_N), Cycle graphs (C_N), Complete graphs (K_N) for N = 4..9
    for N in range(4, 10):
        path_edges = [(i, i + 1) for i in range(N - 1)]
        graphs.append((f"path_N{N}", N, path_edges))

        cycle_edges = path_edges + [(0, N - 1)]
        graphs.append((f"cycle_N{N}", N, cycle_edges))

        clique_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        graphs.append((f"clique_N{N}", N, clique_edges))

    # 2. 30 random graphs with fixed seed
    rng = random.Random(42)
    for idx in range(30):
        N = rng.randint(4, 9)
        p = rng.uniform(0.15, 0.75)
        edges = [(u, v) for u in range(N) for v in range(u + 1, N) if rng.random() < p]
        graphs.append((f"random_{idx:02d}_N{N}", N, edges))

    return graphs

ALL_GRAPHS = generate_graphs()
# Precompute brute-force ground truth for speed
GROUND_TRUTH = {
    name: cyclic_bandwidth_bruteforce(N, edges)
    for name, N, edges in ALL_GRAPHS
}

# 3 symmetry configurations: none, rotation only, rotation + reflection
SYMMETRY_MODES = [
    (False, False),  # no symmetry breaking
    (True, False),   # only rotation
    (True, True),    # rotation + reflection
]

HEURISTIC_MODES = [False, True]

CONFIGS = [
    (sb_rot, sb_ref, heur)
    for sb_rot, sb_ref in SYMMETRY_MODES
    for heur in HEURISTIC_MODES
]

CONFIG_IDS = [
    f"rot={sb_rot},ref={sb_ref},heur={heur}"
    for sb_rot, sb_ref, heur in CONFIGS
]

@pytest.mark.parametrize("graph_info", ALL_GRAPHS, ids=[g[0] for g in ALL_GRAPHS])
@pytest.mark.parametrize("sb_rot,sb_ref,heur", CONFIGS, ids=CONFIG_IDS)
def test_sat_matches_bruteforce(graph_info, sb_rot, sb_ref, heur):
    name, N, edges = graph_info
    expected = GROUND_TRUTH[name]

    actual = solve(
        N=N,
        edges=edges,
        sb_rotation=sb_rot,
        sb_reflection=sb_ref,
        use_heuristic_initial_ub=heur,
        verbose=False,
    )

    assert actual == expected, (
        f"Mismatch on {name} with sb_rotation={sb_rot}, sb_reflection={sb_ref}, heur={heur}: "
        f"expected {expected}, got {actual} (edges={edges})"
    )

def test_reflection_without_rotation_raises_error():
    # Reflection symmetry break must require rotation symmetry break
    with pytest.raises(ValueError, match="requires --sb-rotation"):
        solve(
            N=5,
            edges=[(0, 1), (1, 2), (2, 3)],
            sb_rotation=False,
            sb_reflection=True,
            verbose=False,
        )

def test_symmetry_break_alias():
    # --symmetry-break alias enables both rotation and reflection
    N = 6
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    expected = GROUND_TRUTH["cycle_N6"]
    res = solve(N=N, edges=edges, use_symmetry_break=True, verbose=False)
    assert res == expected

def test_positional_call_interface():
    # Test that solve(N, edges, sb_rot, sb_ref, heur) works positionally
    N = 5
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    expected = GROUND_TRUTH["cycle_N5"]
    res = solve(N, edges, True, False, False)
    assert res == expected

@pytest.mark.parametrize("solver_name", ["cadical195", "glucose4", "maplechrono", "minisat22", "lingeling"])
def test_all_solvers_yield_same_optimum(solver_name):
    N = 6
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    expected = GROUND_TRUTH["cycle_N6"]
    res = solve(N=N, edges=edges, solver=solver_name, verbose=False)
    assert res == expected

def test_solve_returns_stats_dict():
    N = 5
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    ub, stats = solve(N=N, edges=edges, sb_rotation=True, sb_reflection=True, verbose=False)

    required_keys = [
        "dataset", "N", "num_edges", "config", "solver", "status",
        "best_ub", "encode_time", "solve_time", "per_iter_times",
        "num_iterations", "num_vars", "num_clauses"
    ]
    for key in required_keys:
        assert key in stats, f"Missing key in stats: {key}"

    assert stats["status"] == "OPTIMAL"
    assert stats["best_ub"] == ub
    assert isinstance(stats["per_iter_times"], list)
    assert stats["num_iterations"] == len(stats["per_iter_times"])

def test_invalid_solver_raises_error():
    with pytest.raises(ValueError, match="Unknown or unavailable solver"):
        solve(N=4, edges=[(0, 1)], solver="non_existent_solver", verbose=False)

def test_best_ub_not_recorded_without_assignment(monkeypatch):
    import multiprocessing as mp
    from pysat.solvers import Cadical195

    # Simulate timeout/failure before any SAT iteration finishes
    def fake_solve(self):
        raise TimeoutError("Simulated timeout before SAT solution")

    monkeypatch.setattr(Cadical195, "solve", fake_solve)

    best_ub = mp.Value("i", -1)
    shared_stats = mp.Manager().dict()

    with pytest.raises(TimeoutError):
        solve(
            dataset="30_bus",
            use_heuristic_initial_ub=False,
            best_ub=best_ub,
            shared_stats=shared_stats,
            verbose=False,
        )

    # best_ub should NOT be recorded as N//2 + 1; it must remain -1
    assert best_ub.value == -1
    assert shared_stats.get("best_ub") == -1
    assert shared_stats.get("initial_ub") == 30 // 2 + 1

def test_heuristic_ub_recorded_only_when_verified(monkeypatch):
    import multiprocessing as mp
    from pysat.solvers import Cadical195

    def fake_solve(self):
        raise TimeoutError("Simulated timeout before SAT solution")

    monkeypatch.setattr(Cadical195, "solve", fake_solve)

    best_ub = mp.Value("i", -1)
    shared_stats = mp.Manager().dict()

    with pytest.raises(TimeoutError):
        solve(
            dataset="30_bus",
            use_heuristic_initial_ub=True,
            best_ub=best_ub,
            shared_stats=shared_stats,
            verbose=False,
        )

    # Heuristic assignment was verified, so best_ub and initial_ub match the heuristic bandwidth
    assert best_ub.value > 0
    assert shared_stats.get("best_ub") == best_ub.value
    assert shared_stats.get("initial_ub") == best_ub.value
