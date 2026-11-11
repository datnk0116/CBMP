import math
import pytest
import networkx as nx
from tests.brute_force import cyclic_bandwidth_bruteforce
from baselines.cpsat_cbmp import solve_cpsat

def compute_lb_degree_from_edges(edges):
    if not edges:
        return 0
    deg = {}
    for u, v in edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    max_d = max(deg.values()) if deg else 0
    return math.ceil(max_d / 2)

def test_lb_degree_le_optimum_graph_atlas():
    """Verify that lb_degree = ceil(Delta/2) <= optimum on all graphs in atlas with <= 7 vertices."""
    atlas = nx.graph_atlas_g()
    tested = 0
    for g in atlas:
        n = g.number_of_nodes()
        if n < 2 or n > 7:
            continue
        edges = [(u, v) for u, v in g.edges()]
        if not edges:
            continue
        lb = compute_lb_degree_from_edges(edges)
        opt = cyclic_bandwidth_bruteforce(n, edges)
        assert lb <= opt, f"Graph atlas {g.name} (N={n}) violated lower bound: lb={lb} > opt={opt}"
        tested += 1
    assert tested >= 1000

TEST_SMALL_GRAPHS = [
    ("path_4", 4, [(0, 1), (1, 2), (2, 3)]),
    ("cycle_4", 4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
    ("clique_4", 4, [(u, v) for u in range(4) for v in range(u + 1, 4)]),
    ("path_5", 5, [(i, i + 1) for i in range(4)]),
    ("cycle_5", 5, [(i, i + 1) for i in range(4)] + [(0, 4)]),
    ("clique_5", 5, [(u, v) for u in range(5) for v in range(u + 1, 5)]),
    ("path_6", 6, [(i, i + 1) for i in range(5)]),
    ("cycle_6", 6, [(i, i + 1) for i in range(5)] + [(0, 5)]),
]

@pytest.mark.parametrize("name, N, edges", TEST_SMALL_GRAPHS)
def test_cpsat_optimal_lb_equals_ub(name, N, edges):
    """Verify that CP-SAT returns best_lb == best_ub when OPTIMAL."""
    res = solve_cpsat(N=N, edges=edges, timeout=10.0, verbose=False)
    assert res.stats["status"] == "OPTIMAL"
    assert res.stats.get("best_lb") is not None
    assert res.stats["best_lb"] == res.stats["best_ub"]
