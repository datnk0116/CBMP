import pytest
import networkx as nx
from tests.brute_force import cyclic_bandwidth_bruteforce
from cbmp import solve
from baselines.cpsat_cbmp import solve_cpsat

def get_atlas_graphs(min_n=1, max_n=7):
    atlas = nx.graph_atlas_g()
    return [g for g in atlas if min_n <= g.number_of_nodes() <= max_n]

def test_graph_atlas_small_exhaustive():
    """
    Fast exhaustive verification on all undirected graphs with 1..5 nodes (52 graphs).
    Compares brute force with:
    (a) SAT cadical195 (rotation + reflection, heuristic off)
    (b) SAT cadical195 (rotation + reflection, heuristic on)
    (c) CP-SAT
    (d) CPLEX (if available)
    """
    graphs = get_atlas_graphs(1, 5)
    assert len(graphs) == 52

    try:
        import cplex
        from baselines.cplex_cbmp import solve_cplex
        has_cplex = True
    except ImportError:
        has_cplex = False

    for g in graphs:
        N = g.number_of_nodes()
        edges = list(g.edges())
        expected = cyclic_bandwidth_bruteforce(N, edges)

        # SAT heuristic=False
        sat_no_heur = solve(
            N=N,
            edges=edges,
            sb_rotation=True,
            sb_reflection=True,
            use_heuristic_initial_ub=False,
            solver="cadical195",
            verbose=False,
        )
        assert sat_no_heur == expected

        # SAT heuristic=True
        sat_heur = solve(
            N=N,
            edges=edges,
            sb_rotation=True,
            sb_reflection=True,
            use_heuristic_initial_ub=True,
            solver="cadical195",
            verbose=False,
        )
        assert sat_heur == expected

        # CP-SAT
        cpsat_res, _ = solve_cpsat(N=N, edges=edges, timeout=10.0, verbose=False)
        assert cpsat_res == expected

        # CPLEX
        if has_cplex:
            cpx_res, _ = solve_cplex(N=N, edges=edges, timeout=10.0, verbose=False)
            assert cpx_res == expected

@pytest.mark.slow
def test_graph_atlas_exhaustive_sat():
    """
    Exhaustive verification of SAT cadical195 (heur off & on)
    across all 1252 undirected graphs (1..7 nodes) from Graph Atlas against brute force.
    """
    graphs = get_atlas_graphs(1, 7)
    assert len(graphs) == 1252

    for g in graphs:
        N = g.number_of_nodes()
        edges = list(g.edges())
        expected = cyclic_bandwidth_bruteforce(N, edges)

        res_no_heur = solve(
            N=N,
            edges=edges,
            sb_rotation=True,
            sb_reflection=True,
            use_heuristic_initial_ub=False,
            solver="cadical195",
            verbose=False,
        )
        assert res_no_heur == expected

        res_heur = solve(
            N=N,
            edges=edges,
            sb_rotation=True,
            sb_reflection=True,
            use_heuristic_initial_ub=True,
            solver="cadical195",
            verbose=False,
        )
        assert res_heur == expected

@pytest.mark.slow
def test_graph_atlas_exhaustive_cpsat():
    """
    Exhaustive verification of OR-Tools CP-SAT baseline
    across all 1252 undirected graphs (1..7 nodes) from Graph Atlas against brute force.
    """
    graphs = get_atlas_graphs(1, 7)
    assert len(graphs) == 1252

    for g in graphs:
        N = g.number_of_nodes()
        edges = list(g.edges())
        expected = cyclic_bandwidth_bruteforce(N, edges)

        res, stats = solve_cpsat(N=N, edges=edges, timeout=10.0, verbose=False)
        assert res == expected
        assert stats["status"] == "OPTIMAL"

@pytest.mark.slow
def test_graph_atlas_exhaustive_cplex():
    """
    Exhaustive verification of IBM CPLEX baseline
    across all 1252 undirected graphs (1..7 nodes) from Graph Atlas against brute force.
    """
    pytest.importorskip("cplex")
    from baselines.cplex_cbmp import solve_cplex

    graphs = get_atlas_graphs(1, 7)
    assert len(graphs) == 1252

    for g in graphs:
        N = g.number_of_nodes()
        edges = list(g.edges())
        expected = cyclic_bandwidth_bruteforce(N, edges)

        res, stats = solve_cplex(N=N, edges=edges, timeout=10.0, verbose=False)
        assert res == expected
        assert stats["status"] == "OPTIMAL"

