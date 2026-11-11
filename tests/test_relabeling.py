import os
import json
import pytest
import networkx as nx
from cbmp import solve
from main import main as run_main

def test_graph_atlas_relabeling_invariance():
    """
    (a) For all graphs in graph_atlas with <= 6 vertices and seeds 0..2,
    the computed optimal cyclic bandwidth remains strictly identical.
    """
    atlas = nx.graph_atlas_g()
    graphs = [g for g in atlas if 1 <= g.number_of_nodes() <= 6]
    assert len(graphs) == 208

    for g in graphs:
        n = g.number_of_nodes()
        edges = list(g.edges())
        base_res = solve(N=n, edges=edges, sb_rotation=True, sb_reflection=True, use_heuristic_initial_ub=True, verbose=False, relabel_seed=None)
        for s in (0, 1, 2):
            s_res = solve(N=n, edges=edges, sb_rotation=True, sb_reflection=True, use_heuristic_initial_ub=True, verbose=False, relabel_seed=s)
            assert s_res.ub == base_res.ub, f"Mismatch for graph with N={n}, |E|={len(edges)}: seed {s} gave {s_res.ub}, base gave {base_res.ub}"

def test_30_bus_seeds_optimum():
    """
    (b) 30_bus across seeds 0..4 always achieves optimum = 4.
    """
    for s in range(5):
        res = solve("30_bus", sb_rotation=True, sb_reflection=True, use_heuristic_initial_ub=True, verbose=False, relabel_seed=s)
        assert res.ub == 4, f"30_bus seed {s} gave optimum {res.ub}, expected 4"

def test_record_has_relabel_seed(tmp_path):
    """
    (c) Resulting stats record and json output contain relabel_seed field.
    """
    res_none = solve("14_bus", verbose=False, relabel_seed=None)
    assert "relabel_seed" in res_none.stats
    assert res_none.stats["relabel_seed"] is None

    res_0 = solve("14_bus", verbose=False, relabel_seed=0)
    assert "relabel_seed" in res_0.stats
    assert res_0.stats["relabel_seed"] == 0

    json_file = str(tmp_path / "stats.jsonl")
    final_stats = run_main("14_bus", sb_rotation=True, sb_reflection=False, use_heuristic_initial_ub=False, timeout=10, json_out=json_file, relabel_seed=42)
    assert "relabel_seed" in final_stats
    assert final_stats["relabel_seed"] == 42

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.loads(f.readline())
        assert data.get("relabel_seed") == 42

