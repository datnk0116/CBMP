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


def test_lb_degree_all_mtx_datasets_le_known_optima():
    """Verify (a): For all datasets/*.mtx, lb_degree <= known best_ub from runs.jsonl or calibration.csv."""
    import glob
    import os
    import json
    import csv
    from analysis.make_tables import get_lb_degree

    known_optima = {}
    if os.path.exists("docs/data/runs.jsonl"):
        with open("docs/data/runs.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("status") == "OPTIMAL" and d.get("best_ub", -1) >= 0:
                    ds = d.get("dataset")
                    if ds and ds not in known_optima:
                        known_optima[ds] = d["best_ub"]

    if os.path.exists("docs/calibration/calibration.csv"):
        with open("docs/calibration/calibration.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "OPTIMAL":
                    known_optima[row["dataset"]] = int(row["best_ub"])

    mtx_files = sorted(glob.glob("datasets/*.mtx"))
    assert len(mtx_files) > 0, "No .mtx files found in datasets/"

    tested = 0
    skipped = []
    for fpath in mtx_files:
        dname = os.path.basename(fpath).replace(".mtx", "")
        lb = get_lb_degree(fpath)
        if dname in known_optima:
            opt = known_optima[dname]
            assert lb <= opt, f"Dataset {dname}: lb_degree={lb} > known optimum={opt}"
            tested += 1
        else:
            skipped.append(dname)

    assert tested >= 12, f"Expected at least 12 verified datasets, got {tested}"
    # Skipped datasets without proven optima yet:
    # Original 4 unverified datasets + 26 newly fetched Harwell-Boeing Table A.2 datasets (out of 28)
    from scripts.fetch_harwell_boeing import HARWELL_BOEING_28
    expected_skipped = {"494_bus", "494_bus_n150", "662_bus", "662_bus_n150"} | {
        name for name, _, _, _ in HARWELL_BOEING_28
    }
    assert set(skipped) == expected_skipped


def test_lb_degree_star_graphs_with_self_loops():
    """Verify (b): Star graphs K_{1,4} and K_{1,5} with diagonal entries (self-loops).
    lb_degree must be 2 and 3 respectively, matching brute-force cyclic bandwidth.
    """
    import tempfile
    import scipy.sparse as sp
    from scipy.io import mmwrite
    from analysis.make_tables import get_lb_degree

    # K_{1,4}: 5 vertices, center 0
    edges_k14 = [(0, 1), (0, 2), (0, 3), (0, 4)]
    opt_k14 = cyclic_bandwidth_bruteforce(5, edges_k14)
    # Include both directed edges and diagonal self-loops (0,0)..(4,4)
    row_k14 = [0, 0, 0, 0, 1, 2, 3, 4] + list(range(5))
    col_k14 = [1, 2, 3, 4, 0, 0, 0, 0] + list(range(5))
    mat_k14 = sp.coo_matrix(([1] * len(row_k14), (row_k14, col_k14)), shape=(5, 5))

    with tempfile.NamedTemporaryFile(suffix=".mtx") as tmp:
        mmwrite(tmp.name, mat_k14)
        lb_k14 = get_lb_degree(tmp.name)
        assert lb_k14 == 2, f"K1,4 expected lb_degree=2, got {lb_k14}"
        assert lb_k14 == opt_k14, f"K1,4 expected lb_degree == opt, got lb={lb_k14}, opt={opt_k14}"

    # K_{1,5}: 6 vertices, center 0
    edges_k15 = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    opt_k15 = cyclic_bandwidth_bruteforce(6, edges_k15)
    row_k15 = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5] + list(range(6))
    col_k15 = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0] + list(range(6))
    mat_k15 = sp.coo_matrix(([1] * len(row_k15), (row_k15, col_k15)), shape=(6, 6))

    with tempfile.NamedTemporaryFile(suffix=".mtx") as tmp:
        mmwrite(tmp.name, mat_k15)
        lb_k15 = get_lb_degree(tmp.name)
        assert lb_k15 == 3, f"K1,5 expected lb_degree=3, got {lb_k15}"
        assert lb_k15 == opt_k15, f"K1,5 expected lb_degree == opt, got lb={lb_k15}, opt={opt_k15}"


def test_lb_degree_matches_networkx_without_loops():
    """Verify (c): lb_degree == ceil(max_degree_without_loops/2) calculated independently via networkx."""
    import glob
    from scipy.io import mmread
    from analysis.make_tables import get_lb_degree

    mtx_files = sorted(glob.glob("datasets/*.mtx"))
    assert len(mtx_files) > 0

    for fpath in mtx_files:
        try:
            mat = mmread(fpath, spmatrix=True)
        except TypeError:
            mat = mmread(fpath)
        G = nx.Graph(mat)
        G.remove_edges_from(nx.selfloop_edges(G))
        max_deg = max((d for _, d in G.degree()), default=0)
        expected_lb = int(math.ceil(max_deg / 2))
        actual_lb = get_lb_degree(fpath)
        assert actual_lb == expected_lb, f"Mismatch on {fpath}: actual {actual_lb} != expected {expected_lb}"


