import os
import re
import pytest
from scipy.io import mmread
import networkx as nx

SUBGRAPH_NAMES = [
    "494_bus_n40",
    "494_bus_n50",
    "494_bus_n60",
    "494_bus_n80",
    "494_bus_n100",
    "494_bus_n150",
    "662_bus_n40",
    "662_bus_n50",
    "662_bus_n60",
    "662_bus_n80",
    "662_bus_n100",
    "662_bus_n150",
]

@pytest.mark.parametrize("name", SUBGRAPH_NAMES)
def test_subgraph_loader_connected_and_dimensions(name):
    filepath = os.path.join("datasets", f"{name}.mtx")
    assert os.path.exists(filepath), f"File {filepath} must exist"

    # Extract expected N from filename
    match = re.search(r"_n(\d+)$", name)
    assert match is not None, f"Could not parse N from {name}"
    expected_n = int(match.group(1))

    # Read using existing project loader
    mat = mmread(filepath)
    cx = mat.tocoo()
    assert cx.shape[0] == expected_n
    assert cx.shape[1] == expected_n

    edges = [(int(u), int(v)) for u, v in zip(cx.row, cx.col) if u < v]

    # Verify simple graph properties
    G = nx.Graph()
    G.add_nodes_from(range(expected_n))
    G.add_edges_from(edges)

    assert G.number_of_nodes() == expected_n
    assert len(edges) == G.number_of_edges()
    assert nx.is_connected(G), f"Subgraph {name} must be connected"

