#!/usr/bin/env python3
"""
scripts/make_subgraphs.py

Generate deterministic, connected subgraphs of datasets/494_bus.mtx and datasets/662_bus.mtx
using BFS from the highest-degree vertex (tie-break by smallest index).

Subgraphs are generated for N in [40, 60, 80, 100, 150].
Vertices are relabeled 0..N-1 according to BFS discovery order.
Files are saved as symmetric MatrixMarket format in datasets/.
"""

import os
from collections import deque
from pathlib import Path
from scipy.io import mmread
import networkx as nx

def load_graph_from_mtx(filepath):
    """Load an undirected simple graph from an MTX file."""
    mat = mmread(filepath)
    cx = mat.tocoo()
    G = nx.Graph()
    for u, v in zip(cx.row, cx.col):
        u_int = int(u)
        v_int = int(v)
        if u_int != v_int:
            G.add_edge(u_int, v_int)
    return G

def extract_bfs_subgraph(G, n):
    """
    Extract deterministic connected subgraph of size n via BFS from highest degree node.
    - Start node: highest degree (tie-break: smallest node index).
    - Neighbors visited in ascending order of node index.
    - Relabels nodes 0..n-1 in BFS order.
    Returns (relabeled_edges, n, original_nodes_in_order).
    """
    degrees = dict(G.degree())
    max_deg = max(degrees.values())
    candidates = [node for node, deg in degrees.items() if deg == max_deg]
    start_node = min(candidates)

    visited = []
    seen = {start_node}
    queue = deque([start_node])

    while queue and len(visited) < n:
        curr = queue.popleft()
        visited.append(curr)
        neighbors = sorted(list(G.neighbors(curr)))
        for nbr in neighbors:
            if nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)
                if len(visited) + len(queue) >= n and len(visited) < n:
                    pass

    # In case queue was exhausted before n (graph smaller or disconnected component)
    if len(visited) < n:
        raise ValueError(f"Could not reach {n} vertices from start node {start_node} (only {len(visited)} reached)")

    selected_nodes = visited[:n]
    node_to_new_idx = {orig: idx for idx, orig in enumerate(selected_nodes)}

    subG = G.subgraph(set(selected_nodes))
    if not nx.is_connected(subG):
        raise ValueError(f"Extracted subgraph of size {n} is not connected!")

    relabeled_edges = set()
    for u, v in subG.edges():
        new_u = node_to_new_idx[u]
        new_v = node_to_new_idx[v]
        if new_u != new_v:
            relabeled_edges.add((min(new_u, new_v), max(new_u, new_v)))

    sorted_edges = sorted(list(relabeled_edges))
    return sorted_edges, n, selected_nodes

def write_mtx_symmetric(filepath, n, edges):
    """
    Write edges and diagonal self-loops in symmetric MatrixMarket coordinate format.
    Format is 1-based, lower triangular (row >= col), sorted by (col, row).
    """
    entries = []
    # 1. Edge entries (row > col)
    for u, v in edges:
        row = max(u, v) + 1
        col = min(u, v) + 1
        entries.append((row, col, 1.0))

    # 2. Diagonal entries (row == col)
    for i in range(1, n + 1):
        entries.append((i, i, 1.0))

    # Sort by col ascending, then row ascending (standard MTX coordinate order)
    entries.sort(key=lambda x: (x[1], x[0]))

    total_entries = len(entries)
    lines = [
        "%%MatrixMarket matrix coordinate real symmetric",
        f"{n} {n} {total_entries}",
    ]
    for row, col, val in entries:
        lines.append(f"{row} {col} {val:.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def make_all_subgraphs(output_dir="datasets", sizes=(40, 60, 80, 100, 150)):
    """Generate subgraphs for 494_bus and 662_bus."""
    base_graphs = [
        ("494_bus", os.path.join(output_dir, "494_bus.mtx")),
        ("662_bus", os.path.join(output_dir, "662_bus.mtx")),
    ]

    generated_files = []
    for base_name, base_path in base_graphs:
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Source graph not found: {base_path}")
        G = load_graph_from_mtx(base_path)

        for n in sizes:
            out_filename = f"{base_name}_n{n}.mtx"
            out_path = os.path.join(output_dir, out_filename)
            edges, n_out, _ = extract_bfs_subgraph(G, n)
            write_mtx_symmetric(out_path, n_out, edges)
            generated_files.append((out_filename, n_out, len(edges)))
            print(f"Generated {out_filename}: N={n_out}, |E|={len(edges)} (connected)")

    return generated_files

if __name__ == "__main__":
    make_all_subgraphs()

