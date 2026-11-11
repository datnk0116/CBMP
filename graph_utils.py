import os
from pathlib import Path
from typing import Tuple, List, Optional, Any
from scipy.io import mmread
from scipy.sparse import issparse

def resolve_dataset_path(dataset: Any, dataset_dir: str = "datasets") -> Tuple[str, str]:
    """
    Resolve a dataset name or filepath to a valid existing file path and canonical name.

    Supports:
      - '30_bus' -> 'datasets/30_bus.mtx', '30_bus'
      - '30_bus.mtx' -> 'datasets/30_bus.mtx', '30_bus'
      - 'datasets/30_bus.mtx' -> 'datasets/30_bus.mtx', '30_bus'
      - '/abs/path/to/30_bus.mtx' -> '/abs/path/to/30_bus.mtx', '30_bus'

    Raises:
      FileNotFoundError: If the file cannot be located.
    """
    if dataset is None or isinstance(dataset, bool) or isinstance(dataset, int):
        raise ValueError(f"Invalid dataset specifier: {dataset!r}")

    dataset_str = str(dataset).strip()
    p = Path(dataset_str)

    candidates = [
        p,
        Path(f"{dataset_str}.mtx"),
        Path(dataset_dir) / p.name,
        Path(dataset_dir) / f"{p.name}.mtx" if not p.name.endswith(".mtx") else Path(dataset_dir) / p.name,
        Path(dataset_dir) / p,
    ]

    for cand in candidates:
        if cand.is_file():
            canonical_name = cand.stem
            return str(cand), canonical_name

    raise FileNotFoundError(
        f"Dataset file not found for '{dataset}'. Searched at '{dataset_str}', '{dataset_str}.mtx', and within '{dataset_dir}/'."
    )

def validate_and_normalize_graph(N: Any, edges: Any) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Validate graph dimensions and edges, ensuring undirected representation.

    Rules:
      - N must be a positive integer (booleans are strictly rejected).
      - Each edge must be a length-2 sequence (u, v) with integer vertices in [0, N-1].
      - Self-loops (u == v) have cyclic distance 0 under any layout and are ignored.
      - Edges are normalized to (min(u, v), max(u, v)), deduplicated, and sorted.

    Returns:
      (N, normalized_edges)
    """
    if isinstance(N, bool) or not isinstance(N, int):
        raise TypeError(f"N must be an integer, got {N!r} of type {type(N).__name__}")
    if N <= 0:
        raise ValueError(f"N must be a positive integer, got {N}")

    if edges is None:
        return N, []

    if isinstance(edges, (str, bytes)):
        raise TypeError(f"edges must be an iterable of pairs, got {type(edges).__name__}")

    try:
        edge_iter = iter(edges)
    except TypeError:
        raise TypeError(f"edges must be an iterable, got {type(edges).__name__}")

    norm_edges = set()
    for idx, e in enumerate(edge_iter):
        if not isinstance(e, (tuple, list)) or len(e) != 2:
            raise ValueError(f"Edge at index {idx} must be a pair (u, v), got {e!r}")
        u, v = e[0], e[1]
        if isinstance(u, bool) or not isinstance(u, int) or isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"Edge vertices at index {idx} must be integers, got {e!r}")
        if u < 0 or u >= N or v < 0 or v >= N:
            raise ValueError(f"Vertex index out of bounds [0, {N-1}] in edge {e!r}")
        if u == v:
            # Policy: self-loops do not constrain cyclic bandwidth, ignore
            continue
        norm_edges.add((min(u, v), max(u, v)))

    return N, sorted(list(norm_edges))

def load_graph_from_mtx(filepath_or_name: str, dataset_dir: str = "datasets") -> Tuple[int, List[Tuple[int, int]], str]:
    """
    Load, validate, and extract an undirected graph from a MatrixMarket (.mtx) file.

    Validations:
      - File exists and is readable.
      - Matrix must be square.
      - Nonzero adjacency structure must be symmetric.
      - Diagonal elements (self-loops) are ignored.

    Returns:
      (N, edges, dataset_name)
    """
    filepath, dataset_name = resolve_dataset_path(filepath_or_name, dataset_dir=dataset_dir)
    mat = mmread(filepath)

    if not hasattr(mat, "shape") or len(mat.shape) != 2:
        raise ValueError(f"MatrixMarket content must be a 2D matrix, got shape {getattr(mat, 'shape', None)}")

    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"MatrixMarket matrix must be square, got non-square shape {mat.shape}")

    N = int(mat.shape[0])

    if issparse(mat):
        csr = mat.tocsr()
        rows, cols = csr.nonzero()
    else:
        import numpy as np
        rows, cols = np.nonzero(mat)

    # Filter out diagonal entries (self-loops)
    non_diag_mask = (rows != cols)
    r_nd = rows[non_diag_mask]
    c_nd = cols[non_diag_mask]

    # Validate symmetry of non-zero structure
    pairs = set(zip(r_nd, c_nd))
    for u, v in pairs:
        if (v, u) not in pairs:
            raise ValueError(
                f"MatrixMarket matrix in '{filepath}' is not symmetric for an undirected graph: entry ({u}, {v}) is present but ({v}, {u}) is missing."
            )

    # Normalize to undirected edges u < v
    edges = sorted(list(set((min(int(u), int(v)), max(int(u), int(v))) for u, v in pairs)))
    return N, edges, dataset_name

def compute_cyclic_bandwidth(N: int, edges: Any, assignment: Any) -> int:
    """
    Compute cyclic bandwidth of graph (N, edges) under the given assignment:
      CB(G, f) = max_{(u, v) in E} min(|f(u) - f(v)|, N - |f(u) - f(v)|)
    """
    if not edges:
        return 0
    max_d = 0
    for u, v in edges:
        d = abs(int(assignment[u]) - int(assignment[v]))
        cd = min(d, N - d)
        if cd > max_d:
            max_d = cd
    return max_d

def validate_solution(
    N: int,
    edges: Any,
    assignment: Any,
    reported_objective: Optional[int] = None,
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Independently validate a CBMP candidate solution assignment.

    Checks:
      1. Assignment is a sequence of length N.
      2. Every assigned label belongs to the valid label domain [0, N-1].
      3. All labels are unique (permutation property).
      4. Every edge uses valid vertex indices in [0, N-1].
      5. The independently recomputed cyclic bandwidth matches reported_objective (if provided).

    Returns:
      (is_valid: bool, recomputed_bandwidth: Optional[int], error_message: Optional[str])
    """
    if not isinstance(assignment, (list, tuple)):
        return False, None, f"Assignment must be a list or tuple, got {type(assignment).__name__}"

    if len(assignment) != N:
        return False, None, f"Assignment length {len(assignment)} does not match vertex count N={N}"

    for u, label in enumerate(assignment):
        if isinstance(label, bool) or not isinstance(label, int):
            return False, None, f"Label for vertex {u} must be an integer, got {label!r}"
        if label < 0 or label >= N:
            return False, None, f"Label {label} for vertex {u} out of valid domain [0, {N-1}]"

    if len(set(assignment)) != N:
        return False, None, "Assignment contains duplicate labels: not an injective permutation"

    if edges:
        for u, v in edges:
            if u < 0 or u >= N or v < 0 or v >= N:
                return False, None, f"Edge ({u}, {v}) references out-of-bounds vertex for N={N}"

    actual_bw = compute_cyclic_bandwidth(N, edges, assignment)

    if reported_objective is not None:
        if int(reported_objective) != actual_bw:
            return False, actual_bw, (
                f"Reported objective {reported_objective} does not match recomputed cyclic bandwidth {actual_bw}"
            )

    return True, actual_bw, None

