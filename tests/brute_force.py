import itertools

def cyclic_bandwidth_bruteforce(N, edges):
    """
    Computes the optimal cyclic bandwidth for an undirected graph with N vertices
    labeled 0..N-1 and a list of edges (u, v) by iterating over permutations (N <= 9).
    """
    if not edges or N <= 1:
        return 0

    best_bw = N
    for p in itertools.permutations(range(N)):
        bw = 0
        for u, v in edges:
            diff = abs(p[u] - p[v])
            d = diff if diff <= N - diff else N - diff
            if d > bw:
                bw = d
                if bw >= best_bw:
                    break
        if bw < best_bw:
            best_bw = bw
            if best_bw <= 1:
                break

    return best_bw

