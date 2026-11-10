import pytest
from baselines.cpsat_cbmp import solve_cpsat
from baselines.cplex_cbmp import solve_cplex
from tests.brute_force import cyclic_bandwidth_bruteforce

TEST_GRAPHS = [
    ("path_4", 4, [(0, 1), (1, 2), (2, 3)]),
    ("cycle_4", 4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
    ("clique_4", 4, [(u, v) for u in range(4) for v in range(u + 1, 4)]),
    ("path_5", 5, [(i, i + 1) for i in range(4)]),
    ("cycle_5", 5, [(i, i + 1) for i in range(4)] + [(0, 4)]),
    ("clique_5", 5, [(u, v) for u in range(5) for v in range(u + 1, 5)]),
    ("path_6", 6, [(i, i + 1) for i in range(5)]),
    ("cycle_6", 6, [(i, i + 1) for i in range(5)] + [(0, 5)]),
    ("rnd_6", 6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 3), (1, 4)]),
]

@pytest.mark.parametrize("name, N, edges", TEST_GRAPHS)
def test_cpsat_matches_bruteforce(name, N, edges):
    expected = cyclic_bandwidth_bruteforce(N, edges)
    actual, stats = solve_cpsat(N=N, edges=edges, timeout=10.0, verbose=False)
    assert actual == expected
    assert stats["status"] == "OPTIMAL"
    assert stats["solver"] == "cp_sat"

@pytest.mark.parametrize("name, N, edges", TEST_GRAPHS)
def test_cplex_matches_bruteforce(name, N, edges):
    pytest.importorskip("cplex")
    expected = cyclic_bandwidth_bruteforce(N, edges)
    actual, stats = solve_cplex(N=N, edges=edges, timeout=10.0, verbose=False)
    assert actual == expected
    assert stats["status"] == "OPTIMAL"
    assert stats["solver"] == "cplex"

def test_cplex_timeout_with_incumbent():
    pytest.importorskip("cplex")
    import random
    random.seed(42)
    N = 12
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if random.random() < 0.4:
                edges.append((u, v))
    res, stats = solve_cplex(N=N, edges=edges, timeout=0.05, verbose=False)
    assert stats["status"] == "TIMEOUT"
    assert stats["best_ub"] >= 4
    assert res == stats["best_ub"]

def test_cplex_size_limit_handling():
    pytest.importorskip("cplex")
    from docplex.mp.model import Model
    import docplex.mp.utils as utils

    # Check if Community Edition limits (1000 vars/constraints) are present
    m_probe = Model("probe_community_limit")
    xs = [m_probe.binary_var(name=f"x_{i}") for i in range(1005)]
    m_probe.maximize(m_probe.sum(xs))
    is_community = False
    try:
        m_probe.solve(log_output=False)
    except utils.DOcplexLimitsExceeded:
        is_community = True
    except Exception:
        pass

    if not is_community:
        pytest.skip("Full CPLEX edition detected (Community Edition 1000 size limit not active)")

    # 30_bus has 1018 variables, exceeding the 1000 limit of Community Edition
    res, stats = solve_cplex(dataset="30_bus", timeout=5.0, verbose=False)
    assert stats["status"] == "SIZE_LIMIT"
    assert res == -1

def test_cpsat_on_30_bus():
    # CP-SAT should easily solve 30_bus
    res, stats = solve_cpsat(dataset="30_bus", timeout=10.0, verbose=False)
    assert stats["status"] == "OPTIMAL"
    assert res == 4

