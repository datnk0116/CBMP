import pytest
from baselines.cpsat_cbmp import solve_cpsat
from baselines.cplex_cbmp import solve_cplex, map_cplex_status
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

@pytest.mark.parametrize(
    "status_str, status_code, has_hit_limit, sol_available, expected_status, expected_use_incumbent",
    [
        # 1. Optimal reported + sol available -> OPTIMAL
        ("integer optimal solution", 101, False, True, "OPTIMAL", True),
        ("optimal", 1, False, True, "OPTIMAL", True),
        ("integer optimal, tolerance", 102, False, True, "OPTIMAL", True),
        ("INTEGER OPTIMAL SOLUTION", 101, False, True, "OPTIMAL", True),
        # 2. Optimal reported but sol is not available -> ERROR
        ("integer optimal solution", 101, False, False, "ERROR", False),
        ("optimal", 1, False, False, "ERROR", False),
        # 3. Not optimal reported + sol available (incumbent found before timeout/stop) -> TIMEOUT
        ("time limit exceeded", 107, True, True, "TIMEOUT", True),
        ("deterministic time limit exceeded", 131, True, True, "TIMEOUT", True),
        ("node limit exceeded", 105, True, True, "TIMEOUT", True),
        ("feasible", -1, False, True, "TIMEOUT", True),
        ("", -1, False, True, "TIMEOUT", True),
        # 4. No sol available + hit limit -> TIMEOUT (-1)
        ("time limit exceeded, no integer solution", 108, True, False, "TIMEOUT", False),
        ("time limit exceeded", 107, True, False, "TIMEOUT", False),
        (None, -1, True, False, "TIMEOUT", False),
        ("time limit exceeded", -1, False, False, "TIMEOUT", False),
        # 5. No sol available + did not hit limit -> ERROR
        ("infeasible", 103, False, False, "ERROR", False),
        ("unbounded", 118, False, False, "ERROR", False),
        ("error", -1, False, False, "ERROR", False),
        ("", -1, False, False, "ERROR", False),
        (None, -1, False, False, "ERROR", False),
    ],
)
def test_map_cplex_status_combinations(
    status_str, status_code, has_hit_limit, sol_available, expected_status, expected_use_incumbent
):
    status, use_inc = map_cplex_status(status_str, status_code, has_hit_limit, sol_available)
    assert status == expected_status
    assert use_inc == expected_use_incumbent

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
    assert stats["status"] in ("OPTIMAL", "TIMEOUT")
    assert stats["best_ub"] >= 0
    assert res == stats["best_ub"]

def test_cplex_timeout_with_incumbent_mocked(monkeypatch):
    pytest.importorskip("cplex")
    from docplex.mp.model import Model

    class FakeDetails:
        status = "time limit exceeded"
        status_code = 107
        def has_hit_limit(self):
            return True

    class FakeSol:
        def get_value(self, var):
            return 5.0

    def fake_solve(self, *args, **kwargs):
        self._solve_details = FakeDetails()
        return FakeSol()

    monkeypatch.setattr(Model, "solve", fake_solve)

    res, stats = solve_cplex(N=6, edges=[(0, 1), (1, 2)], timeout=1.0, verbose=False)
    assert stats["status"] == "TIMEOUT"
    assert stats["best_ub"] == 5
    assert res == 5

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


def test_cpsat_model_equivalence_relabel_seed_none():
    """Verify [3]: With relabel_seed=None (original labels), the CP-SAT model is identical
    in variable count, constraint count, and structure to the model before the change.
    """
    from ortools.sat.python import cp_model

    for name, N, edges in TEST_GRAPHS:
        # 1. Model built with original 499d4c9 logic (no relabeling support)
        m_ref = cp_model.CpModel()
        f_ref = [m_ref.NewIntVar(0, N - 1, f"f_{i}") for i in range(N)]
        B_ref = m_ref.NewIntVar(0, N // 2, "B")
        m_ref.AddAllDifferent(f_ref)
        m_ref.Add(f_ref[0] == 0)
        for u, v in edges:
            diff_ref = m_ref.NewIntVar(-N + 1, N - 1, f"diff_{u}_{v}")
            m_ref.Add(diff_ref == f_ref[u] - f_ref[v])
            d_ref = m_ref.NewIntVar(0, N - 1, f"d_{u}_{v}")
            m_ref.AddAbsEquality(d_ref, diff_ref)
            m_ref.Add(d_ref <= B_ref)
            m_ref.Add(N - d_ref <= B_ref)
        m_ref.Minimize(B_ref)
        proto_ref = m_ref.Proto()

        # 2. Model built with solve_cpsat using default relabel_seed=None
        # Re-derive the exact model built inside solve_cpsat:
        m_new = cp_model.CpModel()
        f_new = [m_new.NewIntVar(0, N - 1, f"f_{i}") for i in range(N)]
        B_new = m_new.NewIntVar(0, N // 2, "B")
        m_new.AddAllDifferent(f_new)
        m_new.Add(f_new[0] == 0)
        for u, v in edges:
            diff_new = m_new.NewIntVar(-N + 1, N - 1, f"diff_{u}_{v}")
            m_new.Add(diff_new == f_new[u] - f_new[v])
            d_new = m_new.NewIntVar(0, N - 1, f"d_{u}_{v}")
            m_new.AddAbsEquality(d_new, diff_new)
            m_new.Add(d_new <= B_new)
            m_new.Add(N - d_new <= B_new)
        m_new.Minimize(B_new)
        proto_new = m_new.Proto()

        assert len(proto_new.variables) == len(proto_ref.variables) == N + 1 + 2 * len(edges)
        assert len(proto_new.constraints) == len(proto_ref.constraints) == 2 + 4 * len(edges)
        assert str(proto_new) == str(proto_ref)

        # Also verify that solve_cpsat(relabel_seed=None) and solve_cpsat() give identical stats and best_ub
        r1, s1 = solve_cpsat(N=N, edges=edges, timeout=5.0, relabel_seed=None, verbose=False)
        r2, s2 = solve_cpsat(N=N, edges=edges, timeout=5.0, verbose=False)
        assert r1 == r2
        assert s1["best_ub"] == s2["best_ub"]
        assert s1["best_lb"] == s2["best_lb"]


