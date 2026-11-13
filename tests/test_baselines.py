import pytest
from baselines.cpsat_cbmp import solve_cpsat, build_cpsat_model
from baselines.cplex_cbmp import solve_cplex, map_cplex_status, build_cplex_model
from cbmp import solve
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
        ("node limit exceeded", 105, True, True, "FEASIBLE", True),
        ("feasible", -1, False, True, "FEASIBLE", True),
        ("", -1, False, True, "FEASIBLE", True),
        # 4. No sol available + hit limit -> TIMEOUT (-1)
        ("time limit exceeded, no integer solution", 108, True, False, "TIMEOUT", False),
        ("time limit exceeded", 107, True, False, "TIMEOUT", False),
        (None, -1, True, False, "TIMEOUT", False),
        ("time limit exceeded", -1, False, False, "TIMEOUT", False),
        ("node limit exceeded", 105, True, False, "ERROR", False),
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


def test_cpsat_api_calling_conventions():
    # 1. Dataset name keyword
    r1, s1 = solve_cpsat(dataset="14_bus", timeout=5.0, verbose=False)
    assert r1 == 3 and s1["status"] == "OPTIMAL"
    # 2. Dataset name positional
    r2, s2 = solve_cpsat("14_bus", timeout=5.0, verbose=False)
    assert r2 == 3 and s2["status"] == "OPTIMAL"
    # 3. (N, edges) keyword
    r3, s3 = solve_cpsat(N=4, edges=[(0, 1), (1, 2), (2, 3)], timeout=5.0, verbose=False)
    assert r3 == 1 and s3["status"] == "OPTIMAL"
    # 4. (N, edges) positional
    r4, s4 = solve_cpsat(4, [(0, 1), (1, 2), (2, 3)], timeout=5.0, verbose=False)
    assert r4 == 1 and s4["status"] == "OPTIMAL"

def test_cplex_api_calling_conventions():
    pytest.importorskip("cplex")
    # 1. Dataset name keyword
    r1, s1 = solve_cplex(dataset="14_bus", timeout=5.0, verbose=False)
    assert r1 == 3 and s1["status"] == "OPTIMAL"
    # 2. Dataset name positional
    r2, s2 = solve_cplex("14_bus", timeout=5.0, verbose=False)
    assert r2 == 3 and s2["status"] == "OPTIMAL"
    # 3. (N, edges) keyword
    r3, s3 = solve_cplex(N=4, edges=[(0, 1), (1, 2), (2, 3)], timeout=5.0, verbose=False)
    assert r3 == 1 and s3["status"] == "OPTIMAL"
    # 4. (N, edges) positional
    r4, s4 = solve_cplex(4, [(0, 1), (1, 2), (2, 3)], timeout=5.0, verbose=False)
    assert r4 == 1 and s4["status"] == "OPTIMAL"

def test_sat_api_calling_conventions():
    # 1. Dataset name keyword
    r1, s1 = solve(dataset="14_bus", timeout=5.0, verbose=False)
    assert r1 == 3 and s1["status"] == "OPTIMAL"
    # 2. Dataset name positional
    r2, s2 = solve("14_bus", timeout=5.0, verbose=False)
    assert r2 == 3 and s2["status"] == "OPTIMAL"
    # 3. (N, edges) keyword
    r3, s3 = solve(N=4, edges=[(0, 1), (1, 2), (2, 3)], verbose=False)
    assert r3 == 1 and s3["status"] == "OPTIMAL"
    # 4. (N, edges) positional
    r4, s4 = solve(4, [(0, 1), (1, 2), (2, 3)], verbose=False)
    assert r4 == 1 and s4["status"] == "OPTIMAL"

@pytest.mark.parametrize("name, N, edges", TEST_GRAPHS)
def test_cpsat_symmetry_breaking_preserves_optimum(name, N, edges):
    expected = cyclic_bandwidth_bruteforce(N, edges)
    actual_sb, stats_sb = solve_cpsat(N=N, edges=edges, sb_rotation=True, timeout=5.0, verbose=False)
    actual_nosb, stats_nosb = solve_cpsat(N=N, edges=edges, sb_rotation=False, timeout=5.0, verbose=False)
    assert actual_sb == expected
    assert actual_nosb == expected

@pytest.mark.parametrize("name, N, edges", TEST_GRAPHS)
def test_cplex_symmetry_breaking_preserves_optimum(name, N, edges):
    pytest.importorskip("cplex")
    expected = cyclic_bandwidth_bruteforce(N, edges)
    actual_sb, stats_sb = solve_cplex(N=N, edges=edges, sb_rotation=True, timeout=5.0, verbose=False)
    actual_nosb, stats_nosb = solve_cplex(N=N, edges=edges, sb_rotation=False, timeout=5.0, verbose=False)
    assert actual_sb == expected
    assert actual_nosb == expected

def test_cpsat_model_direct_inspection():
    """Directly inspect CP-SAT model structure built by build_cpsat_model."""
    N = 5
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    model, f, k, diff_vars, b_vars = build_cpsat_model(N, edges, sb_rotation=True)

    # Variables: 5 f-vars + 1 k-var + 4 diff-vars + 4 b-vars = 14
    assert len(model.proto.variables) == N + 1 + 2 * len(edges)
    # Constraints: 1 all_diff + 1 sb_rotation + 4 abs_eq + 4 upper_cond + 4 lower_cond = 14
    assert len(model.proto.constraints) == 2 + 3 * len(edges)

    # Rotation symmetry off
    model_nosb, _, _, _, _ = build_cpsat_model(N, edges, sb_rotation=False)
    assert len(model_nosb.proto.constraints) == 1 + 3 * len(edges)

def test_cplex_model_direct_inspection():
    """Directly inspect CPLEX model structure built by build_cplex_model."""
    pytest.importorskip("cplex")
    N = 4
    edges = [(0, 1), (1, 2), (2, 3)]
    mdl, x, f, k = build_cplex_model(N, edges, sb_rotation=True)

    # Variables: N*N binary + 1 integer k + 3*|E| (abs_d, s, b) = 16 + 1 + 9 = 26
    assert mdl.number_of_variables == N * N + 1 + 3 * len(edges)
    # Constraints: N vertex ALO/AMO + N label ALO/AMO + 1 sb_rotation + 6*|E| = 4 + 4 + 1 + 18 = 27
    assert mdl.number_of_constraints == 2 * N + 1 + 6 * len(edges)

    # Rotation symmetry off
    mdl_nosb, _, _, _ = build_cplex_model(N, edges, sb_rotation=False)
    assert mdl_nosb.number_of_constraints == 2 * N + 6 * len(edges)

def test_cpsat_model_equivalence_relabel_seed_none():
    """Verify that build_cpsat_model produces identical model structure and variables."""
    for name, N, edges in TEST_GRAPHS:
        m1, f1, k1, d1, b1 = build_cpsat_model(N, edges, sb_rotation=True)
        m2, f2, k2, d2, b2 = build_cpsat_model(N, edges, sb_rotation=True)
        assert len(m1.proto.variables) == len(m2.proto.variables) == N + 1 + 2 * len(edges)
        assert len(m1.proto.constraints) == len(m2.proto.constraints) == 2 + 3 * len(edges)
        assert str(m1.proto) == str(m2.proto)

        r1, s1 = solve_cpsat(N=N, edges=edges, timeout=5.0, relabel_seed=None, verbose=False)
        r2, s2 = solve_cpsat(N=N, edges=edges, timeout=5.0, verbose=False)
        assert r1 == r2
        assert s1["best_ub"] == s2["best_ub"]
        assert s1["best_lb"] == s2["best_lb"]


