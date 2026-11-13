"""
Mocked test suite for CPLEX CP Optimizer and Gurobi baselines.
Ensures comprehensive status coverage without requiring physical solver installations or commercial licenses.
"""

import sys
import types
import pytest
from unittest.mock import MagicMock, patch

from baselines.cplex_cp_cbmp import solve_cplex_cp
from baselines.gurobi_cbmp import solve_gurobi
from baselines.cpsat_cbmp import solve_cpsat
from baselines.cplex_cbmp import solve_cplex
from graph_utils import compute_cyclic_bandwidth


# ============================================================================
# 1. CPLEX CP OPTIMIZER MOCKED TESTS
# ============================================================================

class FakeCpoSolution:
    def __init__(self, solve_status="Optimal", stop_cause=None, has_solution=True, assignment=None, obj_val=1, obj_bound=None, num_constraints=None, solver_log=""):
        self._solve_status = solve_status
        self._stop_cause = stop_cause
        self._has_solution = has_solution
        self._assignment = assignment or [1, 2, 3]  # 1-indexed in docplex.cp
        self._obj_val = obj_val
        self._obj_bound = obj_bound
        self._num_constraints = num_constraints
        self._solver_log = solver_log

    def is_solution(self):
        return self._has_solution

    def get_solve_status(self):
        return self._solve_status

    def get_stop_cause(self):
        return self._stop_cause

    def get_objective_bound(self):
        return self._obj_bound

    def get_solver_info(self, key):
        if key == "NumberOfConstraints":
            return self._num_constraints
        return None

    def get_solver_log(self):
        return self._solver_log

    def get_value(self, var):
        return self[var]

    def get_objective_value(self):
        return self._obj_val

    def __getitem__(self, var):
        name = getattr(var, "name", str(var))
        if "k" in name:
            return self._obj_val
        for u in range(len(self._assignment)):
            if f"f_{u}" in name:
                return self._assignment[u]
        return 1


@pytest.fixture
def mock_cpo_environment(monkeypatch):
    """Ensure is_cplex_cp_available returns True and mock model.solve."""
    monkeypatch.setattr("baselines.cplex_cp_cbmp.is_cplex_cp_available", lambda: (True, "Mocked CP Optimizer"))


def test_cplex_cp_valid_optimal_solution(monkeypatch, mock_cpo_environment):
    """CPLEX CP returns proven OPTIMAL with consistent assignment and objective."""
    fake_sol = FakeCpoSolution(
        solve_status="Optimal",
        has_solution=True,
        assignment=[1, 2, 3],  # 0-indexed: [0, 1, 2] on path (0,1), (1,2) has bandwidth 1
        obj_val=1,
        obj_bound=1.0,
    )

    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "OPTIMAL"
    assert stats["best_ub"] == 1
    assert stats["best_lb"] == 1
    assert stats["gap"] == 0.0
    assert stats["assignment"] == [0, 1, 2]


def test_cplex_cp_valid_feasible_incumbent(monkeypatch, mock_cpo_environment):
    """CPLEX CP returns Feasible incumbent when stopped without time limit (e.g. IterationLimit)."""
    fake_sol = FakeCpoSolution(
        solve_status="Feasible",
        stop_cause="IterationLimit",
        has_solution=True,
        assignment=[1, 2, 3],
        obj_val=1,
        obj_bound=0.0,
    )

    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "FEASIBLE"
    assert stats["best_ub"] == 1
    assert stats["assignment"] == [0, 1, 2]
    assert stats["best_lb"] == 0
    assert stats["gap"] == 1.0


def test_cplex_cp_timeout_with_incumbent(monkeypatch, mock_cpo_environment):
    """CPLEX CP stops at TimeLimit with valid incumbent preserved."""
    fake_sol = FakeCpoSolution(
        solve_status="Feasible",
        stop_cause="TimeLimit",
        has_solution=True,
        assignment=[1, 2, 3],
        obj_val=1,
    )

    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "TIMEOUT"
    assert stats["best_ub"] == 1
    assert stats["assignment"] == [0, 1, 2]


def test_cplex_cp_timeout_without_incumbent(monkeypatch, mock_cpo_environment):
    """CPLEX CP stops at TimeLimit without finding any incumbent."""
    fake_sol = FakeCpoSolution(
        solve_status="Unknown",
        stop_cause="TimeLimit",
        has_solution=False,
    )

    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "TIMEOUT"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_cplex_cp_infeasible_model(monkeypatch, mock_cpo_environment):
    """CPLEX CP proves model infeasibility."""
    fake_sol = FakeCpoSolution(
        solve_status="Infeasible",
        has_solution=False,
    )

    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "INFEASIBLE"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_cplex_cp_invalid_permutation(monkeypatch, mock_cpo_environment):
    """CPLEX CP returns solution that violates bijection permutation property -> ERROR."""
    fake_sol = FakeCpoSolution(
        solve_status="Optimal",
        has_solution=True,
        assignment=[1, 1, 3],  # duplicate label 1!
        obj_val=1,
    )

    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "ERROR"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None
    assert "validation_error" in stats


def test_cplex_cp_objective_mismatch(monkeypatch, mock_cpo_environment):
    """CPLEX CP reports objective inconsistent with recomputed bandwidth -> ERROR."""
    fake_sol = FakeCpoSolution(
        solve_status="Optimal",
        has_solution=True,
        assignment=[1, 2, 3],  # actual bandwidth is 1
        obj_val=5,              # reported objective claims 5!
    )

    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "ERROR"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None
    assert "validation_error" in stats


def test_cplex_cp_unavailable_reporting(monkeypatch):
    """CPLEX CP reports UNAVAILABLE when solver engine is not found."""
    monkeypatch.setattr(
        "baselines.cplex_cp_cbmp.is_cplex_cp_available",
        lambda: (False, "CPLEX CP Optimizer engine not found in system PATH"),
    )
    res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "UNAVAILABLE"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_cplex_cp_docplex_search_limit_with_incumbent(monkeypatch, mock_cpo_environment):
    """CPLEX CP stops with generic SearchStoppedByLimit with valid incumbent -> FEASIBLE (preserves incumbent)."""
    fake_sol = FakeCpoSolution(
        solve_status="Feasible",
        stop_cause="SearchStoppedByLimit",
        has_solution=True,
        assignment=[1, 2, 3],
        obj_val=1,
    )

    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "FEASIBLE"
    assert stats["best_ub"] == 1
    assert stats["assignment"] == [0, 1, 2]


def test_cplex_cp_docplex_search_limit_without_incumbent(monkeypatch, mock_cpo_environment):
    """CPLEX CP stops with generic SearchStoppedByLimit without solution and without time evidence -> ERROR."""
    fake_sol = FakeCpoSolution(
        solve_status="Unknown",
        stop_cause="SearchStoppedByLimit",
        has_solution=False,
    )

    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "ERROR"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_cplex_cp_node_limit_near_timeout_with_incumbent_is_feasible(monkeypatch, mock_cpo_environment):
    """
    CPLEX CP stops at NodeLimit/SearchStoppedByLimit near configured timeout with incumbent.
    Proves that wall-clock proximity alone cannot turn status into TIMEOUT.
    """
    fake_sol = FakeCpoSolution(
        solve_status="Feasible",
        stop_cause="NodeLimit",
        has_solution=True,
        assignment=[1, 2, 3],
        obj_val=1,
    )

    # Mock time.perf_counter so elapsed time is 4.95s on timeout=5.0s
    times = [100.0, 100.0, 100.0, 104.95]
    monkeypatch.setattr("time.perf_counter", lambda: times.pop(0) if times else 104.95)

    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "FEASIBLE", "NodeLimit must NOT become TIMEOUT solely due to elapsed time"
    assert stats["best_ub"] == 1
    assert stats["assignment"] == [0, 1, 2]


def test_cplex_cp_node_limit_near_timeout_without_incumbent_is_error(monkeypatch, mock_cpo_environment):
    """
    CPLEX CP stops at NodeLimit/SearchStoppedByLimit near configured timeout without incumbent.
    Proves that wall-clock proximity alone cannot turn status into TIMEOUT.
    """
    fake_sol = FakeCpoSolution(
        solve_status="Unknown",
        stop_cause="NodeLimit",
        has_solution=False,
    )

    times = [100.0, 100.0, 100.0, 104.95]
    monkeypatch.setattr("time.perf_counter", lambda: times.pop(0) if times else 104.95)

    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "ERROR", "NodeLimit without solution must NOT become TIMEOUT solely due to elapsed time"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_cplex_cp_statistics_semantics(monkeypatch, mock_cpo_environment):
    """CPLEX CP statistics must report num_clauses as None, estimated_constraints, and num_constraints."""
    # Case A: solver exposes genuine NumberOfConstraints
    fake_sol_with_c = FakeCpoSolution(
        solve_status="Optimal",
        has_solution=True,
        assignment=[1, 2, 3],
        obj_val=1,
        num_constraints=12,
    )
    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol_with_c):
        res, stats_a = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], sb_rotation=True, timeout=5.0, verbose=False)

    assert stats_a["num_clauses"] is None, "SAT clauses must be None for CP Optimizer"
    assert stats_a["estimated_constraints"] == 2 + 1 + 1  # 2 edges + 1 all_different + 1 sb_rotation
    assert stats_a["num_constraints"] == 12

    # Case B: solver does NOT expose NumberOfConstraints -> num_constraints must be None
    fake_sol_no_c = FakeCpoSolution(
        solve_status="Optimal",
        has_solution=True,
        assignment=[1, 2, 3],
        obj_val=1,
        num_constraints=None,
    )
    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol_no_c):
        res, stats_b = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], sb_rotation=True, timeout=5.0, verbose=False)

    assert stats_b["num_clauses"] is None
    assert stats_b["estimated_constraints"] == 2 + 1 + 1
    assert stats_b["num_constraints"] is None, "num_constraints must be None when no genuine solver statistic exists"


def test_cplex_cp_aborted_stop_preserves_incumbent(monkeypatch, mock_cpo_environment):
    """CPLEX CP stopped by abort with incumbent -> INTERRUPTED."""
    fake_sol = FakeCpoSolution(
        solve_status="JobAborted",
        stop_cause="SearchStoppedByAbort",
        has_solution=True,
        assignment=[1, 2, 3],
        obj_val=1,
    )

    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "INTERRUPTED"
    assert stats["best_ub"] == 1
    assert stats["assignment"] == [0, 1, 2]


def test_cplex_cp_aborted_stop_without_solution(monkeypatch, mock_cpo_environment):
    """CPLEX CP stopped by abort without incumbent -> INTERRUPTED."""
    fake_sol = FakeCpoSolution(
        solve_status="JobAborted",
        stop_cause="SearchStoppedByAbort",
        has_solution=False,
    )

    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "INTERRUPTED"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_cplex_cp_job_failed_status(monkeypatch, mock_cpo_environment):
    """CPLEX CP job failed -> ERROR."""
    fake_sol = FakeCpoSolution(
        solve_status="JobFailed",
        stop_cause="SearchStoppedByUnknownCause",
        has_solution=False,
    )

    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "ERROR"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_cplex_cp_unknown_stop_without_limit_evidence_is_error(monkeypatch, mock_cpo_environment):
    """Unknown termination reasons without limit evidence must NOT be classified as TIMEOUT."""
    fake_sol = FakeCpoSolution(
        solve_status="Unknown",
        stop_cause="SearchStoppedByUnknownCause",
        has_solution=False,
    )

    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "ERROR"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_cplex_cp_statistics_invalid_and_exceptional(monkeypatch, mock_cpo_environment):
    """Verify num_constraints independently evaluates solver-info sources and rejects invalid values."""
    # 1. get_solver_info raises, but get_solver_infos succeeds -> returns 20
    class FallbackCpoSolution(FakeCpoSolution):
        def get_solver_info(self, key):
            raise RuntimeError("Primary API failure")
        def get_solver_infos(self):
            class FakeInfos:
                def get_number_of_constraints(self):
                    return 20
            return FakeInfos()

    fake_sol_fb = FallbackCpoSolution(solve_status="Optimal", has_solution=True, assignment=[1, 2, 3], obj_val=1)
    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol_fb):
        res, stats_fb = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats_fb["num_constraints"] == 20

    # 2. get_solver_info returns None, but get_solver_infos succeeds -> returns 25
    class NoneFallbackCpoSolution(FakeCpoSolution):
        def get_solver_info(self, key):
            return None
        def get_solver_infos(self):
            return {"NumberOfConstraints": 25}

    fake_sol_nfb = NoneFallbackCpoSolution(solve_status="Optimal", has_solution=True, assignment=[1, 2, 3], obj_val=1)
    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol_nfb):
        res, stats_nfb = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats_nfb["num_constraints"] == 25

    # 3. Both sources raise -> None
    class BothRaiseCpoSolution(FakeCpoSolution):
        def get_solver_info(self, key):
            raise RuntimeError("Corrupt solver info stream")
        def get_solver_infos(self):
            raise ValueError("Invalid solver infos")

    fake_sol_br = BothRaiseCpoSolution(solve_status="Optimal", has_solution=True, assignment=[1, 2, 3], obj_val=1)
    with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol_br):
        res, stats_br = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats_br["num_constraints"] is None
    assert stats_br["estimated_constraints"] == 2 + 1 + 1

    # 4. Invalid values (boolean True, negative, NaN, infinity, bad string) must all be rejected -> None
    for invalid_val in [True, False, -5, float("nan"), float("inf"), "invalid_str"]:
        class InvalidValCpoSolution(FakeCpoSolution):
            def get_solver_info(self, key):
                return invalid_val
            def get_solver_infos(self):
                return {"NumberOfConstraints": invalid_val}

        fake_sol_inv = InvalidValCpoSolution(solve_status="Optimal", has_solution=True, assignment=[1, 2, 3], obj_val=1)
        with patch("docplex.cp.model.CpoModel.solve", return_value=fake_sol_inv):
            res, stats_inv = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
        assert stats_inv["num_constraints"] is None, f"Invalid value {invalid_val!r} must be rejected"

    # 5. No solver-info API available at all -> None
    class BareCpoSolution:
        def is_solution(self): return True
        def get_solve_status(self): return "Optimal"
        def is_solution_optimal(self): return True
        def get_stop_cause(self): return None
        def get_fail_status(self): return None
        def get_search_status(self): return None
        def __getitem__(self, var):
            name = getattr(var, "name", str(var))
            return 1 if "k" in name else 1

    with patch("docplex.cp.model.CpoModel.solve", return_value=BareCpoSolution()):
        res, stats_bare = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats_bare["num_constraints"] is None


def test_cplex_cp_solver_exception_and_license_error(monkeypatch, mock_cpo_environment):
    """Verify solve_cplex_cp handles model.solve exceptions and distinguishes LICENSE_ERROR."""
    # Sub-case 1: General solver exception -> ERROR
    def mock_solve_err(*args, **kwargs):
        raise RuntimeError("Engine segmentation violation")

    with patch("docplex.cp.model.CpoModel.solve", mock_solve_err):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1)], timeout=5.0, verbose=False)
    assert stats["status"] == "ERROR"
    assert "Engine segmentation violation" in stats["error"]
    assert stats["best_ub"] == -1

    # Sub-case 2: License failure -> LICENSE_ERROR
    def mock_solve_lic(*args, **kwargs):
        raise RuntimeError("CPLEX Error 1014: License token unavailable")

    with patch("docplex.cp.model.CpoModel.solve", mock_solve_lic):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1)], timeout=5.0, verbose=False)
    assert stats["status"] == "LICENSE_ERROR"
    assert stats["best_ub"] == -1


def test_cplex_cp_conditional_live_integration():
    """Live integration test against installed CP Optimizer; skips explicitly if engine binary not in PATH."""
    import shutil
    if not shutil.which("cpoptimizer"):
        pytest.skip("CP Optimizer executable 'cpoptimizer' is not installed or not in PATH.")

    from baselines.cplex_cp_cbmp import solve_cplex_cp, is_cplex_cp_available
    avail, reason = is_cplex_cp_available()
    if not avail:
        pytest.skip(f"CP Optimizer unavailable: {reason}")

    res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "OPTIMAL"
    assert res == 1
    assert stats["best_ub"] == 1
    assert stats["assignment"] is not None


# ============================================================================
# 2. GUROBI MOCKED TESTS
# ============================================================================

class FakeGrbVar:
    def __init__(self, val):
        self.X = val


class FakeGrbModel:
    def __init__(self, status=2, sol_count=1, assignment_vals=None, k_val=1, obj_bound=1.0):
        self.Status = status
        self.SolCount = sol_count
        self.NumVars = 10
        self.NumConstrs = 10
        self.ObjBound = obj_bound
        self._assignment_vals = assignment_vals or [1.0, 2.0, 3.0]
        self._k_val = k_val

    def setParam(self, param, val):
        pass

    def optimize(self):
        pass


@pytest.fixture
def mock_gurobi_environment(monkeypatch):
    """Ensure is_gurobi_available returns True and mock gurobipy in sys.modules."""
    monkeypatch.setattr("baselines.gurobi_cbmp.is_gurobi_available", lambda: (True, "Mocked Gurobi"))
    fake_grb = types.SimpleNamespace(
        OPTIMAL=2,
        INFEASIBLE=3,
        INF_OR_UNBD=4,
        TIME_LIMIT=9,
        NODE_LIMIT=8,
        ITERATION_LIMIT=7,
        SOLUTION_LIMIT=10,
        INTERRUPTED=11,
        NUMERIC=12,
        SUBOPTIMAL=13,
        BINARY="B",
        INTEGER="I",
        MINIMIZE=1,
    )
    fake_gp = types.ModuleType("gurobipy")
    fake_gp.GRB = fake_grb
    monkeypatch.setitem(sys.modules, "gurobipy", fake_gp)


def test_gurobi_valid_optimal_solution(monkeypatch, mock_gurobi_environment):
    """Gurobi returns GRB.OPTIMAL with valid assignment and matching objective."""
    fake_model = FakeGrbModel(status=2, sol_count=1, assignment_vals=[1.0, 2.0, 3.0], k_val=1, obj_bound=1.0)
    fake_f = {u: FakeGrbVar(fake_model._assignment_vals[u]) for u in range(3)}
    fake_k = FakeGrbVar(1.0)

    monkeypatch.setattr("baselines.gurobi_cbmp.build_gurobi_model", lambda N, edges, sb_rotation=True: (fake_model, {}, fake_f, fake_k))

    res, stats = solve_gurobi(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "OPTIMAL"
    assert stats["best_ub"] == 1
    assert stats["best_lb"] == 1
    assert stats["gap"] == 0.0
    assert stats["assignment"] == [0, 1, 2]


def test_gurobi_time_limit_with_incumbent(monkeypatch, mock_gurobi_environment):
    """Gurobi hits TIME_LIMIT (code 9) with feasible incumbent preserved."""
    fake_model = FakeGrbModel(status=9, sol_count=1, assignment_vals=[1.0, 2.0, 3.0], k_val=1, obj_bound=0.0)
    fake_f = {u: FakeGrbVar(fake_model._assignment_vals[u]) for u in range(3)}
    fake_k = FakeGrbVar(1.0)

    monkeypatch.setattr("baselines.gurobi_cbmp.build_gurobi_model", lambda N, edges, sb_rotation=True: (fake_model, {}, fake_f, fake_k))

    res, stats = solve_gurobi(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "TIMEOUT"
    assert stats["best_ub"] == 1
    assert stats["assignment"] == [0, 1, 2]


def test_gurobi_time_limit_without_incumbent(monkeypatch, mock_gurobi_environment):
    """Gurobi hits TIME_LIMIT (code 9) without finding an incumbent."""
    fake_model = FakeGrbModel(status=9, sol_count=0)
    fake_f = {u: FakeGrbVar(0.0) for u in range(3)}
    fake_k = FakeGrbVar(0.0)

    monkeypatch.setattr("baselines.gurobi_cbmp.build_gurobi_model", lambda N, edges, sb_rotation=True: (fake_model, {}, fake_f, fake_k))

    res, stats = solve_gurobi(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "TIMEOUT"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_gurobi_node_limit_with_incumbent(monkeypatch, mock_gurobi_environment):
    """Gurobi terminates due to NODE_LIMIT (code 8) with incumbent -> FEASIBLE."""
    fake_model = FakeGrbModel(status=8, sol_count=1, assignment_vals=[1.0, 2.0, 3.0], k_val=1, obj_bound=0.0)
    fake_f = {u: FakeGrbVar(fake_model._assignment_vals[u]) for u in range(3)}
    fake_k = FakeGrbVar(1.0)

    monkeypatch.setattr("baselines.gurobi_cbmp.build_gurobi_model", lambda N, edges, sb_rotation=True: (fake_model, {}, fake_f, fake_k))

    res, stats = solve_gurobi(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "FEASIBLE"
    assert stats["best_ub"] == 1
    assert stats["assignment"] == [0, 1, 2]


def test_gurobi_interrupted_with_incumbent(monkeypatch, mock_gurobi_environment):
    """Gurobi is INTERRUPTED (code 11) with incumbent preserved."""
    fake_model = FakeGrbModel(status=11, sol_count=1, assignment_vals=[1.0, 2.0, 3.0], k_val=1, obj_bound=0.0)
    fake_f = {u: FakeGrbVar(fake_model._assignment_vals[u]) for u in range(3)}
    fake_k = FakeGrbVar(1.0)

    monkeypatch.setattr("baselines.gurobi_cbmp.build_gurobi_model", lambda N, edges, sb_rotation=True: (fake_model, {}, fake_f, fake_k))

    res, stats = solve_gurobi(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "INTERRUPTED"
    assert stats["best_ub"] == 1
    assert stats["assignment"] == [0, 1, 2]


def test_gurobi_infeasible_model(monkeypatch, mock_gurobi_environment):
    """Gurobi proves model infeasibility (status 3)."""
    fake_model = FakeGrbModel(status=3, sol_count=0)
    fake_f = {u: FakeGrbVar(0.0) for u in range(3)}
    fake_k = FakeGrbVar(0.0)

    monkeypatch.setattr("baselines.gurobi_cbmp.build_gurobi_model", lambda N, edges, sb_rotation=True: (fake_model, {}, fake_f, fake_k))

    res, stats = solve_gurobi(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "INFEASIBLE"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_gurobi_invalid_assignment(monkeypatch, mock_gurobi_environment):
    """Gurobi returns non-permutation assignment -> ERROR."""
    fake_model = FakeGrbModel(status=2, sol_count=1, assignment_vals=[1.0, 1.0, 3.0], k_val=1)
    fake_f = {u: FakeGrbVar(fake_model._assignment_vals[u]) for u in range(3)}
    fake_k = FakeGrbVar(1.0)

    monkeypatch.setattr("baselines.gurobi_cbmp.build_gurobi_model", lambda N, edges, sb_rotation=True: (fake_model, {}, fake_f, fake_k))

    res, stats = solve_gurobi(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "ERROR"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None
    assert "validation_error" in stats


def test_gurobi_objective_mismatch(monkeypatch, mock_gurobi_environment):
    """Gurobi objective variable disagrees with recomputed cyclic bandwidth -> ERROR."""
    fake_model = FakeGrbModel(status=2, sol_count=1, assignment_vals=[1.0, 2.0, 3.0], k_val=10)
    fake_f = {u: FakeGrbVar(fake_model._assignment_vals[u]) for u in range(3)}
    fake_k = FakeGrbVar(10.0)

    monkeypatch.setattr("baselines.gurobi_cbmp.build_gurobi_model", lambda N, edges, sb_rotation=True: (fake_model, {}, fake_f, fake_k))

    res, stats = solve_gurobi(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "ERROR"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None
    assert "validation_error" in stats


def test_gurobi_missing_or_invalid_license(monkeypatch):
    """Gurobi reports LICENSE_ERROR when license is expired or missing."""
    monkeypatch.setattr(
        "baselines.gurobi_cbmp.is_gurobi_available",
        lambda: (False, "Gurobi solver unavailable (license error): No Gurobi license found"),
    )
    res, stats = solve_gurobi(N=3, edges=[(0, 1)], verbose=False)
    assert stats["status"] == "LICENSE_ERROR"
    assert stats["best_ub"] == -1
    assert "error" in stats


# ============================================================================
# 3. CROSS-BASELINE VALIDATION CONTRACT TESTS
# ============================================================================

@pytest.mark.parametrize("solver_fn", [solve_cpsat, solve_cplex_cp, solve_gurobi])
def test_baseline_empty_graph_contract(solver_fn):
    """All baselines on empty graph must return OPTIMAL with ub=0, lb=0, gap=0.0, valid assignment."""
    res, stats = solver_fn(N=4, edges=[], verbose=False)
    assert stats["status"] == "OPTIMAL"
    assert stats["best_ub"] == 0
    assert stats["best_lb"] == 0
    assert stats["gap"] == 0.0
    assert stats["assignment"] == [0, 1, 2, 3]
    assert stats["solve_time"] >= 0.0
    assert stats["encode_time"] >= 0.0


# ============================================================================
# 4. ADDITIONAL CPLEX CP API PROTECTION & STATUS REFINEMENT TESTS
# ============================================================================

def test_cplex_cp_api_missing_methods(mock_cpo_environment):
    """CPLEX CP solve result missing all get_* status methods does not crash and defaults safely."""
    class NakedSolution:
        pass  # Has zero methods

    naked_sol = NakedSolution()
    with patch("docplex.cp.model.CpoModel.solve", return_value=naked_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "ERROR"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_cplex_cp_api_raising_exceptions(mock_cpo_environment):
    """CPLEX CP solve result methods raising exceptions do not crash solve_cplex_cp."""
    class BrokenSolution:
        def is_solution(self):
            raise RuntimeError("API failure in is_solution")
        def get_solve_status(self):
            raise RuntimeError("API failure in get_solve_status")
        def get_stop_cause(self):
            raise RuntimeError("API failure in get_stop_cause")
        def get_fail_status(self):
            raise RuntimeError("API failure in get_fail_status")
        def get_search_status(self):
            raise RuntimeError("API failure in get_search_status")
        def is_solution_optimal(self):
            raise RuntimeError("API failure in is_solution_optimal")

    broken_sol = BrokenSolution()
    with patch("docplex.cp.model.CpoModel.solve", return_value=broken_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "ERROR"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_cplex_cp_api_returning_none(mock_cpo_environment):
    """CPLEX CP solve result methods returning None do not crash solve_cplex_cp."""
    class NoneSolution:
        def is_solution(self):
            return None
        def get_solve_status(self):
            return None
        def get_stop_cause(self):
            return None
        def get_fail_status(self):
            return None
        def get_search_status(self):
            return None
        def is_solution_optimal(self):
            return None

    none_sol = NoneSolution()
    with patch("docplex.cp.model.CpoModel.solve", return_value=none_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "ERROR"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_cplex_cp_api_malformed_values(mock_cpo_environment):
    """CPLEX CP solve result methods returning non-string/malformed values do not crash."""
    class MalformedSolution:
        def is_solution(self):
            return "not_a_bool"
        def get_solve_status(self):
            return ["weird", 123]
        def get_stop_cause(self):
            return {"cause": "unknown"}
        def get_fail_status(self):
            return 999
        def get_search_status(self):
            return False
        def __getitem__(self, var):
            return 1

    malformed_sol = MalformedSolution()
    with patch("docplex.cp.model.CpoModel.solve", return_value=malformed_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "ERROR"
    assert stats["best_ub"] == -1


def test_cplex_cp_is_solution_string_truthy_not_accepted(mock_cpo_environment):
    """
    CPLEX CP is_solution() returns 'false' or 'not_a_bool' (truthy strings).
    Even with valid assignment retrievable via __getitem__, string truthiness must NOT be
    accepted as boolean True.
    """
    class StringSolution:
        def is_solution(self):
            return "false"
        def get_solve_status(self):
            return "Feasible"
        def __getitem__(self, var):
            name = getattr(var, "name", str(var))
            if "k" in name:
                return 1
            for u in range(3):
                if f"f_{u}" in name:
                    return u + 1
            return 1

    str_sol = StringSolution()
    with patch("docplex.cp.model.CpoModel.solve", return_value=str_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "ERROR"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_cplex_cp_is_solution_not_a_bool_with_valid_assignment(mock_cpo_environment):
    """is_solution() returns 'not_a_bool' with valid assignment -> not accepted as solution."""
    class NonBoolSolution:
        def is_solution(self):
            return "not_a_bool"
        def __getitem__(self, var):
            name = getattr(var, "name", str(var))
            if "k" in name:
                return 1
            for u in range(3):
                if f"f_{u}" in name:
                    return u + 1
            return 1

    with patch("docplex.cp.model.CpoModel.solve", return_value=NonBoolSolution()):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "ERROR"
    assert stats["best_ub"] == -1


@pytest.mark.parametrize("malformed_optimal", ["false", "not_a_bool", 1, 0, None, [True]])
def test_cplex_cp_is_solution_optimal_malformed_not_accepted(mock_cpo_environment, malformed_optimal):
    """
    When is_solution() is genuine True and assignment is valid,
    malformed is_solution_optimal() (strings like 'false', numbers 1/0, None, list)
    must NOT be classified as OPTIMAL.
    """
    class ValidIncumbentMalformedOptimal:
        def is_solution(self):
            return True
        def is_solution_optimal(self):
            return malformed_optimal
        def get_solve_status(self):
            return "Feasible"
        def __getitem__(self, var):
            name = getattr(var, "name", str(var))
            if "k" in name:
                return 1
            for u in range(3):
                if f"f_{u}" in name:
                    return u + 1
            return 1

    sol = ValidIncumbentMalformedOptimal()
    with patch("docplex.cp.model.CpoModel.solve", return_value=sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    # Valid incumbent preserved, but NOT optimal
    assert stats["status"] == "FEASIBLE"
    assert stats["status"] != "OPTIMAL"
    assert stats["best_ub"] == 1
    assert stats["assignment"] == [0, 1, 2]


def test_cplex_cp_is_solution_optimal_true_but_invalid_assignment(mock_cpo_environment):
    """
    is_solution_optimal() returns True, but assignment has duplicate labels -> ERROR.
    """
    class BrokenAssignmentOptimal:
        def is_solution(self):
            return True
        def is_solution_optimal(self):
            return True
        def get_solve_status(self):
            return "Optimal"
        def __getitem__(self, var):
            name = getattr(var, "name", str(var))
            if "k" in name:
                return 1
            # Invalid non-permutation: duplicate position 1
            return 1

    broken_sol = BrokenAssignmentOptimal()
    with patch("docplex.cp.model.CpoModel.solve", return_value=broken_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "ERROR"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_cplex_cp_genuine_booleans_work(mock_cpo_environment):
    """True and False booleans operate as expected."""
    # Case 1: True + True -> OPTIMAL
    sol_opt = FakeCpoSolution(solve_status="Optimal", has_solution=True, assignment=[1, 2, 3], obj_val=1)
    with patch("docplex.cp.model.CpoModel.solve", return_value=sol_opt):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "OPTIMAL"

    # Case 2: True + False -> FEASIBLE
    sol_feas = FakeCpoSolution(solve_status="Feasible", has_solution=True, assignment=[1, 2, 3], obj_val=1)
    with patch("docplex.cp.model.CpoModel.solve", return_value=sol_feas):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "FEASIBLE"

    # Case 3: False -> ERROR (no solution)
    sol_no = FakeCpoSolution(solve_status="Unknown", has_solution=False)
    with patch("docplex.cp.model.CpoModel.solve", return_value=sol_no):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "ERROR"


def test_cplex_cp_partial_apis_functional(mock_cpo_environment):
    """When some APIs fail/raise, remaining working APIs allow proper classification."""
    class PartiallyWorkingSolution:
        def is_solution(self):
            return True
        def get_solve_status(self):
            raise RuntimeError("solve_status unavailable")
        def get_stop_cause(self):
            return "SearchStoppedByLimit"
        def get_fail_status(self):
            return None
        def get_search_status(self):
            return "SearchStopped"
        def get_solver_log(self):
            return "! Time limit reached"
        def is_solution_optimal(self):
            return False
        def __getitem__(self, var):
            name = getattr(var, "name", str(var))
            if "k" in name:
                return 1
            for u in range(3):
                if f"f_{u}" in name:
                    return u + 1
            return 1

    part_sol = PartiallyWorkingSolution()
    with patch("docplex.cp.model.CpoModel.solve", return_value=part_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    # Has solution + time limit reached log -> TIMEOUT with incumbent preserved
    assert stats["status"] == "TIMEOUT"
    assert stats["best_ub"] == 1
    assert stats["assignment"] == [0, 1, 2]


def test_cplex_cp_solution_unverifiable_yields_error(mock_cpo_environment):
    """When is_solution is True but variable extraction raises exception, status must be ERROR."""
    class BrokenVarSolution:
        def is_solution(self):
            return True
        def is_solution_optimal(self):
            return True
        def get_solve_status(self):
            return "Optimal"
        def __getitem__(self, var):
            raise RuntimeError("Variable value extraction failed")

    broken_var_sol = BrokenVarSolution()
    with patch("docplex.cp.model.CpoModel.solve", return_value=broken_var_sol):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "ERROR"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_cplex_cp_real_docplex_raw_statuses(mock_cpo_environment):
    """Test against actual Docplex CP Optimizer constant values."""
    # STOP_CAUSE_LIMIT + SearchStopped without timelimit log -> FEASIBLE with incumbent
    sol_feas = FakeCpoSolution(
        solve_status="Feasible",
        stop_cause="SearchStoppedByLimit",
        has_solution=True,
        assignment=[1, 2, 3],
        obj_val=1,
    )
    with patch("docplex.cp.model.CpoModel.solve", return_value=sol_feas):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "FEASIBLE"
    assert stats["best_ub"] == 1

    # STOP_CAUSE_ABORT + JobAborted -> INTERRUPTED
    sol_abort = FakeCpoSolution(
        solve_status="JobAborted",
        stop_cause="SearchStoppedByAbort",
        has_solution=False,
    )
    with patch("docplex.cp.model.CpoModel.solve", return_value=sol_abort):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "INTERRUPTED"

    # Infeasible -> INFEASIBLE
    sol_infeas = FakeCpoSolution(
        solve_status="Infeasible",
        stop_cause="SearchCompleted",
        has_solution=False,
    )
    with patch("docplex.cp.model.CpoModel.solve", return_value=sol_infeas):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "INFEASIBLE"


# ============================================================================
# 5. ADDITIONAL GUROBI TERMINATION CLASSIFICATION TESTS
# ============================================================================

def test_build_gurobi_model_without_gurobipy_raises_importerror(monkeypatch):
    """Calling build_gurobi_model when gurobipy is not installed raises ImportError, not NameError."""
    from baselines.gurobi_cbmp import build_gurobi_model

    monkeypatch.setitem(sys.modules, "gurobipy", None)
    with pytest.raises(ImportError) as exc_info:
        build_gurobi_model(3, [(0, 1)])
    assert "gurobipy" in str(exc_info.value).lower()


def test_gurobi_node_limit_without_incumbent_is_error(monkeypatch, mock_gurobi_environment):
    """Gurobi hits NODE_LIMIT (status 8) without finding an incumbent -> ERROR (NOT TIMEOUT)."""
    fake_model = FakeGrbModel(status=8, sol_count=0)
    monkeypatch.setattr(
        "baselines.gurobi_cbmp.build_gurobi_model",
        lambda N, edges, sb_rotation=True: (fake_model, {}, {}, FakeGrbVar(0.0)),
    )
    res, stats = solve_gurobi(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "ERROR"
    assert stats["status"] != "TIMEOUT"
    assert stats["termination_reason"] == "search_limit"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_gurobi_iteration_limit_without_incumbent_is_error(monkeypatch, mock_gurobi_environment):
    """Gurobi hits ITERATION_LIMIT (status 7) without finding an incumbent -> ERROR (NOT TIMEOUT)."""
    fake_model = FakeGrbModel(status=7, sol_count=0)
    monkeypatch.setattr(
        "baselines.gurobi_cbmp.build_gurobi_model",
        lambda N, edges, sb_rotation=True: (fake_model, {}, {}, FakeGrbVar(0.0)),
    )
    res, stats = solve_gurobi(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "ERROR"
    assert stats["status"] != "TIMEOUT"
    assert stats["termination_reason"] == "search_limit"
    assert stats["best_ub"] == -1


def test_gurobi_solution_limit_without_incumbent_is_error(monkeypatch, mock_gurobi_environment):
    """Gurobi hits SOLUTION_LIMIT (status 10) without finding an incumbent -> ERROR (NOT TIMEOUT)."""
    fake_model = FakeGrbModel(status=10, sol_count=0)
    monkeypatch.setattr(
        "baselines.gurobi_cbmp.build_gurobi_model",
        lambda N, edges, sb_rotation=True: (fake_model, {}, {}, FakeGrbVar(0.0)),
    )
    res, stats = solve_gurobi(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "ERROR"
    assert stats["status"] != "TIMEOUT"
    assert stats["termination_reason"] == "search_limit"
    assert stats["best_ub"] == -1


def test_gurobi_numeric_error_without_incumbent(monkeypatch, mock_gurobi_environment):
    """Gurobi terminates due to NUMERIC error (status 12) without incumbent -> NUMERIC_ERROR."""
    fake_model = FakeGrbModel(status=12, sol_count=0)
    monkeypatch.setattr(
        "baselines.gurobi_cbmp.build_gurobi_model",
        lambda N, edges, sb_rotation=True: (fake_model, {}, {}, FakeGrbVar(0.0)),
    )
    res, stats = solve_gurobi(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "NUMERIC_ERROR"
    assert stats["termination_reason"] == "numeric_error"
    assert stats["best_ub"] == -1


def test_gurobi_interrupted_without_incumbent(monkeypatch, mock_gurobi_environment):
    """Gurobi terminates due to INTERRUPTED (status 11) without incumbent -> INTERRUPTED."""
    fake_model = FakeGrbModel(status=11, sol_count=0)
    monkeypatch.setattr(
        "baselines.gurobi_cbmp.build_gurobi_model",
        lambda N, edges, sb_rotation=True: (fake_model, {}, {}, FakeGrbVar(0.0)),
    )
    res, stats = solve_gurobi(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "INTERRUPTED"
    assert stats["termination_reason"] == "interrupted"
    assert stats["best_ub"] == -1


def test_gurobi_iteration_limit_with_incumbent_is_feasible(monkeypatch, mock_gurobi_environment):
    """Gurobi terminates due to ITERATION_LIMIT (status 7) with incumbent -> FEASIBLE."""
    fake_model = FakeGrbModel(status=7, sol_count=1, assignment_vals=[1.0, 2.0, 3.0], k_val=1, obj_bound=0.0)
    fake_f = {u: FakeGrbVar(fake_model._assignment_vals[u]) for u in range(3)}
    monkeypatch.setattr(
        "baselines.gurobi_cbmp.build_gurobi_model",
        lambda N, edges, sb_rotation=True: (fake_model, {}, fake_f, FakeGrbVar(1.0)),
    )
    res, stats = solve_gurobi(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "FEASIBLE"
    assert stats["termination_reason"] == "search_limit"
    assert stats["best_ub"] == 1
    assert stats["assignment"] == [0, 1, 2]


def test_gurobi_solution_limit_with_incumbent_is_feasible(monkeypatch, mock_gurobi_environment):
    """Gurobi terminates due to SOLUTION_LIMIT (status 10) with incumbent -> FEASIBLE."""
    fake_model = FakeGrbModel(status=10, sol_count=1, assignment_vals=[1.0, 2.0, 3.0], k_val=1, obj_bound=0.0)
    fake_f = {u: FakeGrbVar(fake_model._assignment_vals[u]) for u in range(3)}
    monkeypatch.setattr(
        "baselines.gurobi_cbmp.build_gurobi_model",
        lambda N, edges, sb_rotation=True: (fake_model, {}, fake_f, FakeGrbVar(1.0)),
    )
    res, stats = solve_gurobi(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "FEASIBLE"
    assert stats["termination_reason"] == "search_limit"
    assert stats["best_ub"] == 1
    assert stats["assignment"] == [0, 1, 2]


def test_gurobi_suboptimal_with_incumbent_is_feasible(monkeypatch, mock_gurobi_environment):
    """Gurobi terminates due to SUBOPTIMAL (status 13) with incumbent -> FEASIBLE."""
    fake_model = FakeGrbModel(status=13, sol_count=1, assignment_vals=[1.0, 2.0, 3.0], k_val=1, obj_bound=0.0)
    fake_f = {u: FakeGrbVar(fake_model._assignment_vals[u]) for u in range(3)}
    monkeypatch.setattr(
        "baselines.gurobi_cbmp.build_gurobi_model",
        lambda N, edges, sb_rotation=True: (fake_model, {}, fake_f, FakeGrbVar(1.0)),
    )
    res, stats = solve_gurobi(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "FEASIBLE"
    assert stats["termination_reason"] == "search_limit"
    assert stats["best_ub"] == 1
    assert stats["assignment"] == [0, 1, 2]


def test_gurobi_optimal_with_invalid_assignment_yields_error(monkeypatch, mock_gurobi_environment):
    """Gurobi reports GRB.OPTIMAL but assignment fails validation -> ERROR."""
    fake_model = FakeGrbModel(status=2, sol_count=1, assignment_vals=[1.0, 1.0, 3.0], k_val=1)
    fake_f = {u: FakeGrbVar(fake_model._assignment_vals[u]) for u in range(3)}
    monkeypatch.setattr(
        "baselines.gurobi_cbmp.build_gurobi_model",
        lambda N, edges, sb_rotation=True: (fake_model, {}, fake_f, FakeGrbVar(1.0)),
    )
    res, stats = solve_gurobi(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)
    assert stats["status"] == "ERROR"
    assert stats["termination_reason"] == "validation_error"
    assert stats["best_ub"] == -1
    assert stats["assignment"] is None


def test_cplex_cp_jobfailed_with_valid_incumbent_records_stop_reason_metadata(mock_cpo_environment):
    """
    A5: When CP Optimizer terminates with JobFailed or Exception fail_status
    yet has a valid validated incumbent, status remains FEASIBLE and the
    stop/failure cause must be recorded in metadata (not swallowed).
    """
    class JobFailedWithIncumbentSolution(FakeCpoSolution):
        def __init__(self, fail_status="Exception", **kwargs):
            super().__init__(**kwargs)
            self._fail_status = fail_status

        def get_fail_status(self):
            return self._fail_status

    sol_incumbent = JobFailedWithIncumbentSolution(
        solve_status="JobFailed",
        stop_cause="Fail",
        fail_status="Exception",
        has_solution=True,
        assignment=[1, 2, 3],
        obj_val=1,
        obj_bound=0,
    )
    with patch("docplex.cp.model.CpoModel.solve", return_value=sol_incumbent):
        res, stats = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats["status"] == "FEASIBLE"
    assert stats["best_ub"] == 1
    assert stats["assignment"] == [0, 1, 2]
    assert stats["termination_reason"] == "fail_stop_with_incumbent"
    assert "warning" in stats
    assert "JobFailed" in stats["warning"]
    assert "Exception" in stats["warning"]

    # Without incumbent under JobFailed -> ERROR with detailed termination metadata
    sol_no_incumbent = JobFailedWithIncumbentSolution(
        solve_status="JobFailed",
        stop_cause="Fail",
        fail_status="Exception",
        has_solution=False,
    )
    with patch("docplex.cp.model.CpoModel.solve", return_value=sol_no_incumbent):
        res2, stats2 = solve_cplex_cp(N=3, edges=[(0, 1), (1, 2)], timeout=5.0, verbose=False)

    assert stats2["status"] == "ERROR"
    assert stats2["best_ub"] == -1
    assert stats2["termination_reason"] == "fail_stop"
    assert "Exception" in stats2["error"]
