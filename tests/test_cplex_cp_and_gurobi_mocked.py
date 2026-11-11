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
    def __init__(self, solve_status="Optimal", stop_cause=None, has_solution=True, assignment=None, obj_val=1, obj_bound=None):
        self._solve_status = solve_status
        self._stop_cause = stop_cause
        self._has_solution = has_solution
        self._assignment = assignment or [1, 2, 3]  # 1-indexed in docplex.cp
        self._obj_val = obj_val
        self._obj_bound = obj_bound

    def is_solution(self):
        return self._has_solution

    def get_solve_status(self):
        return self._solve_status

    def get_stop_cause(self):
        return self._stop_cause

    def get_objective_bound(self):
        return self._obj_bound

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
    res, stats = solve_cplex_cp(N=3, edges=[(0, 1)], verbose=False)
    assert stats["status"] == "UNAVAILABLE"
    assert stats["best_ub"] == -1
    assert "error" in stats


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
