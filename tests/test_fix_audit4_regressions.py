"""Regression tests for defects found in the fix/audit4 comprehensive audit.

See reports/fix_audit4_issues.md (A4-xx identifiers).
"""
import itertools
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import run_experiments as rexp  # noqa: E402


# ---------------------------------------------------------------------------
# A4-02: positional (N, edges=...) calls in solve_cplex_cp / solve_gurobi
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2)]


def _unavailable_cplex_cp(monkeypatch):
    monkeypatch.setattr(
        "baselines.cplex_cp_cbmp.is_cplex_cp_available",
        lambda: (False, "forced unavailable for argument-parsing test"),
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda f: f(3, edges=EDGES, verbose=False),
        lambda f: f(3, EDGES, verbose=False),
        lambda f: f(N=3, edges=EDGES, verbose=False),
        lambda f: f(3, N=3, edges=EDGES, verbose=False),
    ],
    ids=["N_positional_edges_kw", "both_positional", "both_kw", "N_twice_consistent"],
)
@pytest.mark.parametrize("solver_name", ["cplex_cp", "gurobi"])
def test_positional_graph_arguments_are_normalized(monkeypatch, solver_name, call):
    if solver_name == "cplex_cp":
        _unavailable_cplex_cp(monkeypatch)
        from baselines.cplex_cp_cbmp import solve_cplex_cp as fn
    else:
        import baselines.gurobi_cbmp as gm
        monkeypatch.setattr(gm, "is_gurobi_available", lambda: (False, "forced unavailable"))
        fn = gm.solve_gurobi
    res = call(fn)
    stats = res.stats
    assert stats["N"] == 3
    assert stats["num_edges"] == 2
    assert stats["status"] in ("UNAVAILABLE", "ERROR")


@pytest.mark.parametrize("solver_name", ["cplex_cp", "gurobi"])
def test_conflicting_positional_N_raises(monkeypatch, solver_name):
    if solver_name == "cplex_cp":
        _unavailable_cplex_cp(monkeypatch)
        from baselines.cplex_cp_cbmp import solve_cplex_cp as fn
    else:
        from baselines.gurobi_cbmp import solve_gurobi as fn
    with pytest.raises(ValueError, match="Conflicting values for N"):
        fn(3, 4, edges=EDGES, verbose=False)


# ---------------------------------------------------------------------------
# A4-03: relabel_seed requested for a backend that cannot relabel
# ---------------------------------------------------------------------------

def _task(method, seed):
    return {
        "method": method,
        "dataset": "14_bus",
        "solver": method,
        "flags": (True,),
        "relabel_seed": seed,
        "config_dict": {"sb_rotation": True},
        "timeout": 5,
    }


@pytest.mark.parametrize("method", ["cplex", "cplex_cp", "gurobi"])
def test_build_task_cmd_rejects_relabel_for_unsupported_baselines(method):
    with pytest.raises(ValueError, match="does not support relabel_seed"):
        rexp.build_task_cmd(_task(method, 3), "/tmp/unused.jsonl")


@pytest.mark.parametrize("method", ["cplex", "cplex_cp", "gurobi"])
def test_build_task_cmd_without_seed_still_works_for_baselines(method):
    cmd = rexp.build_task_cmd(_task(method, None), "/tmp/unused.jsonl")
    assert "--relabel-seed" not in cmd


def test_build_task_cmd_cpsat_keeps_relabel_seed():
    t = _task("cpsat", 4)
    t["config_dict"] = {"sb_rotation": True, "num_workers": 1}
    cmd = rexp.build_task_cmd(t, "/tmp/unused.jsonl")
    i = cmd.index("--relabel-seed")
    assert cmd[i + 1] == "4"


def test_plan_rejects_relabel_seeds_for_cplex_method():
    config = {
        "defaults": {"timeout": 5, "repetitions": 1},
        "groups": {
            "bad": {
                "datasets": [{"name": "14_bus", "relabel_seeds": [0]}],
                "methods": [{"name": "cplex", "type": "cplex"}],
            }
        },
    }
    all_ds = [{"name": "14_bus", "N": 14, "M": 20, "path": "datasets/14_bus.mtx"}]
    with pytest.raises(ValueError, match="does not support relabel_seed"):
        rexp.build_experiment_plan(config, all_ds, selected_group="bad")


def test_shipped_config_plan_has_no_unsupported_relabel_tasks():
    import yaml
    with open(os.path.join(ROOT, "experiments", "configs.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    all_ds = rexp.discover_datasets(os.path.join(ROOT, "datasets"))
    plan, _ = rexp.build_experiment_plan(config, all_ds, selected_group="all")
    for t in plan:
        if t.get("relabel_seed") is not None:
            assert t["method"] in rexp.RELABEL_SUPPORTED_METHODS


# ---------------------------------------------------------------------------
# A4-01: map_cplex_status without time-limit evidence and without incumbent
# ---------------------------------------------------------------------------

def test_cplex_limit_without_time_evidence_or_incumbent_is_error():
    from baselines.cplex_cbmp import map_cplex_status
    assert map_cplex_status(None, -1, True, False) == ("ERROR", False)
    assert map_cplex_status("time limit exceeded", -1, True, False) == ("TIMEOUT", False)
    assert map_cplex_status(None, 107, True, False) == ("TIMEOUT", False)


# ---------------------------------------------------------------------------
# Independent brute-force oracle (no project code) vs. SAT and CP-SAT
# ---------------------------------------------------------------------------

def _oracle(n, edges):
    es = {(min(u, v), max(u, v)) for u, v in edges if u != v}
    if not es:
        return 0
    best = n
    for p in itertools.permutations(range(n)):
        w = max(min(abs(p[u] - p[v]), n - abs(p[u] - p[v])) for u, v in es)
        best = min(best, w)
    return best


def _random_graphs(count, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        n = rng.randint(3, 7)
        all_e = list(itertools.combinations(range(n), 2))
        es = rng.sample(all_e, rng.randint(1, len(all_e)))
        out.append((n, es + [(es[0][1], es[0][0]), (0, 0)]))  # dup + self-loop noise
    return out


@pytest.mark.parametrize("n,edges", _random_graphs(15, 4242))
def test_sat_and_cpsat_match_independent_oracle(n, edges):
    from cbmp import solve
    from baselines.cpsat_cbmp import solve_cpsat
    exp = _oracle(n, edges)
    for kw in (
        dict(encoding="order", sb_rotation=True, sb_reflection=True, use_heuristic_initial_ub=True),
        dict(encoding="pairwise", sb_rotation=True, sb_reflection=True),
        dict(encoding="order", sb_rotation=False, solving_mode="iterative"),
    ):
        r = solve(N=n, edges=edges, verbose=False, **kw)
        assert int(r) == exp, kw
        assert r.stats["status"] == "OPTIMAL"
    r = solve_cpsat(N=n, edges=edges, timeout=30, verbose=False)
    assert int(r) == exp
    assert r.stats["status"] == "OPTIMAL"


# ---------------------------------------------------------------------------
# A4-04: datasets/MANIFEST.md must agree with the actual .mtx files
# ---------------------------------------------------------------------------

def _manifest_rows():
    import re
    rows = {}
    with open(os.path.join(ROOT, "datasets", "MANIFEST.md"), encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\| `([^`]+\.mtx)` \| [^|]+ \| (\d+) \| (\d+) \| ([0-9.]+) \|", line)
            if m:
                rows[m.group(1)] = (int(m.group(2)), int(m.group(3)), float(m.group(4)))
    return rows


def test_manifest_edge_counts_match_files():
    import glob
    from graph_utils import load_graph_from_mtx
    rows = _manifest_rows()
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "datasets", "*.mtx")))
    assert sorted(rows) == files
    for fname, (n, m, dens) in rows.items():
        N, E, _ = load_graph_from_mtx(os.path.join(ROOT, "datasets", fname))
        assert (N, len(E)) == (n, m), fname
        assert abs(2 * len(E) / (N * (N - 1)) - dens) < 5e-5, fname


@pytest.mark.parametrize("path", sorted(__import__("glob").glob(os.path.join(ROOT, "datasets", "*.mtx"))))
def test_raw_mtx_integrity_independent_parser(path):
    """Independent raw parse: header counts, index range, no duplicate entries,
    entries == N diagonal + |E| off-diagonal, graph connected."""
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("%") and ln.strip()]
    nr, nc, nnz = map(int, lines[0].split()[:3])
    assert nr == nc
    ent = [tuple(map(int, ln.split()[:2])) for ln in lines[1:]]
    assert len(ent) == nnz
    assert all(1 <= i <= nr and 1 <= j <= nr for i, j in ent)
    off = {(min(i, j), max(i, j)) for i, j in ent if i != j}
    diag = [i for i, j in ent if i == j]
    assert len(off) + len(diag) == nnz  # no duplicate undirected entries
    from graph_utils import load_graph_from_mtx
    N, E, _ = load_graph_from_mtx(path)
    assert N == nr and len(E) == len(off)
    adj = {v: set() for v in range(1, nr + 1)}
    for i, j in off:
        adj[i].add(j)
        adj[j].add(i)
    seen, stack = {1}, [1]
    while stack:
        for w in adj[stack.pop()]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    assert len(seen) == nr, "graph is disconnected"


# ---------------------------------------------------------------------------
# A4-05: planned canonical key for CPLEX/CP/Gurobi must equal the key of the
# record the baseline actually writes (sb_rotation defaults to True in CLIs)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m_type,solver", [("cplex", "cplex"), ("cplex_cp", "cplex_cp"), ("gurobi", "gurobi")])
@pytest.mark.parametrize("method_cfg_extra", [{}, {"sb_rotation": True}, {"sb_rotation": False}])
def test_planned_key_matches_recorded_key_for_baselines(m_type, solver, method_cfg_extra):
    from analysis.make_tables import get_canonical_key
    config = {
        "defaults": {"timeout": 7, "repetitions": 1},
        "groups": {"g": {"datasets": ["14_bus"], "methods": [dict(name="x", type=m_type, **method_cfg_extra)]}},
    }
    all_ds = [{"name": "14_bus", "N": 14, "M": 20, "path": "datasets/14_bus.mtx"}]
    plan, _ = rexp.build_experiment_plan(config, all_ds, selected_group="g")
    assert len(plan) == 1
    task = plan[0]
    cmd = rexp.build_task_cmd(task, "/tmp/unused.jsonl")
    expected_rot = method_cfg_extra.get("sb_rotation", True)
    assert ("--sb-rotation" if expected_rot else "--no-sb-rotation") in cmd
    # What the baseline CLI writes (see stats["config"] in baselines/*_cbmp.py)
    record = {"dataset": "14_bus", "solver": solver, "config": {"time_limit": 7.0, "sb_rotation": expected_rot}}
    assert get_canonical_key(record) == task["canonical_key"]


def test_resume_skips_completed_cplex_task(tmp_path):
    import json
    from analysis.make_tables import get_canonical_key
    config = {
        "defaults": {"timeout": 7, "repetitions": 1},
        "groups": {"g": {"datasets": ["14_bus"], "methods": [{"name": "cplex", "type": "cplex"}]}},
    }
    all_ds = [{"name": "14_bus", "N": 14, "M": 20, "path": "datasets/14_bus.mtx"}]
    plan, _ = rexp.build_experiment_plan(config, all_ds, selected_group="g")
    rec = {"dataset": "14_bus", "N": 14, "solver": "cplex", "status": "OPTIMAL", "best_ub": 3, "best_lb": 3,
           "config": {"time_limit": 7.0, "sb_rotation": True}, "rep": 1, "groups": ["g"]}
    runs = tmp_path / "runs.jsonl"
    runs.write_text(json.dumps(rec) + "\n")
    existing = rexp.load_existing_runs(str(runs))
    assert (plan[0]["canonical_key"], 1) in existing
    assert get_canonical_key(rec) == plan[0]["canonical_key"]
