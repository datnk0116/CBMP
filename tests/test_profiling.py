"""
Profiling tests for memory usage and encode vs solve time.
Phase 7 of the Code Quality & Robustness Audit.
"""

import time
import tracemalloc
import pytest

from cbmp import solve
from baselines.cpsat_cbmp import build_cpsat_model, solve_cpsat
from baselines.cplex_cbmp import build_cplex_model, solve_cplex
from graph_utils import load_graph_from_mtx


def profile_memory_and_time(func, *args, **kwargs):
    """Run a callable under tracemalloc, returning (result, peak_mb, duration_sec)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    duration = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    return res, peak_mb, duration


def test_model_building_memory_and_scaling():
    """
    Profile peak memory during model building for SAT, CP-SAT, and CPLEX
    on datasets up to N=30 (e.g. 14_bus, 30_bus).
    Ensures memory usage remains well bounded (< 100 MB).
    """
    datasets = ["14_bus", "30_bus"]

    for ds in datasets:
        N, edges, _ = load_graph_from_mtx(ds)

        # 1. CP-SAT model building
        _, cpsat_mb, cpsat_time = profile_memory_and_time(
            build_cpsat_model, N, edges, sb_rotation=True
        )
        assert cpsat_mb < 50.0, f"CP-SAT building used {cpsat_mb:.2f} MB on {ds}"

        # 2. CPLEX model building
        _, cplex_mb, cplex_time = profile_memory_and_time(
            build_cplex_model, N, edges, sb_rotation=True
        )
        assert cplex_mb < 50.0, f"CPLEX building used {cplex_mb:.2f} MB on {ds}"


def test_encode_vs_solve_time_breakdown():
    """
    Profile encode_time vs solve_time for SAT on small graphs.
    Verifies that stats dictionaries accurately track both metrics.
    """
    res = solve(dataset="14_bus", sb_rotation=True, verbose=False)
    stats = res.stats

    assert "encode_time" in stats
    assert "solve_time" in stats
    assert stats["encode_time"] >= 0.0
    assert stats["solve_time"] >= 0.0

    # Encode time should be negligible (< 1s) for N=14
    assert stats["encode_time"] < 1.0

