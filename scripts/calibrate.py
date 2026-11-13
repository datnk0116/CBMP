#!/usr/bin/env python3
"""
scripts/calibrate.py

Run SAT solver (Cadical195, rotation+reflection+heuristic) on candidate subgraph instances
with a 120s timeout to calibrate runtime, memory, and difficulty.
Outputs to calibration/ (not results/).
"""

import os
import sys
import json
import time
import threading
import subprocess
from pathlib import Path
from scipy.io import mmread

DATASETS = [
    "14_bus",
    "30_bus",
    "494_bus_n40",
    "494_bus_n60",
    "494_bus_n80",
    "494_bus_n100",
    "494_bus_n150",
    "662_bus_n40",
    "662_bus_n60",
    "662_bus_n80",
    "662_bus_n100",
    "662_bus_n150",
]

def get_graph_meta(dataset_name, dataset_dir="datasets"):
    filepath = os.path.join(dataset_dir, f"{dataset_name}.mtx")
    try:
        mat = mmread(filepath, spmatrix=True)
    except TypeError:
        mat = mmread(filepath)
    cx = mat.tocoo()
    n = cx.shape[0]
    edges = set()
    for u, v in zip(cx.row, cx.col):
        u_int, v_int = int(u), int(v)
        if u_int != v_int:
            edges.add((min(u_int, v_int), max(u_int, v_int)))
    return n, len(edges)

def monitor_process_memory(proc, max_mem_box, stop_event):
    """Monitor peak RSS of process and all its child processes in KB."""
    while not stop_event.is_set():
        try:
            # Get RSS for proc and any descendants
            out = subprocess.check_output(
                ["pgrep", "-P", str(proc.pid)], text=True, stderr=subprocess.DEVNULL
            ).strip()
            pids = [proc.pid]
            if out:
                for line in out.splitlines():
                    pids.append(int(line.strip()))
            
            total_rss = 0
            for pid in pids:
                try:
                    rss_str = subprocess.check_output(
                        ["ps", "-o", "rss=", "-p", str(pid)], text=True, stderr=subprocess.DEVNULL
                    ).strip()
                    if rss_str:
                        total_rss += int(rss_str.split()[0])
                except Exception:
                    pass
            if total_rss > max_mem_box[0]:
                max_mem_box[0] = total_rss
        except Exception:
            pass
        time.sleep(0.1)

def run_calibration(datasets=DATASETS, timeout=120.0, output_dir="calibration"):
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "calibration.jsonl")
    csv_path = os.path.join(output_dir, "calibration.csv")

    results = []
    print("=" * 115)
    print(f"{'Dataset':<16} | {'N':>4} | {'|E|':>5} | {'Status':<8} | {'Best UB':>7} | {'Last UB':>7} | {'Iters':>5} | {'Time (s)':>9} | {'Clauses':>9} | {'RAM (MB)':>8}")
    print("-" * 115)

    for ds in datasets:
        n, m = get_graph_meta(ds)
        temp_json = os.path.join(output_dir, f"_temp_{os.getpid()}_{ds}.json")
        if os.path.exists(temp_json):
            os.remove(temp_json)

        cmd = [
            sys.executable,
            "main.py",
            ds,
            "--timeout", str(timeout),
            "--solver", "cadical195",
            "--sb-rotation",
            "--sb-reflection",
            "--heuristic-ub",
            "--json-out", temp_json,
        ]

        max_mem_box = [0]
        stop_event = threading.Event()
        t0 = time.perf_counter()

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        monitor_thread = threading.Thread(target=monitor_process_memory, args=(proc, max_mem_box, stop_event))
        monitor_thread.daemon = True
        monitor_thread.start()

        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout + 10)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            timed_out = True

        stop_event.set()
        monitor_thread.join(timeout=1.0)
        wall_time = round(time.perf_counter() - t0, 4)

        record = None
        if os.path.exists(temp_json) and os.path.getsize(temp_json) > 0:
            try:
                with open(temp_json, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]
                    if lines:
                        record = json.loads(lines[-1])
            except Exception:
                pass
            try:
                os.remove(temp_json)
            except Exception:
                pass

        if record is None:
            status = "TIMEOUT" if timed_out else "ERROR"
            best_ub = -1
            solve_time = wall_time
            num_clauses = 0
            n_sat_iterations = 0
        else:
            status = record.get("status", "UNKNOWN")
            best_ub = record.get("best_ub", -1)
            # For TIMEOUT rows, solve_time records wall time (~120s)
            if status == "TIMEOUT":
                solve_time = wall_time
            else:
                solve_time = record.get("solve_time", wall_time)
            num_clauses = record.get("num_clauses", 0)
            n_sat_iterations = record.get("num_iterations", len(record.get("per_iter_times", [])))

        last_iter_ub = best_ub
        peak_ram_mb = round(max_mem_box[0] / 1024, 2)

        res_entry = {
            "dataset": ds,
            "N": n,
            "num_edges": m,
            "status": status,
            "best_ub": best_ub,
            "last_iter_ub": last_iter_ub,
            "n_sat_iterations": n_sat_iterations,
            "solve_time": solve_time,
            "wall_time": wall_time,
            "num_clauses": num_clauses,
            "peak_ram_mb": peak_ram_mb,
        }
        results.append(res_entry)

        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(res_entry) + "\n")

        clauses_str = f"{num_clauses:,}" if num_clauses else "-"
        ub_str = str(best_ub) if best_ub is not None and best_ub >= 0 else "-"
        last_ub_str = str(last_iter_ub) if last_iter_ub is not None and last_iter_ub >= 0 else "-"
        print(f"{ds:<16} | {n:>4} | {m:>5} | {status:<8} | {ub_str:>7} | {last_ub_str:>7} | {n_sat_iterations:>5} | {solve_time:>9.3f} | {clauses_str:>9} | {peak_ram_mb:>8.1f}")
        sys.stdout.flush()

    # Write CSV summary
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("dataset,N,num_edges,status,best_ub,last_iter_ub,n_sat_iterations,solve_time,wall_time,num_clauses,peak_ram_mb\n")
        for r in results:
            f.write(f"{r['dataset']},{r['N']},{r['num_edges']},{r['status']},{r['best_ub']},{r['last_iter_ub']},{r['n_sat_iterations']},{r['solve_time']:.4f},{r['wall_time']:.4f},{r['num_clauses']},{r['peak_ram_mb']}\n")

    print("=" * 115)
    print(f"Calibration completed! Results saved to {jsonl_path} and {csv_path}\n")
    return results

if __name__ == "__main__":
    run_calibration()

