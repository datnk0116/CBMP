import os
import sys
import glob
import json
import time
import math
import yaml
import signal
import logging
import argparse
import platform
import subprocess
from pathlib import Path

# Setup module logger
logger = logging.getLogger("experiments")

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.make_tables import normalize_status, validate_best_ub

def get_dataset_info(filepath):
    """Quickly read N and M from MatrixMarket header without parsing entire matrix."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    return int(parts[0]), int(parts[2])
                elif len(parts) >= 2:
                    return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return 0, 0

def discover_datasets(dataset_dir="datasets", min_n=1, max_n=None, allowed_names=None):
    """Scan dataset_dir for .mtx files, filtered and sorted by N."""
    mtx_files = sorted(glob.glob(os.path.join(dataset_dir, "*.mtx")))
    discovered = []

    for path in mtx_files:
        name = Path(path).stem
        if allowed_names and name not in allowed_names:
            continue
        n, m = get_dataset_info(path)
        if min_n is not None and n < min_n:
            continue
        if max_n is not None and n > max_n:
            continue
        discovered.append({
            "name": name,
            "path": path,
            "N": n,
            "M": m,
        })

    discovered.sort(key=lambda d: (d["N"], d["name"]))
    return discovered

def get_git_commit_info():
    """Retrieve current git commit hash and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "unknown"
    try:
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        is_dirty = bool(status_out)
    except Exception:
        is_dirty = False
    return commit, is_dirty

def get_git_commit_hash():
    """Retrieve current git commit hash (with -dirty suffix if repo is dirty)."""
    commit, is_dirty = get_git_commit_info()
    if is_dirty and commit != "unknown":
        return f"{commit}-dirty"
    return commit

def collect_environment_info(output_path="results_v2/env.json", timeout=None, config_path=None, record_metadata=False):
    """Collect hardware and software environment details."""
    info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "os_platform": platform.platform(),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }

    if record_metadata:
        info["git_commit"] = get_git_commit_hash()
        if timeout is not None:
            info["timeout"] = timeout
        if config_path is not None:
            try:
                info["config_path"] = os.path.relpath(config_path, start=str(REPO_ROOT))
            except Exception:
                info["config_path"] = str(config_path)

    # CPU model
    try:
        if platform.system() == "Darwin":
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            info["cpu_model"] = brand
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        info["cpu_model"] = line.split(":")[1].strip()
                        break
    except Exception:
        info["cpu_model"] = "Unknown"

    # RAM
    try:
        if platform.system() == "Darwin":
            ram_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            info["ram_gb"] = round(ram_bytes / (1024 ** 3), 2)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        info["ram_gb"] = round(kb / (1024 ** 2), 2)
                        break
    except Exception:
        info["ram_gb"] = "Unknown"

    # Solver & Package versions
    packages = ["python-sat", "scipy", "numpy", "ortools", "docplex", "cplex", "pyyaml", "pytest"]
    pkg_versions = {}
    for p in packages:
        try:
            mod_name = "yaml" if p == "pyyaml" else ("pysat" if p == "python-sat" else p)
            m = __import__(mod_name)
            pkg_versions[p] = getattr(m, "__version__", "installed")
        except Exception:
            pkg_versions[p] = "not installed"
    info["packages"] = pkg_versions

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    return info

def get_canonical_key(record):
    """
    Extracts canonical (dataset, method, solver, flags, relabel_seed) tuple from a result dictionary.
    """
    dataset = record.get("dataset")
    solver = str(record.get("solver", "")).lower()
    cfg = record.get("config", {})
    relabel_seed = record.get("relabel_seed")
    if relabel_seed is None and isinstance(cfg, dict):
        relabel_seed = cfg.get("relabel_seed", None)
    if solver in ("cp_sat", "cpsat"):
        method = "cpsat"
        workers = cfg.get("num_workers", 1)
        if "sb_rotation" in cfg:
            sb_rot = bool(cfg["sb_rotation"])
        elif "sb_rotation" in record:
            sb_rot = bool(record["sb_rotation"])
        else:
            sb_rot = True
        flags = (workers, sb_rot)
    elif solver == "cplex":
        method = "cplex"
        if "sb_rotation" in cfg:
            flags = (bool(cfg["sb_rotation"]),)
        else:
            flags = ()
    elif solver in ("cplex_cp", "cpoptimizer"):
        method = "cplex_cp"
        if "sb_rotation" in cfg:
            flags = (bool(cfg["sb_rotation"]),)
        else:
            flags = ()
    elif solver in ("gurobi", "grb"):
        method = "gurobi"
        if "sb_rotation" in cfg:
            flags = (bool(cfg["sb_rotation"]),)
        else:
            flags = ()
    else:
        method = "sat"
        rot = bool(record.get("sb_rotation") if "sb_rotation" in record else cfg.get("sb_rotation", False))
        ref = bool(record.get("sb_reflection") if "sb_reflection" in record else cfg.get("sb_reflection", False))
        heur = bool(record.get("heuristic_ub") if "heuristic_ub" in record else cfg.get("heuristic_ub", False))
        enc = str(record.get("encoding") or cfg.get("encoding", "order")).lower()
        mode = str(record.get("solving_mode") or cfg.get("solving_mode", "incremental")).lower()
        eo = str(record.get("eo_strategy") or cfg.get("eo_strategy", "commander")).lower()
        if enc != "order" or mode != "incremental" or eo != "commander":
            flags = (rot, ref, heur, enc, mode, eo)
        else:
            flags = (rot, ref, heur)
    return (dataset, method, solver, flags, relabel_seed)

def is_run_completed(record: Dict[str, Any], retry_failed: bool = False) -> bool:
    """
    Determine whether an existing run record should be considered complete.

    Semantics:
    - Normal resume (retry_failed=False):
        All existing records in runs.jsonl are preserved and treated as completed.
    - Retry failed mode (retry_failed=True):
        * Complete proofs & hard bounds (NEVER retry):
            - OPTIMAL: Optimality is proven mathematically.
            - INFEASIBLE: Problem/bound is proven infeasible.
            - SIZE_LIMIT: Graph size exceeded configuration limits.
        * Stable environment/license barriers (DO NOT retry in same environment):
            - UNAVAILABLE: Solver binary/engine is not installed.
            - LICENSE_ERROR: Commercial license is missing or invalid.
        * Incumbent preservation (PRESERVED):
            - FEASIBLE with valid best_ub >= 0: Valid incumbent is preserved.
        * Retryable transient failures (RETRIED):
            - TIMEOUT: Solver exceeded time limit.
            - ERROR: Solver crashed, raised an unhandled exception, or failed validation.
            - INTERRUPTED: Process killed or interrupted by external signal.
            - FEASIBLE with missing or negative best_ub.
    """
    if not retry_failed:
        return True

    status = normalize_status(record.get("status"))
    if not status:
        status = str(record.get("status", "")).strip().upper()

    # 1. Conclusive proofs and limits
    if status in ("OPTIMAL", "INFEASIBLE", "SIZE_LIMIT"):
        return True

    # 2. Solver installation / license unavailability
    if status in ("UNAVAILABLE", "LICENSE_ERROR"):
        return True

    # 3. Preserved feasible incumbents
    if status == "FEASIBLE":
        ub_val = validate_best_ub(record.get("best_ub"))
        if ub_val is not None and not record.get("validation_error"):
            return True
        return False

    # 4. Transient failures are retryable
    return False

def load_existing_runs(runs_jsonl_path, retry_failed=False):
    """
    Load runs already recorded in runs.jsonl mapping (canonical_key, rep) -> record.
    If retry_failed is True, completed runs are filtered via is_run_completed(..., retry_failed=True).
    """
    if not os.path.exists(runs_jsonl_path):
        return {}
    completed = {}
    valid_count = 0
    skipped_count = 0
    try:
        with open(runs_jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("_type") == "metadata" or "dataset" not in data:
                        continue
                    key = get_canonical_key(data)
                    rep = int(data.get("rep", 1))
                    if is_run_completed(data, retry_failed=retry_failed):
                        completed[(key, rep)] = data
                        valid_count += 1
                except (json.JSONDecodeError, ValueError) as e:
                    skipped_count += 1
                    logger.warning("Corrupt or invalid JSON at %s:%d: %s (skipped)", runs_jsonl_path, line_no, e)
                except Exception as e:
                    skipped_count += 1
                    logger.warning("Error processing record at %s:%d: %s (skipped)", runs_jsonl_path, line_no, e)
    except Exception as e:
        logger.error("Could not read existing runs from %s: %s", runs_jsonl_path, e)

    if skipped_count > 0:
        logger.warning(
            "Loaded %d completed runs from %s (%d invalid/corrupt lines skipped)",
            valid_count, runs_jsonl_path, skipped_count
        )
    return completed

def filter_datasets_for_group(all_datasets, group_cfg, defaults, cli_datasets=None, cli_min_n=None, cli_max_n=None):
    """
    Filters datasets for a specific group based on:
    - CLI overrides if provided (cli_datasets, cli_min_n, cli_max_n)
    - Group config (datasets, min_n/min_N, max_n/max_N)
    - Fallback to defaults.filters
    """
    default_filters = defaults.get("filters", {})

    # 1. Dataset name filter & relabel_seeds map
    seeds_map = {}
    if cli_datasets is not None:
        allowed = set(s.strip() for s in cli_datasets.split(","))
    else:
        grp_ds = group_cfg.get("datasets")
        if grp_ds is None:
            def_ds = default_filters.get("datasets")
            if isinstance(def_ds, list):
                allowed = set(def_ds)
            elif def_ds == "all" or def_ds is None:
                allowed = None
            else:
                allowed = set([str(def_ds)])
        elif grp_ds == "all":
            allowed = None
        elif isinstance(grp_ds, (list, tuple, set)):
            allowed = set()
            for item in grp_ds:
                if isinstance(item, str):
                    allowed.add(item)
                elif isinstance(item, dict) and "name" in item:
                    name = item["name"]
                    allowed.add(name)
                    if "relabel_seeds" in item:
                        seeds_map[name] = item["relabel_seeds"]
                elif isinstance(item, dict):
                    for k, v in item.items():
                        allowed.add(k)
                        if isinstance(v, list):
                            seeds_map[k] = v
        elif isinstance(grp_ds, dict):
            allowed = set(grp_ds.keys())
            for k, v in grp_ds.items():
                if isinstance(v, list):
                    seeds_map[k] = v
        elif isinstance(grp_ds, str):
            allowed = set([s.strip() for s in grp_ds.split(",")])
        else:
            allowed = None

    # Populate seeds_map also if grp_ds has seeds configured and cli_datasets filtered it
    if cli_datasets is not None and isinstance(group_cfg.get("datasets"), (list, tuple, set, dict)):
        raw_ds = group_cfg.get("datasets")
        if isinstance(raw_ds, (list, tuple, set)):
            for item in raw_ds:
                if isinstance(item, dict) and "name" in item and "relabel_seeds" in item:
                    seeds_map[item["name"]] = item["relabel_seeds"]
                elif isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, list):
                            seeds_map[k] = v
        elif isinstance(raw_ds, dict):
            for k, v in raw_ds.items():
                if isinstance(v, list):
                    seeds_map[k] = v

    # 2. Min N
    if cli_min_n is not None:
        eff_min_n = cli_min_n
    else:
        eff_min_n = group_cfg.get("min_n", group_cfg.get("min_N"))
        if eff_min_n is None:
            eff_min_n = default_filters.get("min_N", 1)

    # 3. Max N
    if cli_max_n is not None:
        eff_max_n = cli_max_n
    else:
        eff_max_n = group_cfg.get("max_n", group_cfg.get("max_N"))
        if eff_max_n is None:
            eff_max_n = default_filters.get("max_N", None)

    filtered = []
    for d in all_datasets:
        if allowed is not None and d["name"] not in allowed:
            continue
        if eff_min_n is not None and d["N"] < eff_min_n:
            continue
        if eff_max_n is not None and d["N"] > eff_max_n:
            continue
        d_out = dict(d)
        if d["name"] in seeds_map:
            d_out["relabel_seeds"] = seeds_map[d["name"]]
        filtered.append(d_out)

    return filtered

def build_experiment_plan(
    config,
    all_datasets,
    selected_group=None,
    timeout_override=None,
    repetitions_override=None,
    cli_datasets=None,
    cli_min_n=None,
    cli_max_n=None,
):
    """
    Constructs the deduplicated list of planned tasks according to configs.yaml.
    Two tasks are identical if they have the same (dataset, method, solver, flags).
    Returns (plan, group_stats).
    """
    defaults = config.get("defaults", {})
    default_timeout = defaults.get("timeout", 600)
    default_repetitions = defaults.get("repetitions", 1)
    groups = config.get("groups", {})

    groups_to_run = [selected_group] if (selected_group and selected_group != "all") else list(groups.keys())

    tasks_map = {}
    group_stats = {}

    for grp in groups_to_run:
        if grp not in groups:
            continue
        grp_cfg = groups[grp]
        grp_timeout = timeout_override or grp_cfg.get("timeout", default_timeout)
        grp_reps = repetitions_override if repetitions_override is not None else grp_cfg.get("repetitions", default_repetitions)

        grp_datasets = filter_datasets_for_group(
            all_datasets=all_datasets,
            group_cfg=grp_cfg,
            defaults=defaults,
            cli_datasets=cli_datasets,
            cli_min_n=cli_min_n,
            cli_max_n=cli_max_n,
        )

        grp_tasks_count = 0

        if grp == "per_dataset":
            solver = grp_cfg.get("solver", "cadical195")
            rot = bool(grp_cfg.get("sb_rotation", True))
            ref = bool(grp_cfg.get("sb_reflection", True))
            heur = bool(grp_cfg.get("heuristic_ub", True))
            method = "sat"
            flags = (rot, ref, heur)

            for d in grp_datasets:
                grp_tasks_count += 1
                key = (d["name"], method, solver.lower(), flags, None)
                if key not in tasks_map:
                    tasks_map[key] = {
                        "canonical_key": key,
                        "dataset": d["name"],
                        "N": d["N"],
                        "M": d["M"],
                        "method": method,
                        "solver": solver,
                        "flags": flags,
                        "relabel_seed": None,
                        "config_dict": {
                            "sb_rotation": rot,
                            "sb_reflection": ref,
                            "heuristic_ub": heur,
                            "time_limit": grp_timeout,
                            "relabel_seed": None,
                        },
                        "timeout": grp_timeout,
                        "group_reps": {grp: grp_reps},
                        "groups": [grp],
                    }
                else:
                    tasks_map[key]["group_reps"][grp] = grp_reps
                    if grp not in tasks_map[key]["groups"]:
                        tasks_map[key]["groups"].append(grp)
                    tasks_map[key]["timeout"] = max(tasks_map[key]["timeout"], grp_timeout)

        elif grp == "solvers" or grp.startswith("solvers_"):
            rot = bool(grp_cfg.get("sb_rotation", True))
            ref = bool(grp_cfg.get("sb_reflection", True))
            heur = bool(grp_cfg.get("heuristic_ub", True))
            solvers = grp_cfg.get("solvers", ["cadical195"])
            method = "sat"
            flags = (rot, ref, heur)

            for d in grp_datasets:
                seeds = d.get("relabel_seeds", [None])
                for seed in seeds:
                    for s in solvers:
                        grp_tasks_count += 1
                        key = (d["name"], method, s.lower(), flags, seed)
                        if key not in tasks_map:
                            tasks_map[key] = {
                                "canonical_key": key,
                                "dataset": d["name"],
                                "N": d["N"],
                                "M": d["M"],
                                "method": method,
                                "solver": s,
                                "flags": flags,
                                "relabel_seed": seed,
                                "config_dict": {
                                    "sb_rotation": rot,
                                    "sb_reflection": ref,
                                    "heuristic_ub": heur,
                                    "time_limit": grp_timeout,
                                    "relabel_seed": seed,
                                },
                                "timeout": grp_timeout,
                                "group_reps": {grp: grp_reps},
                                "groups": [grp],
                            }
                        else:
                            tasks_map[key]["group_reps"][grp] = grp_reps
                            if grp not in tasks_map[key]["groups"]:
                                tasks_map[key]["groups"].append(grp)
                            tasks_map[key]["timeout"] = max(tasks_map[key]["timeout"], grp_timeout)

        elif "methods" in grp_cfg or grp == "methods" or grp.startswith("methods_") or grp.startswith("cpsat_") or grp == "encodings":
            methods = grp_cfg.get("methods", [])
            for d in grp_datasets:
                seeds = d.get("relabel_seeds", [None])
                for seed in seeds:
                    for m in methods:
                        grp_tasks_count += 1
                        m_type = m.get("type", "sat")
                        if m_type == "sat":
                            solver = m.get("solver", "cadical195")
                            rot = bool(m.get("sb_rotation", True))
                            ref = bool(m.get("sb_reflection", True))
                            heur = bool(m.get("heuristic_ub", True))
                            enc = str(m.get("encoding", "order")).lower()
                            mode = str(m.get("solving_mode", "incremental")).lower()
                            eo = str(m.get("eo_strategy", "commander")).lower()
                            method = "sat"
                            if enc != "order" or mode != "incremental" or eo != "commander":
                                flags = (rot, ref, heur, enc, mode, eo)
                            else:
                                flags = (rot, ref, heur)
                            key = (d["name"], method, solver.lower(), flags, seed)
                            cfg_dict = {
                                "sb_rotation": rot,
                                "sb_reflection": ref,
                                "heuristic_ub": heur,
                                "encoding": enc,
                                "solving_mode": mode,
                                "eo_strategy": eo,
                                "time_limit": grp_timeout,
                                "relabel_seed": seed,
                            }
                        elif m_type == "cpsat":
                            num_w = m.get("num_workers", 1)
                            method = "cpsat"
                            solver = "cp_sat"
                            sb_rot = bool(m.get("sb_rotation", True))
                            flags = (num_w, sb_rot)
                            key = (d["name"], method, solver, flags, seed)
                            cfg_dict = {
                                "num_workers": num_w,
                                "sb_rotation": sb_rot,
                                "time_limit": grp_timeout,
                                "relabel_seed": seed,
                            }
                        elif m_type == "cplex":
                            method = "cplex"
                            solver = "cplex"
                            sb_rot = bool(m["sb_rotation"]) if "sb_rotation" in m else None
                            flags = (sb_rot,) if sb_rot is not None else ()
                            key = (d["name"], method, solver, flags, seed)
                            cfg_dict = {
                                "time_limit": grp_timeout,
                                "relabel_seed": seed,
                            }
                            if sb_rot is not None:
                                cfg_dict["sb_rotation"] = sb_rot
                        elif m_type == "cplex_cp":
                            method = "cplex_cp"
                            solver = "cplex_cp"
                            sb_rot = bool(m["sb_rotation"]) if "sb_rotation" in m else None
                            flags = (sb_rot,) if sb_rot is not None else ()
                            key = (d["name"], method, solver, flags, seed)
                            cfg_dict = {
                                "time_limit": grp_timeout,
                                "relabel_seed": seed,
                            }
                            if sb_rot is not None:
                                cfg_dict["sb_rotation"] = sb_rot
                        elif m_type == "gurobi":
                            method = "gurobi"
                            solver = "gurobi"
                            sb_rot = bool(m["sb_rotation"]) if "sb_rotation" in m else None
                            flags = (sb_rot,) if sb_rot is not None else ()
                            key = (d["name"], method, solver, flags, seed)
                            cfg_dict = {
                                "time_limit": grp_timeout,
                                "relabel_seed": seed,
                            }
                            if sb_rot is not None:
                                cfg_dict["sb_rotation"] = sb_rot
                        else:
                            continue

                        if key not in tasks_map:
                            tasks_map[key] = {
                                "canonical_key": key,
                                "dataset": d["name"],
                                "N": d["N"],
                                "M": d["M"],
                                "method": method,
                                "solver": solver,
                                "flags": flags,
                                "relabel_seed": seed,
                                "config_dict": cfg_dict,
                                "timeout": grp_timeout,
                                "group_reps": {grp: grp_reps},
                                "groups": [grp],
                            }
                        else:
                            tasks_map[key]["group_reps"][grp] = grp_reps
                            if grp not in tasks_map[key]["groups"]:
                                tasks_map[key]["groups"].append(grp)
                            tasks_map[key]["timeout"] = max(tasks_map[key]["timeout"], grp_timeout)

        elif grp == "ablation" or grp.startswith("ablation_"):
            solver = grp_cfg.get("solver", "cadical195")
            variants = grp_cfg.get("variants", [])
            method = "sat"
            for d in grp_datasets:
                seeds = d.get("relabel_seeds", [None])
                for seed in seeds:
                    for var in variants:
                        grp_tasks_count += 1
                        rot = bool(var.get("sb_rotation", False))
                        ref = bool(var.get("sb_reflection", False))
                        heur = bool(var.get("heuristic_ub", False))
                        flags = (rot, ref, heur)
                        key = (d["name"], method, solver.lower(), flags, seed)
                        cfg_dict = {
                            "sb_rotation": rot,
                            "sb_reflection": ref,
                            "heuristic_ub": heur,
                            "time_limit": grp_timeout,
                            "relabel_seed": seed,
                        }
                        if key not in tasks_map:
                            tasks_map[key] = {
                                "canonical_key": key,
                                "dataset": d["name"],
                                "N": d["N"],
                                "M": d["M"],
                                "method": method,
                                "solver": solver,
                                "flags": flags,
                                "relabel_seed": seed,
                                "config_dict": cfg_dict,
                                "timeout": grp_timeout,
                                "group_reps": {grp: grp_reps},
                                "groups": [grp],
                            }
                        else:
                            tasks_map[key]["group_reps"][grp] = grp_reps
                            if grp not in tasks_map[key]["groups"]:
                                tasks_map[key]["groups"].append(grp)
                            tasks_map[key]["timeout"] = max(tasks_map[key]["timeout"], grp_timeout)

        if len(grp_datasets) > 0 and grp_tasks_count == 0:
            raise ValueError(
                f"Group '{grp}' has {len(grp_datasets)} dataset(s) configured, "
                f"but generated 0 tasks! Check group methods/variants/solvers configuration."
            )

        group_stats[grp] = {
            "datasets_count": len(grp_datasets),
            "tasks_count": grp_tasks_count,
            "repetitions": grp_reps,
            "runs_count": grp_tasks_count * grp_reps,
            "timeout": grp_timeout,
            "worst_case_seconds": grp_tasks_count * grp_reps * grp_timeout,
        }

    plan = list(tasks_map.values())
    for task in plan:
        task["repetitions"] = max(task["group_reps"].values())

    return plan, group_stats

def build_task_cmd(task, json_out_path):
    """Builds subprocess command line for a given task."""
    method = task["method"]
    timeout = task["timeout"]
    dataset = task["dataset"]
    cfg = task.get("config_dict", {})
    if method == "sat":
        rot = bool(cfg.get("sb_rotation", False))
        ref = bool(cfg.get("sb_reflection", False))
        heur = bool(cfg.get("heuristic_ub", False))
        enc = cfg.get("encoding", "order")
        mode = cfg.get("solving_mode", "incremental")
        eo = cfg.get("eo_strategy", "commander")
        cmd = [
            sys.executable, "main.py", dataset,
            "--timeout", str(timeout),
            "--solver", task["solver"],
            "--encoding", enc,
            "--solving-mode", mode,
            "--eo-strategy", eo,
            "--json-out", json_out_path,
        ]
        if rot: cmd.append("--sb-rotation")
        if ref: cmd.append("--sb-reflection")
        if heur: cmd.append("--heuristic-ub")
        if task.get("relabel_seed") is not None:
            cmd.extend(["--relabel-seed", str(task["relabel_seed"])])
        return cmd
    elif method == "cpsat":
        num_w = task["flags"][0] if task["flags"] else 1
        cmd = [
            sys.executable, "-m", "baselines.cpsat_cbmp", dataset,
            "--timeout", str(timeout),
            "--num-workers", str(num_w),
            "--json-out", json_out_path,
        ]
        sb_rot = task.get("config_dict", {}).get("sb_rotation", True)
        if len(task.get("flags", ())) > 1:
            sb_rot = task["flags"][1]
        cmd.append("--sb-rotation" if sb_rot else "--no-sb-rotation")
        if task.get("relabel_seed") is not None:
            cmd.extend(["--relabel-seed", str(task["relabel_seed"])])
        return cmd
    elif method in ("cplex", "cplex_cp", "gurobi"):
        mod_map = {
            "cplex": "baselines.cplex_cbmp",
            "cplex_cp": "baselines.cplex_cp_cbmp",
            "gurobi": "baselines.gurobi_cbmp",
        }
        cmd = [
            sys.executable, "-m", mod_map[method], dataset,
            "--timeout", str(timeout),
            "--json-out", json_out_path,
        ]
        if "sb_rotation" in task["config_dict"]:
            sb_rot = bool(task["config_dict"]["sb_rotation"])
            cmd.append("--sb-rotation" if sb_rot else "--no-sb-rotation")
        if task.get("relabel_seed") is not None and method in ("cpsat", "cplex_cp", "gurobi"):
            cmd.extend(["--relabel-seed", str(task["relabel_seed"])])
        return cmd
    else:
        raise ValueError(f"Unknown method: {method}")

def format_task_flags(task):
    """Human-readable representation of configuration flags."""
    if task["method"] == "sat":
        rot, ref, heur = task["flags"][:3]
        parts = []
        if len(task["flags"]) > 3:
            enc, mode, eo = task["flags"][3:6]
            if enc != "order": parts.append(enc)
            if mode != "incremental": parts.append(mode)
            if eo != "commander": parts.append(eo)
        if rot: parts.append("rot")
        if ref: parts.append("ref")
        if heur: parts.append("heur_ub")
        res = "+".join(parts) if parts else "none"
        if task.get("relabel_seed") is not None:
            res += f"+seed={task['relabel_seed']}"
        return res
    elif task["method"] == "cpsat":
        parts = [f"workers={task['flags'][0]}"]
        if len(task["flags"]) > 1:
            parts.append(f"rot={task['flags'][1]}")
        if task.get("relabel_seed") is not None:
            parts.append(f"seed={task['relabel_seed']}")
        return "+".join(parts)
    elif task["method"] in ("cplex", "cplex_cp", "gurobi"):
        parts = []
        if task.get("flags") and len(task["flags"]) > 0:
            parts.append(f"rot={task['flags'][0]}")
        else:
            parts.append("default")
        if task.get("relabel_seed") is not None:
            parts.append(f"seed={task['relabel_seed']}")
        return "+".join(parts)
    return "-"

def run_experiments(
    config_path="experiments/configs.yaml",
    group="all",
    timeout=None,
    repetitions=None,
    min_n=None,
    max_n=None,
    dataset_filter=None,
    resume=False,
    force_resume=False,
    retry_failed=False,
    allow_dirty=False,
    dry_run=False,
    output_dir="results_v2",
    record_metadata=False,
):
    from scripts.check_datasets import validate_datasets

    if retry_failed:
        resume = True

    commit_hash, is_dirty = get_git_commit_info()
    if is_dirty and not allow_dirty and not dry_run:
        msg = "Working directory has uncommitted changes (git_dirty=True). Commit all changes or pass --allow-dirty to proceed."
        print(f"[ERROR] {msg}", file=sys.stderr)
        raise ValueError(msg)

    # 1. Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    defaults = config.get("defaults", {})
    dataset_dir = defaults.get("dataset_dir", "datasets")

    # 1b. Validate datasets referenced in config
    try:
        validate_datasets(config_path=config_path, dataset_dir=dataset_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] Dataset validation failed:\n{e}", file=sys.stderr)
        sys.exit(1)

    # 2. Discover all available datasets in dataset_dir
    all_datasets = discover_datasets(dataset_dir=dataset_dir, min_n=1, max_n=None, allowed_names=None)
    if not all_datasets:
        print("No datasets found in dataset directory.")
        return

    # 3. Environment info (only for actual execution)
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
        env_file = os.path.join(output_dir, "env.json")
        effective_timeout = timeout if timeout is not None else defaults.get("timeout", 600)
        collect_environment_info(
            output_path=env_file,
            timeout=effective_timeout,
            config_path=config_path,
            record_metadata=record_metadata,
        )

    # 4. Build deduplicated execution plan
    plan, group_stats = build_experiment_plan(
        config=config,
        all_datasets=all_datasets,
        selected_group=group,
        timeout_override=timeout,
        repetitions_override=repetitions,
        cli_datasets=dataset_filter,
        cli_min_n=min_n,
        cli_max_n=max_n,
    )

    total_runs_after_dedup = sum(t["repetitions"] for t in plan)
    worst_case_seconds = sum(t["repetitions"] * t["timeout"] for t in plan)
    hours = int(worst_case_seconds // 3600)
    minutes = int((worst_case_seconds % 3600) // 60)
    seconds = int(worst_case_seconds % 60)
    time_str = f"{worst_case_seconds:.0f}s ({hours}h {minutes:02d}m {seconds:02d}s)"

    # 5. Handle --dry-run
    if dry_run:
        print("=" * 115)
        print(f"{'Idx':>3} | {'Dataset':<12} | {'N':>3} | {'Method':<7} | {'Solver':<12} | {'Flags':<18} | {'Timeout':>7} | {'Reps':>4} | {'Groups'}")
        print("-" * 115)
        for idx, item in enumerate(plan, start=1):
            groups_str = ",".join(item["groups"])
            flags_str = format_task_flags(item)
            print(f"{idx:>3} | {item['dataset']:<12} | {item['N']:>3} | {item['method']:<7} | {item['solver']:<12} | {flags_str:<18} | {item['timeout']:>6.0f}s | {item['repetitions']:>4} | {groups_str}")
        print("=" * 115)

        print("\n[DRY-RUN] Planned Runs per Group (before deduplication):")
        total_before_dedup = 0
        for grp_name, stats in group_stats.items():
            total_before_dedup += stats["runs_count"]
            print(f"  - {grp_name:<14}: {stats['runs_count']:>3} runs ({stats['datasets_count']} datasets, {stats['tasks_count']} configs x {stats['repetitions']} rep(s), timeout: {stats['timeout']:.0f}s)")
        print(f"  Total planned runs before deduplication: {total_before_dedup}")

        print(f"\n[DRY-RUN] Deduplication Summary:")
        print(f"  - Total unique configurations: {len(plan)}")
        print(f"  - Total runs to execute after deduplication: {total_runs_after_dedup}")
        print(f"  - Total worst-case time after deduplication: {time_str}")
        print("  Exiting without execution.")
        return

    # 6. Sequential execution
    runs_file = os.path.join(output_dir, "runs.jsonl")
    commit_hash, is_dirty = get_git_commit_info()

    # If resume, validate git commit hash matches
    if resume and os.path.exists(runs_file):
        existing_commits = set()
        try:
            with open(runs_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        c = d.get("git_commit")
                        if c and c != "unknown":
                            clean_c = c.replace("-dirty", "")
                            existing_commits.add(clean_c)
                    except Exception:
                        pass
        except Exception:
            pass

        clean_current = commit_hash.replace("-dirty", "")
        if existing_commits and clean_current != "unknown" and clean_current not in existing_commits:
            msg = (
                f"Git commit mismatch in runs file '{runs_file}': "
                f"existing runs from commit(s) {sorted(list(existing_commits))}, "
                f"current HEAD is '{clean_current}'. Use --force-resume to proceed."
            )
            if not force_resume:
                print(f"[ERROR] {msg}", file=sys.stderr)
                raise ValueError(msg)
            else:
                print(f"[WARNING] {msg}")

    if record_metadata and not os.path.exists(runs_file):
        try:
            rel_config_path = os.path.relpath(config_path, start=str(REPO_ROOT))
        except Exception:
            rel_config_path = str(config_path)
        meta_header = {
            "_type": "metadata",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": commit_hash,
            "git_dirty": is_dirty,
            "config_path": rel_config_path,
            "timeout": effective_timeout,
        }
        with open(runs_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta_header) + "\n")

    existing_runs = load_existing_runs(runs_file, retry_failed=retry_failed) if resume else {}

    # Build execution queue: (task, rep, rep_groups)
    execution_queue = []
    max_reps = max((t["repetitions"] for t in plan), default=1)
    for rep in range(1, max_reps + 1):
        for task in plan:
            if rep <= task["repetitions"]:
                rep_groups = [g for g in task["groups"] if task["group_reps"][g] >= rep]
                execution_queue.append((task, rep, rep_groups))

    total_executions = len(execution_queue)
    exec_counter = 0

    print(f"Running experiments (deduplicated {len(plan)} unique tasks, {total_executions} total runs)...")
    print(f"Results destination: {runs_file}\n")

    # If resume, check if any existing record's groups need updating
    if resume and existing_runs:
        needs_rewrite = False
        for task, rep, rep_groups in execution_queue:
            key = task["canonical_key"]
            if (key, rep) in existing_runs:
                rec = existing_runs[(key, rep)]
                cur_groups = set(rec.get("groups", []))
                if not set(rep_groups).issubset(cur_groups):
                    rec["groups"] = sorted(list(cur_groups.union(rep_groups)))
                    needs_rewrite = True
        if needs_rewrite:
            with open(runs_file, "w", encoding="utf-8") as f:
                for r in existing_runs.values():
                    f.write(json.dumps(r) + "\n")

    for task, rep, rep_groups in execution_queue:
        exec_counter += 1
        key = task["canonical_key"]

        if resume and (key, rep) in existing_runs:
            print(f"[{exec_counter}/{total_executions}] Skipping (already completed): {task['dataset']} | {task['solver']} | rep={rep}")
            continue

        flags_str = format_task_flags(task)
        print(f"[{exec_counter}/{total_executions}] Running {task['dataset']} (N={task['N']}) | {task['solver']} | flags={flags_str} | rep={rep} ...")

        temp_json_path = os.path.join(output_dir, f"_temp_{os.getpid()}_{exec_counter}_{rep}.json")
        if os.path.exists(temp_json_path):
            try:
                os.remove(temp_json_path)
            except Exception:
                pass

        cmd = build_task_cmd(task, temp_json_path)
        task_timeout = task["timeout"]

        t0 = time.perf_counter()
        timed_out = False
        returncode = 0

        try:
            # 10s grace period for child process to flush its own timeout handling
            popen_kwargs = {
                "cwd": str(REPO_ROOT),
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
            }
            if hasattr(os, "setsid"):
                popen_kwargs["start_new_session"] = True

            proc = subprocess.Popen(cmd, **popen_kwargs)
            try:
                stdout, stderr = proc.communicate(timeout=task_timeout + 10)
                wall_time = round(time.perf_counter() - t0, 4)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                wall_time = round(time.perf_counter() - t0, 4)
                timed_out = True
                returncode = -1
                if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                else:
                    proc.kill()
                try:
                    proc.communicate(timeout=2)
                except Exception:
                    pass
            except KeyboardInterrupt:
                wall_time = round(time.perf_counter() - t0, 4)
                timed_out = False
                returncode = -signal.SIGINT
                if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                else:
                    proc.kill()
                try:
                    proc.communicate(timeout=2)
                except Exception:
                    pass
        except Exception:
            wall_time = round(time.perf_counter() - t0, 4)
            timed_out = False
            returncode = -1

        # Read child output json if created
        record = None
        if os.path.exists(temp_json_path) and os.path.getsize(temp_json_path) > 0:
            try:
                with open(temp_json_path, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]
                    if lines:
                        record = json.loads(lines[-1])
            except Exception:
                record = None
            try:
                os.remove(temp_json_path)
            except Exception:
                pass

        # Fallback if child crashed, timed out, or was interrupted before writing json
        if record is None:
            if timed_out:
                status = "TIMEOUT"
                diagnostic = f"Process timed out after {task_timeout}s without writing JSON output"
            elif returncode in (-signal.SIGINT, -signal.SIGTERM, 130, 143):
                status = "INTERRUPTED"
                diagnostic = f"Process interrupted by signal (returncode {returncode}) without writing JSON output"
            else:
                status = "ERROR"
                diagnostic = f"Child process exited with code {returncode} without writing JSON output"
            record = {
                "dataset": task["dataset"],
                "N": task["N"],
                "num_edges": task["M"],
                "relabel_seed": task.get("relabel_seed"),
                "config": task["config_dict"],
                "solver": task["solver"],
                "status": status,
                "best_ub": -1,
                "encode_time": 0.0,
                "solve_time": wall_time,
                "per_iter_times": [],
                "num_iterations": 0,
                "num_vars": 0,
                "num_clauses": 0,
                "diagnostic": diagnostic,
            }
        elif record.get("best_ub") is None:
            record["best_ub"] = -1

        # Attach parent-measured wall_time, rep, groups, git_commit, git_dirty, relabel_seed
        record["groups"] = rep_groups
        record["rep"] = rep
        record["wall_time"] = wall_time
        record["git_commit"] = commit_hash
        record["git_dirty"] = is_dirty
        if "relabel_seed" not in record or record["relabel_seed"] is None:
            if task.get("relabel_seed") is not None:
                record["relabel_seed"] = task["relabel_seed"]

        # Append record
        with open(runs_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if resume:
            existing_runs[(key, rep)] = record

        status_str = normalize_status(record.get("status")) or str(record.get("status", "UNKNOWN"))
        ub_val = validate_best_ub(record.get("best_ub"))
        ub_str = f"ub={ub_val}" if ub_val is not None else (f"ub={record.get('best_ub')}" if record.get('best_ub') is not None else "")
        st_val = record.get("solve_time")
        if isinstance(st_val, (int, float)) and not isinstance(st_val, bool) and math.isfinite(st_val):
            st_disp = f"{float(st_val):.3f}s"
        else:
            st_disp = "-"
        wt_disp = f"{float(wall_time):.3f}s" if (isinstance(wall_time, (int, float)) and not isinstance(wall_time, bool) and math.isfinite(wall_time)) else "-"
        print(f"       -> {status_str} {ub_str} (solve_time: {st_disp}, wall_time: {wt_disp})")

    print(f"\nExperiment execution finished. All results saved in: {runs_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBMP Experiment Runner")
    parser.add_argument("--config", type=str, default="experiments/configs.yaml", help="path to config YAML")
    parser.add_argument(
        "--group",
        type=str,
        default="all",
        choices=[
            "all",
            "per_dataset",
            "solvers_small",
            "solvers_medium",
            "methods",
            "ablation_small",
            "ablation_medium",
            "ablation_family",
            "methods_family",
            "cpsat_large",
            "methods_large_seeds",
            "encodings",
            "exact_solvers",
            "cplex_cp",
            "gurobi",
        ],
        help="experiment group to run",
    )
    parser.add_argument("--timeout", type=float, default=None, help="override timeout in seconds")
    parser.add_argument("--repetitions", type=int, default=None, help="override number of repetitions per configuration")
    parser.add_argument("--min-n", type=int, default=None, help="override minimum N vertices filter")
    parser.add_argument("--max-n", type=int, default=None, help="override maximum N vertices filter")
    parser.add_argument("--datasets", type=str, default=None, help="comma-separated dataset names to filter")
    parser.add_argument("--resume", action="store_true", help="skip combinations already recorded in runs.jsonl")
    parser.add_argument("--force-resume", action="store_true", help="force resume even if git commit hash differs from recorded runs")
    parser.add_argument("--retry-failed", action="store_true", help="re-run tasks that previously resulted in TIMEOUT or ERROR")
    parser.add_argument("--allow-dirty", action="store_true", help="allow running experiments even if working directory has uncommitted changes")
    parser.add_argument("--dry-run", action="store_true", help="list planned runs without executing")
    parser.add_argument("--output-dir", type=str, default="results_v2", help="directory to store results")
    parser.add_argument("--record-metadata", action="store_true", default=False, help="record timeout, git commit hash, and config path into env.json and runs.jsonl header")
    args = parser.parse_args()

    run_experiments(
        config_path=args.config,
        group=args.group,
        timeout=args.timeout,
        repetitions=args.repetitions,
        min_n=args.min_n,
        max_n=args.max_n,
        dataset_filter=args.datasets,
        resume=args.resume,
        force_resume=args.force_resume,
        retry_failed=args.retry_failed,
        allow_dirty=args.allow_dirty,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        record_metadata=args.record_metadata,
    )
