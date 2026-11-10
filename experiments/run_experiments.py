import os
import sys
import glob
import json
import time
import yaml
import argparse
import platform
import subprocess
from pathlib import Path

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

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

def collect_environment_info(output_path="results/env.json", timeout=None, config_path=None, record_metadata=False):
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
            info["config_path"] = os.path.abspath(config_path)

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
    Extracts canonical (dataset, method, solver, flags) tuple from a result dictionary.
    """
    dataset = record.get("dataset")
    solver = str(record.get("solver", "")).lower()
    cfg = record.get("config", {})
    if solver in ("cp_sat", "cpsat"):
        method = "cpsat"
        flags = (cfg.get("num_workers", 1),)
    elif solver == "cplex":
        method = "cplex"
        flags = ()
    else:
        method = "sat"
        rot = bool(cfg.get("sb_rotation", False))
        ref = bool(cfg.get("sb_reflection", False))
        heur = bool(cfg.get("heuristic_ub", False))
        flags = (rot, ref, heur)
    return (dataset, method, solver, flags)

def load_existing_runs(runs_jsonl_path, retry_failed=False):
    """
    Load runs already recorded in runs.jsonl mapping (canonical_key, rep) -> record.
    If retry_failed is True, only OPTIMAL and SIZE_LIMIT runs are retained in completed (so TIMEOUT/ERROR are retried).
    """
    if not os.path.exists(runs_jsonl_path):
        return {}
    completed = {}
    try:
        with open(runs_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("_type") == "metadata" or "dataset" not in data:
                        continue
                    key = get_canonical_key(data)
                    rep = int(data.get("rep", 1))
                    status = data.get("status")
                    if retry_failed:
                        if status in ("OPTIMAL", "SIZE_LIMIT"):
                            completed[(key, rep)] = data
                    else:
                        completed[(key, rep)] = data
                except Exception:
                    pass
    except Exception:
        pass
    return completed

def filter_datasets_for_group(all_datasets, group_cfg, defaults, cli_datasets=None, cli_min_n=None, cli_max_n=None):
    """
    Filters datasets for a specific group based on:
    - CLI overrides if provided (cli_datasets, cli_min_n, cli_max_n)
    - Group config (datasets, min_n/min_N, max_n/max_N)
    - Fallback to defaults.filters
    """
    default_filters = defaults.get("filters", {})

    # 1. Dataset name filter
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
            allowed = set(grp_ds)
        elif isinstance(grp_ds, str):
            allowed = set([s.strip() for s in grp_ds.split(",")])
        else:
            allowed = None

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
        filtered.append(d)

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
                key = (d["name"], method, solver.lower(), flags)
                if key not in tasks_map:
                    tasks_map[key] = {
                        "canonical_key": key,
                        "dataset": d["name"],
                        "N": d["N"],
                        "M": d["M"],
                        "method": method,
                        "solver": solver,
                        "flags": flags,
                        "config_dict": {
                            "sb_rotation": rot,
                            "sb_reflection": ref,
                            "heuristic_ub": heur,
                            "time_limit": grp_timeout,
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

        elif grp == "solvers":
            rot = bool(grp_cfg.get("sb_rotation", True))
            ref = bool(grp_cfg.get("sb_reflection", True))
            heur = bool(grp_cfg.get("heuristic_ub", True))
            solvers = grp_cfg.get("solvers", ["cadical195"])
            method = "sat"
            flags = (rot, ref, heur)

            for d in grp_datasets:
                for s in solvers:
                    grp_tasks_count += 1
                    key = (d["name"], method, s.lower(), flags)
                    if key not in tasks_map:
                        tasks_map[key] = {
                            "canonical_key": key,
                            "dataset": d["name"],
                            "N": d["N"],
                            "M": d["M"],
                            "method": method,
                            "solver": s,
                            "flags": flags,
                            "config_dict": {
                                "sb_rotation": rot,
                                "sb_reflection": ref,
                                "heuristic_ub": heur,
                                "time_limit": grp_timeout,
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

        elif grp == "methods":
            methods = grp_cfg.get("methods", [])
            for d in grp_datasets:
                for m in methods:
                    grp_tasks_count += 1
                    m_type = m.get("type", "sat")
                    if m_type == "sat":
                        solver = m.get("solver", "cadical195")
                        rot = bool(m.get("sb_rotation", True))
                        ref = bool(m.get("sb_reflection", True))
                        heur = bool(m.get("heuristic_ub", True))
                        method = "sat"
                        flags = (rot, ref, heur)
                        key = (d["name"], method, solver.lower(), flags)
                        cfg_dict = {
                            "sb_rotation": rot,
                            "sb_reflection": ref,
                            "heuristic_ub": heur,
                            "time_limit": grp_timeout,
                        }
                    elif m_type == "cpsat":
                        num_w = m.get("num_workers", 1)
                        method = "cpsat"
                        solver = "cp_sat"
                        flags = (num_w,)
                        key = (d["name"], method, solver, flags)
                        cfg_dict = {
                            "num_workers": num_w,
                            "time_limit": grp_timeout,
                        }
                    elif m_type == "cplex":
                        method = "cplex"
                        solver = "cplex"
                        flags = ()
                        key = (d["name"], method, solver, flags)
                        cfg_dict = {
                            "time_limit": grp_timeout,
                        }
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

        elif grp == "ablation":
            solver = grp_cfg.get("solver", "cadical195")
            variants = grp_cfg.get("variants", [])
            method = "sat"
            for d in grp_datasets:
                for var in variants:
                    grp_tasks_count += 1
                    rot = bool(var.get("sb_rotation", False))
                    ref = bool(var.get("sb_reflection", False))
                    heur = bool(var.get("heuristic_ub", False))
                    flags = (rot, ref, heur)
                    key = (d["name"], method, solver.lower(), flags)
                    cfg_dict = {
                        "sb_rotation": rot,
                        "sb_reflection": ref,
                        "heuristic_ub": heur,
                        "time_limit": grp_timeout,
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
    if method == "sat":
        rot, ref, heur = task["flags"]
        cmd = [
            sys.executable, "main.py", dataset,
            "--timeout", str(timeout),
            "--solver", task["solver"],
            "--json-out", json_out_path,
        ]
        if rot: cmd.append("--sb-rotation")
        if ref: cmd.append("--sb-reflection")
        if heur: cmd.append("--heuristic-ub")
        return cmd
    elif method == "cpsat":
        num_w = task["flags"][0] if task["flags"] else 1
        return [
            sys.executable, "-m", "baselines.cpsat_cbmp", dataset,
            "--timeout", str(timeout),
            "--num-workers", str(num_w),
            "--json-out", json_out_path,
        ]
    elif method == "cplex":
        return [
            sys.executable, "-m", "baselines.cplex_cbmp", dataset,
            "--timeout", str(timeout),
            "--json-out", json_out_path,
        ]
    else:
        raise ValueError(f"Unknown method: {method}")

def format_task_flags(task):
    """Human-readable representation of configuration flags."""
    if task["method"] == "sat":
        rot, ref, heur = task["flags"]
        parts = []
        if rot: parts.append("rot")
        if ref: parts.append("ref")
        if heur: parts.append("heur_ub")
        return "+".join(parts) if parts else "none"
    elif task["method"] == "cpsat":
        return f"workers={task['flags'][0]}"
    elif task["method"] == "cplex":
        return "default"
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
    dry_run=False,
    output_dir="results",
    record_metadata=False,
):
    from scripts.check_datasets import validate_datasets

    if retry_failed:
        resume = True

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

    # 3. Environment info
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
        meta_header = {
            "_type": "metadata",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": commit_hash,
            "git_dirty": is_dirty,
            "config_path": os.path.abspath(config_path),
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
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=task_timeout + 10,
            )
            wall_time = round(time.perf_counter() - t0, 4)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            wall_time = round(time.perf_counter() - t0, 4)
            timed_out = True
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

        # Fallback if child crashed or timed out before writing json
        if record is None:
            status = "TIMEOUT" if timed_out else "ERROR"
            record = {
                "dataset": task["dataset"],
                "N": task["N"],
                "num_edges": task["M"],
                "config": task["config_dict"],
                "solver": task["solver"],
                "status": status,
                "best_ub": None,
                "encode_time": 0.0,
                "solve_time": wall_time,
                "per_iter_times": [],
                "num_iterations": 0,
                "num_vars": 0,
                "num_clauses": 0,
            }

        # Attach parent-measured wall_time, rep, groups, git_commit, git_dirty
        record["groups"] = rep_groups
        record["rep"] = rep
        record["wall_time"] = wall_time
        record["git_commit"] = commit_hash
        record["git_dirty"] = is_dirty

        # Append record
        with open(runs_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if resume:
            existing_runs[(key, rep)] = record

        status_str = record.get("status", "UNKNOWN")
        ub_str = f"ub={record.get('best_ub')}" if record.get("best_ub") is not None else ""
        print(f"       -> {status_str} {ub_str} (solve_time: {record.get('solve_time', 0):.3f}s, wall_time: {wall_time:.3f}s)")

    print(f"\nExperiment execution finished. All results saved in: {runs_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBMP Experiment Runner")
    parser.add_argument("--config", type=str, default="experiments/configs.yaml", help="path to config YAML")
    parser.add_argument("--group", type=str, default="all", choices=["all", "per_dataset", "solvers", "methods", "ablation"], help="experiment group to run")
    parser.add_argument("--timeout", type=float, default=None, help="override timeout in seconds")
    parser.add_argument("--repetitions", type=int, default=None, help="override number of repetitions per configuration")
    parser.add_argument("--min-n", type=int, default=None, help="override minimum N vertices filter")
    parser.add_argument("--max-n", type=int, default=None, help="override maximum N vertices filter")
    parser.add_argument("--datasets", type=str, default=None, help="comma-separated dataset names to filter")
    parser.add_argument("--resume", action="store_true", help="skip combinations already recorded in runs.jsonl")
    parser.add_argument("--force-resume", action="store_true", help="force resume even if git commit hash differs from recorded runs")
    parser.add_argument("--retry-failed", action="store_true", help="re-run tasks that previously resulted in TIMEOUT or ERROR")
    parser.add_argument("--dry-run", action="store_true", help="list planned runs without executing")
    parser.add_argument("--output-dir", type=str, default="results", help="directory to store results")
    parser.add_argument("--record-metadata", action="store_true", default=False, help="record timeout, git commit hash, and config path into results/env.json and runs.jsonl header")
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
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        record_metadata=args.record_metadata,
    )
