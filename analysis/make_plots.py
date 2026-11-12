import os
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.make_tables import (
    load_records,
    cross_validate_optima,
    validate_solve_time,
    extract_repetition_solve_time,
    is_optimal_status,
    normalize_status,
    validate_best_ub,
)

def setup_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 14
    plt.rcParams["figure.autolayout"] = True

def save_fig(fig, base_filename, output_dirs=("results_v2/figures",)):
    if isinstance(output_dirs, str):
        output_dirs = (output_dirs,)
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        png_path = os.path.join(out_dir, f"{base_filename}.png")
        pdf_path = os.path.join(out_dir, f"{base_filename}.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Figure saved: {base_filename}.png and {base_filename}.pdf in {output_dirs[0]}/")
    plt.close(fig)

def get_ablation_var_name(cfg):
    rot = cfg.get("sb_rotation", False)
    ref = cfg.get("sb_reflection", False)
    heur = cfg.get("heuristic_ub", False)
    if not rot and not ref and not heur:
        return "none"
    elif rot and not ref and not heur:
        return "rotation"
    elif rot and ref and not heur:
        return "rot+ref"
    elif not rot and not ref and heur:
        return "heuristic-ub"
    elif rot and ref and heur:
        return "all (SB+Heur)"
    return "other"

def plot_cactus(records, group_name, category_key, title, base_filename, output_dirs=("results_v2/figures",)):
    """
    Plots cumulative solved instances over solve time (Cactus plot).
    """
    recs = [
        r for r in records
        if (group_name in r.get("groups", []) or (group_name == "solvers_small" and "solvers" in r.get("groups", [])) or (group_name == "ablation_small" and "ablation" in r.get("groups", [])))
        and is_optimal_status(r)
    ]
    if not recs:
        return

    grouped = {}
    for r in recs:
        cat = category_key(r)
        t = extract_repetition_solve_time(r)
        if t is not None and t > 0:
            grouped.setdefault(cat, []).append(t)

    # If no positive times found for any category, return
    if not any(grouped.values()):
        return

    fig, ax = plt.subplots(figsize=(7, 4.8))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(grouped), 10)))

    for idx, (cat, times) in enumerate(sorted(grouped.items())):
        if not times:
            continue
        sorted_times = sorted(times)
        # Avoid 0.0 on log-scaled axis:
        # Step curve starts at first instance time with y=0 before jumping to 1
        x_vals = [sorted_times[0]] + sorted_times
        y_vals = [0] + list(range(1, len(sorted_times) + 1))
        ax.step(x_vals, y_vals, where="post", label=cat, color=colors[idx], linewidth=2.0)
        ax.scatter(sorted_times, range(1, len(sorted_times) + 1), color=colors[idx], s=30, zorder=5)

    ax.set_title(title, pad=10, fontweight="bold")
    ax.set_xlabel("Solve Time (seconds, log scale)")
    ax.set_ylabel("Number of Solved Instances")
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="upper left")

    save_fig(fig, base_filename, output_dirs=output_dirs)

def plot_scatter_sb_vs_nosb(records, group_name="ablation_small", base_filename="scatter_sb_vs_nosb_small", title="Solve Time: No Symmetry Breaking vs With Symmetry Breaking (Small Graphs)", output_dirs=("results_v2/figures",)):
    """
    Log-log scatter plot comparing solve time with vs without symmetry breaking.
    """
    ablation_recs = [
        r for r in records
        if (group_name in r.get("groups", []) or (group_name == "ablation_small" and "ablation" in r.get("groups", [])))
        and is_optimal_status(r)
    ]
    if not ablation_recs:
        return

    by_ds = {}
    for r in ablation_recs:
        ds = r.get("dataset")
        if not ds:
            continue
        cfg = r.get("config", {})
        rot = cfg.get("sb_rotation", False)
        ref = cfg.get("sb_reflection", False)
        heur = cfg.get("heuristic_ub", False)

        t = extract_repetition_solve_time(r)
        if t is None or t <= 0:
            continue

        if not rot and not ref and not heur:
            by_ds.setdefault(ds, {})["no_sb"] = t
        elif rot and ref and heur:
            by_ds.setdefault(ds, {})["with_sb"] = t

    pairs = []
    labels = []
    for ds, vals in by_ds.items():
        if "no_sb" in vals and "with_sb" in vals:
            pairs.append((vals["no_sb"], vals["with_sb"]))
            labels.append(ds)

    if not pairs:
        return

    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(x_vals, y_vals, color="#1f77b4", s=70, edgecolors="black", linewidth=1.2, zorder=5)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (x_vals[i], y_vals[i]), textcoords="offset points", xytext=(8, -4), fontsize=9)

    min_val = max(min(min(x_vals), min(y_vals)) * 0.5, 1e-6)
    max_val = max(max(x_vals), max(y_vals)) * 2.0

    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5, label="y = x (equal performance)")
    ax.fill_between([min_val, max_val], [min_val, max_val], [min_val * 0.01, min_val * 0.01], color="#2ca02c", alpha=0.1, label="Speedup with Symmetry Breaking")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])

    ax.set_title(title, pad=10, fontweight="bold")
    ax.set_xlabel("Solve Time without Symmetry Breaking (s) [log scale]")
    ax.set_ylabel("Solve Time with Symmetry Breaking (s) [log scale]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="upper left")

    save_fig(fig, base_filename, output_dirs=output_dirs)

def plot_ablation_median_time(records, group_name="ablation_small", base_filename="ablation_median_time_small", title="Ablation Study: Median Solve Time Across Configurations (Small Graphs)", output_dirs=("results_v2/figures",)):
    """
    Bar chart showing median solve time per configuration.
    """
    ablation_recs = [
        r for r in records
        if (group_name in r.get("groups", []) or (group_name == "ablation_small" and "ablation" in r.get("groups", [])))
        and is_optimal_status(r)
    ]
    if not ablation_recs:
        return

    order = ["none", "rotation", "rot+ref", "heuristic-ub", "all (SB+Heur)"]
    times_by_var = {v: [] for v in order}

    for r in ablation_recs:
        var = get_ablation_var_name(r.get("config", {}))
        if var in times_by_var:
            t = extract_repetition_solve_time(r)
            if t is not None and t > 0:
                times_by_var[var].append(t)

    medians = [float(np.median(times_by_var[v])) if times_by_var[v] else 0.0 for v in order]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = ["#7f7f7f", "#aec7e8", "#1f77b4", "#ffbb78", "#2ca02c"]
    bars = ax.bar(order, medians, color=colors, edgecolor="black", width=0.55)

    max_med = max(medians) if any(medians) else 1.0
    for bar, v_name, val in zip(bars, order, medians):
        if times_by_var[v_name]:
            ax.text(bar.get_x() + bar.get_width() / 2, val + (max_med * 0.02), f"{val:.4f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, max_med * 0.02, "-", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title(title, pad=10, fontweight="bold")
    ax.set_ylabel("Median Solve Time (seconds)")
    ax.set_ylim([0, max_med * 1.15])
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    save_fig(fig, base_filename, output_dirs=output_dirs)

def plot_ablation_iterations_and_unsat(records, group_name="ablation_small", base_filename="ablation_iterations_and_unsat_small", title="Impact of Symmetry Breaking on SAT Iterations and UNSAT Proof Time (Small Graphs)", output_dirs=("results_v2/figures",)):
    """
    Plots number of SAT iterations and time spent on the final UNSAT iteration.
    """
    ablation_recs = [
        r for r in records
        if (group_name in r.get("groups", []) or (group_name == "ablation_small" and "ablation" in r.get("groups", [])))
        and is_optimal_status(r)
    ]
    if not ablation_recs:
        return

    order = ["none", "rotation", "rot+ref", "heuristic-ub", "all (SB+Heur)"]
    iter_counts = {v: [] for v in order}
    unsat_times = {v: [] for v in order}

    for r in ablation_recs:
        var = get_ablation_var_name(r.get("config", {}))
        if var in iter_counts:
            it = r.get("num_iterations")
            if isinstance(it, int) and not isinstance(it, bool) and it >= 0:
                iter_counts[var].append(it)
            per_iter = r.get("per_iter_times", [])
            if isinstance(per_iter, list) and per_iter:
                t_unsat = validate_solve_time(per_iter[-1])
                if t_unsat is not None and t_unsat > 0:
                    unsat_times[var].append(t_unsat)

    avg_iters = [float(np.mean(iter_counts[v])) if iter_counts[v] else 0.0 for v in order]
    avg_unsat = [float(np.mean(unsat_times[v])) if unsat_times[v] else 0.0 for v in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    b1 = ax1.bar(order, avg_iters, color="#1f77b4", edgecolor="black", width=0.5)
    max_iters = max(avg_iters) if any(avg_iters) else 1.0
    for bar, v_name, val in zip(b1, order, avg_iters):
        if iter_counts[v_name]:
            ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        else:
            ax1.text(bar.get_x() + bar.get_width() / 2, 0.2, "-", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_title("Average Number of SAT Iterations", pad=8, fontweight="bold")
    ax1.set_ylabel("Iterations")
    ax1.set_ylim([0, max_iters * 1.2])
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.tick_params(axis="x", rotation=15)

    b2 = ax2.bar(order, avg_unsat, color="#2ca02c", edgecolor="black", width=0.5)
    max_unsat = max(avg_unsat) if any(avg_unsat) else 1.0
    for bar, v_name, val in zip(b2, order, avg_unsat):
        if unsat_times[v_name]:
            ax2.text(bar.get_x() + bar.get_width() / 2, val + (max_unsat * 0.02), f"{val:.4f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
        else:
            ax2.text(bar.get_x() + bar.get_width() / 2, max_unsat * 0.02, "-", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_title("Mean Time of Final (UNSAT) Iteration", pad=8, fontweight="bold")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_ylim([0, max_unsat * 1.2])
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.tick_params(axis="x", rotation=15)

    fig.suptitle(title, fontsize=13, fontweight="bold")

    save_fig(fig, base_filename, output_dirs=output_dirs)

def plot_ablation_family_boxplots(records, base_filename="boxplot_ablation_family", output_dirs=("results_v2/figures",)):
    """
    Boxplot (log scale) of time ratio relative to 'none' across vertex relabeling families.
    Separate panels for N=40 and N=50.
    Variants compared to none: rotation, rot+ref, heur_ub, all.
    """
    family_recs = [r for r in records if r.get("relabel_seed") is not None]
    if not family_recs:
        return

    def get_var(rec):
        cfg = rec.get("config", {})
        rot = cfg.get("sb_rotation", False)
        ref = cfg.get("sb_reflection", False)
        heur = cfg.get("heuristic_ub", False)
        if not rot and not ref and not heur:
            return "none"
        elif rot and not ref and not heur:
            return "rotation"
        elif rot and ref and not heur:
            return "rot_ref"
        elif not rot and not ref and heur:
            return "heur_ub"
        elif rot and ref and heur:
            return "all"
        return "other"

    pairs_map = {}
    for r in family_recs:
        ds = r.get("dataset")
        seed = r.get("relabel_seed")
        var = get_var(r)
        pairs_map.setdefault((ds, seed), {})[var] = r

    variants = ["rotation", "rot_ref", "heur_ub", "all"]
    labels = ["Rotation", "Rot + Ref", "Heur UB", "All (SB+Heur)"]

    data_n40 = [[] for _ in variants]
    data_n50 = [[] for _ in variants]

    for (ds, seed), vmap in pairs_map.items():
        if "none" not in vmap or not is_optimal_status(vmap["none"]):
            continue
        t_none = validate_solve_time(vmap["none"].get("solve_time"))
        if t_none is None or t_none <= 0:
            continue
        n_val = vmap["none"].get("N", 0)

        for i, var in enumerate(variants):
            if var in vmap and is_optimal_status(vmap[var]):
                t_var = validate_solve_time(vmap[var].get("solve_time"))
                if t_var is not None and t_var > 0:
                    ratio = t_var / t_none
                    if math.isfinite(ratio) and ratio > 0:
                        if n_val == 40:
                            data_n40[i].append(ratio)
                        elif n_val == 50:
                            data_n50[i].append(ratio)

    if not (all(len(d) >= 1 for d in data_n40) and all(len(d) >= 1 for d in data_n50)):
        print("[WARNING] Skipping boxplot_ablation_family: not all variants have >= 1 valid ratio relative to 'none'.")
        return

    setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    if any(any(d) for d in data_n40):
        bp1 = ax1.boxplot(data_n40, tick_labels=labels, patch_artist=True)
        for box in bp1["boxes"]:
            box.set(facecolor="#4C72B0", alpha=0.7)
    ax1.set_yscale("log")
    ax1.axhline(1.0, color="red", linestyle="--", alpha=0.8, label="none (baseline = 1.0)")
    ax1.set_title("N = 40 (494_bus & 662_bus, 10 seeds each)")
    ax1.set_ylabel("Time Ratio relative to 'none' (log scale)")
    ax1.grid(True, linestyle=":", alpha=0.6, which="both")
    ax1.legend(loc="upper right")

    if any(any(d) for d in data_n50):
        bp2 = ax2.boxplot(data_n50, tick_labels=labels, patch_artist=True)
        for box in bp2["boxes"]:
            box.set(facecolor="#55A868", alpha=0.7)
    ax2.set_yscale("log")
    ax2.axhline(1.0, color="red", linestyle="--", alpha=0.8, label="none (baseline = 1.0)")
    ax2.set_title("N = 50 (494_bus & 662_bus, 5 seeds each)")
    ax2.grid(True, linestyle=":", alpha=0.6, which="both")
    ax2.legend(loc="upper right")

    fig.suptitle("Ablation Study on Random Relabeling Families: Time Ratio vs 'none'", y=1.02)
    save_fig(fig, base_filename, output_dirs=output_dirs)

def main():
    print("=" * 70)
    print("ANALYSIS: GENERATING FIGURES (PNG AND PDF)")
    print("=" * 70)

    import argparse
    parser = argparse.ArgumentParser(description="Generate performance figures from experiment records")
    parser.add_argument("input_file", nargs="?", default=None, help="path to runs.jsonl or results directory")
    parser.add_argument("output_dir_pos", nargs="?", default=None, help="optional positional output directory")
    parser.add_argument("--output-dir", type=str, default=None, help="directory to store figures (defaults to <run_dir>/figures)")
    args = parser.parse_args()

    setup_matplotlib()

    input_file = args.input_file
    output_dir = args.output_dir or args.output_dir_pos
    if not output_dir:
        if input_file and (os.path.isfile(input_file) or os.path.isdir(input_file)):
            run_dir = os.path.dirname(input_file) if os.path.isfile(input_file) else input_file
            run_dir = run_dir or "results_v2"
        else:
            run_dir = "results_v2"
        output_dir = os.path.join(run_dir, "figures")

    records = load_records(input_file)
    if not records:
        print("No records found in results. Please run experiments first.")
        return

    # Cross-validation
    try:
        run_tables_dir = os.path.join(os.path.dirname(output_dir), "tables")
        cross_validate_optima(records, output_dir=run_tables_dir)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    output_dirs = (output_dir,)

    # 1. Cactus plots for groups
    plot_cactus(
        records,
        group_name="solvers_small",
        category_key=lambda r: r.get("solver", "unknown"),
        title="Cactus Plot: Solvers (Small Graphs)",
        base_filename="cactus_solvers_small",
        output_dirs=output_dirs,
    )
    plot_cactus(
        records,
        group_name="solvers_medium",
        category_key=lambda r: r.get("solver", "unknown"),
        title="Cactus Plot: Solvers (Medium Graphs)",
        base_filename="cactus_solvers_medium",
        output_dirs=output_dirs,
    )

    plot_cactus(
        records,
        group_name="methods",
        category_key=lambda r: "cpsat" if r.get("solver", "").lower() in ("cp_sat", "cpsat") else ("cplex" if r.get("solver", "").lower() == "cplex" else f"sat ({r.get('solver')})"),
        title="Cactus Plot: Methods Comparison",
        base_filename="cactus_methods",
        output_dirs=output_dirs,
    )

    plot_cactus(
        records,
        group_name="ablation_small",
        category_key=lambda r: get_ablation_var_name(r.get("config", {})),
        title="Cactus Plot: Ablation (Small Graphs)",
        base_filename="cactus_ablation_small",
        output_dirs=output_dirs,
    )
    plot_cactus(
        records,
        group_name="ablation_medium",
        category_key=lambda r: get_ablation_var_name(r.get("config", {})),
        title="Cactus Plot: Ablation (Medium Graph)",
        base_filename="cactus_ablation_medium",
        output_dirs=output_dirs,
    )

    # 2. Scatter log-log
    plot_scatter_sb_vs_nosb(records, group_name="ablation_small", base_filename="scatter_sb_vs_nosb_small", title="Solve Time: No Symmetry Breaking vs With Symmetry Breaking (Small Graphs)", output_dirs=output_dirs)

    # 3. Ablation median time bar chart
    plot_ablation_median_time(records, group_name="ablation_small", base_filename="ablation_median_time_small", title="Ablation Study: Median Solve Time Across Configurations (Small Graphs)", output_dirs=output_dirs)
    plot_ablation_median_time(records, group_name="ablation_medium", base_filename="ablation_median_time_medium", title="Ablation Study: Solve Time Across Configurations (662_bus_n60)", output_dirs=output_dirs)

    # 4. Iterations & UNSAT time
    plot_ablation_iterations_and_unsat(records, group_name="ablation_small", base_filename="ablation_iterations_and_unsat_small", title="Impact of Symmetry Breaking on SAT Iterations and UNSAT Proof Time (Small Graphs)", output_dirs=output_dirs)
    plot_ablation_iterations_and_unsat(records, group_name="ablation_medium", base_filename="ablation_iterations_and_unsat_medium", title="Impact of Symmetry Breaking on SAT Iterations and UNSAT Proof Time (662_bus_n60)", output_dirs=output_dirs)

    # 5. Ablation Family boxplots
    plot_ablation_family_boxplots(records, base_filename="boxplot_ablation_family", output_dirs=output_dirs)

    print(f"\nAll figures generated successfully in {output_dir}/!")

if __name__ == "__main__":
    main()
