from cbmp import solve, AVAILABLE_SOLVERS
import argparse
import multiprocessing as mp
import json
import os

def main(
    dataset,
    sb_rotation,
    sb_reflection,
    use_heuristic_initial_ub,
    timeout,
    solver="cadical195",
    json_out=None,
    relabel_seed=None,
    encoding="order",
    solving_mode="incremental",
    eo_strategy="commander",
):
    manager = mp.Manager()
    shared_stats = manager.dict()
    best_ub = mp.Value('i', -1)

    p = mp.Process(
        target=solve,
        args=(dataset, sb_rotation, sb_reflection, use_heuristic_initial_ub, best_ub),
        kwargs={
            "solver": solver,
            "shared_stats": shared_stats,
            "verbose": True,
            "relabel_seed": relabel_seed,
            "encoding": encoding,
            "solving_mode": solving_mode,
            "eo_strategy": eo_strategy,
        },
    )
    
    p.start()
    p.join(timeout)       # parent lets this child run for up to [timeout] seconds

    if p.is_alive():      # child still running after waiting [timeout] seconds
        p.terminate()
        p.join()
        shared_stats['status'] = 'TIMEOUT'
        shared_stats['best_ub'] = best_ub.value
        if best_ub.value == -1:
            print("timeout, no valid assignment was found")
        else:
            print("timeout, lowest bandwidth found:", best_ub.value)
    
    elif p.exitcode == 0: # child exited normally
        child_status = shared_stats.get('status')
        if child_status in ('OPTIMAL', 'FEASIBLE', 'UNSAT', 'TIMEOUT'):
            shared_stats['status'] = child_status
            shared_stats['best_ub'] = best_ub.value
            if child_status == 'OPTIMAL':
                print("minimum bandwidth:", best_ub.value)
            elif child_status == 'FEASIBLE':
                print("feasible bandwidth found:", best_ub.value)
            elif child_status == 'UNSAT':
                print("problem is unsatisfiable")
            elif child_status == 'TIMEOUT':
                print("timeout, lowest bandwidth found:", best_ub.value)
        else:
            shared_stats['status'] = child_status if child_status else 'ERROR'
            shared_stats['best_ub'] = best_ub.value
            print(f"child process finished with status: {shared_stats['status']} - lowest bandwidth found: {best_ub.value}")
    
    else:                 # child crashed
        shared_stats['status'] = 'ERROR'
        shared_stats['best_ub'] = best_ub.value
        print("child process crashed with exit code:", p.exitcode, "- lowest bandwidth found:", best_ub.value)

    final_stats = dict(shared_stats)
    if "config" in final_stats and hasattr(final_stats["config"], "items"):
        final_stats["config"] = dict(final_stats["config"])
    if "per_iter_times" in final_stats:
        final_stats["per_iter_times"] = list(final_stats["per_iter_times"])

    if "relabel_seed" not in final_stats:
        final_stats["relabel_seed"] = relabel_seed

    if json_out:
        out_path = os.path.abspath(json_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(final_stats) + "\n")

    manager.shutdown()
    return final_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBMP SAT Solver Entry Point")
    parser.add_argument("dataset", 
                        help="name of .mtx file in datasets/")
    parser.add_argument("--timeout", 
                        type=float, 
                        default=600, 
                        help="time limit in seconds")
    parser.add_argument("--symmetry-break", 
                        action="store_true", 
                        help="add rotation + reflection symmetry breaking (alias for --sb-rotation --sb-reflection)")
    parser.add_argument("--sb-rotation", 
                        action="store_true", 
                        help="add rotation symmetry breaking (fix f(x1) = 1)")
    parser.add_argument("--sb-reflection", 
                        action="store_true", 
                        help="add reflection symmetry breaking (requires --sb-rotation)")
    parser.add_argument("--heuristic-ub", 
                        action="store_true", 
                        help="use Cuthill-McKee initial upper bound")
    parser.add_argument("--solver",
                        type=str,
                        default="cadical195",
                        help=f"SAT solver to use (available: {', '.join(sorted(AVAILABLE_SOLVERS.keys()))}; default: cadical195)")
    parser.add_argument("--encoding",
                        type=str,
                        choices=["order", "pairwise"],
                        default="order",
                        help="SAT encoding to use (order or pairwise, default: order)")
    parser.add_argument("--solving-mode",
                        type=str,
                        choices=["incremental", "iterative"],
                        default="incremental",
                        help="Solving strategy (incremental or iterative, default: incremental)")
    parser.add_argument("--eo-strategy",
                        type=str,
                        choices=["commander", "pairwise", "binary", "sequential", "product"],
                        default="commander",
                        help="Exactly-one encoding strategy (default: commander)")
    parser.add_argument("--json-out",
                        type=str,
                        default=None,
                        help="path to append statistics dict as a single JSON line")
    parser.add_argument("--relabel-seed",
                        type=int,
                        default=None,
                        help="seed for random vertex relabeling (default: None = original labels)")
    args = parser.parse_args()

    sb_rotation = args.sb_rotation
    sb_reflection = args.sb_reflection
    if args.symmetry_break:
        sb_rotation = True
        sb_reflection = True

    if sb_reflection and not sb_rotation:
        parser.error("--sb-reflection requires --sb-rotation (reflection symmetry breaking assumes rotation symmetry breaking is active)")

    if args.solver.lower() not in AVAILABLE_SOLVERS:
        avail_str = ", ".join(sorted(AVAILABLE_SOLVERS.keys()))
        parser.error(f"Solver '{args.solver}' is not available. Available solvers: {avail_str}")

    main(
        args.dataset,
        sb_rotation,
        sb_reflection,
        args.heuristic_ub,
        args.timeout,
        solver=args.solver.lower(),
        json_out=args.json_out,
        relabel_seed=args.relabel_seed,
        encoding=args.encoding,
        solving_mode=args.solving_mode,
        eo_strategy=args.eo_strategy,
    )
