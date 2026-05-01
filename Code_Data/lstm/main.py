# An LLM was used to help write this file. It runs all the experiments in parallel
# across multiple worker processes so the full sweep finishes faster, and prints
# progress as each job finishes.

import sys
import time
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

from jobs import run_one, tag_for
from plotting import run_plots


SEEDS = [0, 1, 2, 3, 4]
EVAL_EPISODES = 50
MAX_WORKERS = 14

ENV_SPECS = [
    ("InvertedPendulum-v4", False),
    ("InvertedPendulum-v4", True),
    ("HalfCheetah-v4",      False),
    ("HalfCheetah-v4",      True),
    ("Swimmer-v4",          False),
]


def budget_for(algo, env_name):
    """Return (total_iterations, eval_every) for a given algo and env."""
    if algo == "ARS":
        return (20_000, 50) if "HalfCheetah" in env_name else (25_000, 50)
    elif algo == "PPO":
        return (488, 5)
    raise ValueError(algo)


def build_configs(algos, hidden_dim=12, project_dim=None, tag_suffix=""):
    """Build one config per (env, seed, algo) combination."""
    configs = []
    for (env_name, partial), seed in product(ENV_SPECS, SEEDS):
        for algo in algos:
            total_iter, eval_every = budget_for(algo, env_name)
            cfg = {
                "algo": algo, "env_name": env_name, "partial": partial,
                "hidden_dim": hidden_dim,
                "total_iterations": total_iter, "seed": seed,
                "eval_every": eval_every, "eval_episodes": EVAL_EPISODES,
                "out_dir": "runs",
                "tag_suffix": tag_suffix,
            }
            if project_dim is not None:
                cfg["project_dim"] = project_dim
            configs.append(cfg)
    return configs


def run_sweep(configs, label):
    """Run all configs in parallel and print progress as each job finishes."""
    t0 = time.time()
    print(f"\n{'='*80}\nStarting {label} sweep: {len(configs)} jobs on {MAX_WORKERS} workers\n{'='*80}", flush=True)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_one, cfg): cfg for cfg in configs}
        completed = 0
        for fut in as_completed(futures):
            cfg = futures[fut]
            res = fut.result()
            completed += 1
            elapsed = time.time() - t0
            if res["status"] == "error":
                print(f"[{label}] [{completed:2d}/{len(configs)}] [{elapsed/60:6.1f}m] "
                      f"ERROR {tag_for(cfg)}: {res['error']}", flush=True)
            else:
                s = res["summary"]
                marker = "cached" if res["status"] == "cached" else "done"
                print(f"[{label}] [{completed:2d}/{len(configs)}] [{elapsed/60:6.1f}m] "
                      f"{marker:6s} {tag_for(cfg):55s} "
                      f"final={s['final_eval_mean']:8.1f} ± {s['final_eval_std']:6.1f} "
                      f"({s['elapsed_seconds']/60:.1f}m)", flush=True)

    print(f"\n{label} sweep done in {(time.time()-t0)/60:.1f} min", flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t_global = time.time()

    if which == "ars":
        run_sweep(build_configs(["ARS"], hidden_dim=12), "ARS")
        run_plots()

    elif which == "ppo":
        run_sweep(build_configs(["PPO"], hidden_dim=12), "PPO")
        run_plots()

    elif which == "all":
        run_sweep(build_configs(["ARS"], hidden_dim=12), "ARS")
        run_sweep(build_configs(["PPO"], hidden_dim=12), "PPO")
        run_plots()

    elif which == "ars_proj":
        run_sweep(
            build_configs(["ARS"], hidden_dim=12, project_dim=2, tag_suffix="_proj2"),
            "ARS_proj2")
        run_plots(plots_dir="plots_proj2", suffix="_proj2",
                  algos=["ARS"], label_suffix=" (proj2)")

    elif which == "ppo_small_proj":
        run_sweep(
            build_configs(["PPO"], hidden_dim=12, project_dim=2, tag_suffix="_proj2"),
            "PPO_proj2")
        run_plots(plots_dir="plots_proj2", suffix="_proj2",
                  algos=["PPO"], label_suffix=" (proj2)")

    elif which == "ppo_big":
        run_sweep(
            build_configs(["PPO"], hidden_dim=64, tag_suffix="_h64"),
            "PPO_h64")
        run_plots(plots_dir="plots_h64", suffix="_h64",
                  algos=["PPO"], label_suffix=" (h64)")

    elif which == "extras":
        run_sweep(
            build_configs(["PPO"], hidden_dim=12, project_dim=2, tag_suffix="_proj2"),
            "PPO_proj2")
        run_sweep(
            build_configs(["PPO"], hidden_dim=64, tag_suffix="_h64"),
            "PPO_h64")
        run_sweep(
            build_configs(["ARS"], hidden_dim=12, project_dim=2, tag_suffix="_proj2"),
            "ARS_proj2")
        run_plots(plots_dir="plots_proj2", suffix="_proj2", label_suffix=" (proj2)")
        run_plots(plots_dir="plots_h64", suffix="_h64",
                  algos=["PPO"], label_suffix=" (h64)")

    else:
        raise ValueError(f"Unknown mode: {which}. "
                         f"Valid: ars, ppo, all, ars_proj, ppo_small_proj, ppo_big, extras")

    print(f"\n{'='*80}\nTotal wall-clock: {(time.time()-t_global)/60:.1f} min\n{'='*80}", flush=True)


if __name__ == "__main__":
    main()