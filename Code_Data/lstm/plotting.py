"""Combines results into mean/std learning curves and saves PNGs."""
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALGOS = ["ARS", "PPO"]
ENV_SPECS_PLOT = [
    ("InvertedPendulum-v4", False),
    ("InvertedPendulum-v4", True),
    ("HalfCheetah-v4",      False),
    ("HalfCheetah-v4",      True),
    ("Swimmer-v4",          False),
]


def _load_seeds(algo, env_name, partial, suffix="", out_dir="runs"):
    po = "PO" if partial else "Full"
    paths = sorted(glob.glob(os.path.join(out_dir, f"{algo}_{env_name}_{po}{suffix}_seed*.json")))
    return [json.load(open(p)) for p in paths]


def _pretty_env(env_name, partial):
    base = env_name.replace("-v4", "")
    return f"{base} {'PO' if partial else 'Full'}"


def run_plots(runs_dir="runs", plots_dir="plots", suffix="", algos=None, label_suffix=""):
    """Save one PNG per (algo, env). Only loads runs whose filename contains suffix.
    label_suffix is added to plot titles, e.g. ' (proj2)' or ' (h64)'."""
    print(f"\n{'='*80}\nGenerating plots (suffix='{suffix}', dir='{plots_dir}')\n{'='*80}", flush=True)
    os.makedirs(plots_dir, exist_ok=True)
    algos = algos or ALGOS
    summary_rows = []

    for env_name, partial in ENV_SPECS_PLOT:
        for algo in algos:
            results = _load_seeds(algo, env_name, partial, suffix, runs_dir)
            if not results:
                print(f"  SKIP: no results for {algo} {env_name} {'PO' if partial else 'Full'}", flush=True)
                continue

            iters = np.array(results[0]["eval_iterations"])
            curves = np.array([r["eval_means"] for r in results])
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            finals = np.array([r["final_eval_mean"] for r in results])
            fm, fs = finals.mean(), finals.std()

            fig, ax = plt.subplots(figsize=(10, 5.5))
            color = "#1f77b4"
            ax.plot(iters, mean, color=color, linewidth=1.2,
                    label=f"Eval avg (n={len(results)})")
            ax.fill_between(iters, mean - std, mean + std, color=color, alpha=0.2)
            ax.set_title(f"{_pretty_env(env_name, partial)} - LSTM {algo}{label_suffix}")
            ax.set_xlabel("Search iterations" if algo == "ARS" else "PPO updates")
            ax.set_ylabel("Return")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right")
            fig.text(0.02, 0.02, f"Final eval return: {fm:.1f} ± {fs:.1f}",
                     family="monospace", fontsize=10)
            plt.tight_layout(rect=[0, 0.05, 1, 1])

            po = "PO" if partial else "Full"
            out_path = os.path.join(plots_dir, f"{algo}_{env_name}_{po}{suffix}.png")
            plt.savefig(out_path, dpi=130, bbox_inches="tight")
            plt.close(fig)

            summary_rows.append({
                "algo": algo, "env": _pretty_env(env_name, partial),
                "n_seeds": len(results), "final_mean": fm, "final_std": fs,
            })
            print(f"  saved {out_path}  (n={len(results)}, final={fm:.1f} ± {fs:.1f})", flush=True)

    summary_path = os.path.join(plots_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"{'Algo':6s} {'Environment':30s} {'Seeds':>6s} {'Final Return':>20s}\n")
        f.write("-" * 70 + "\n")
        for row in summary_rows:
            f.write(f"{row['algo']:6s} {row['env']:30s} {row['n_seeds']:>6d} "
                    f"{row['final_mean']:>10.1f} ± {row['final_std']:<8.1f}\n")

    print(f"\nSummary table saved to {summary_path}", flush=True)