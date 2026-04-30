# An LLM was used to help write this file. It runs one experiment in a worker
# process, saves the result to a JSON file, and returns a short summary back to
# the parallel launcher in main.py.

import os
import json
import traceback
import torch

from training import train_ars_with_eval, train_ppo_with_eval

torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def tag_for(config):
    """Build the filename for an experiment based on its config."""
    po = "PO" if config["partial"] else "Full"
    suffix = config.get("tag_suffix", "")
    return f"{config['algo']}_{config['env_name']}_{po}{suffix}_seed{config['seed']}"


def run_one(config):
    """Train one experiment, save the result to disk, and return a short summary."""
    torch.set_num_threads(1)
    out_dir = config.get("out_dir", "runs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag_for(config)}.json")

    # Skip if already run
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                cached = json.load(f)
            return {"status": "cached", "config": config, "result_path": out_path,
                    "summary": {"final_eval_mean": cached["final_eval_mean"],
                                "final_eval_std": cached["final_eval_std"],
                                "elapsed_seconds": cached["elapsed_seconds"]}}
        except Exception:
            pass

    try:
        if config["algo"] == "ARS":
            result = train_ars_with_eval(
                env_name=config["env_name"], partial=config["partial"],
                hidden_dim=config["hidden_dim"],
                total_iterations=config["total_iterations"],
                seed=config["seed"],
                eval_every=config.get("eval_every", 50),
                eval_episodes=config.get("eval_episodes", 50),
                project_dim=config.get("project_dim", None))
        elif config["algo"] == "PPO":
            result = train_ppo_with_eval(
                env_name=config["env_name"], partial=config["partial"],
                hidden_dim=config["hidden_dim"],
                total_iterations=config["total_iterations"],
                seed=config["seed"],
                eval_every=config.get("eval_every", 50),
                eval_episodes=config.get("eval_episodes", 50),
                project_dim=config.get("project_dim", None))
        else:
            raise ValueError(f"Unknown algo: {config['algo']}")

        with open(out_path, "w") as f:
            json.dump(result, f)
        return {"status": "ok", "config": config, "result_path": out_path,
                "summary": {"final_eval_mean": result["final_eval_mean"],
                            "final_eval_std": result["final_eval_std"],
                            "elapsed_seconds": result["elapsed_seconds"]}}
    except Exception as e:
        return {"status": "error", "config": config,
                "error": str(e), "traceback": traceback.format_exc()}