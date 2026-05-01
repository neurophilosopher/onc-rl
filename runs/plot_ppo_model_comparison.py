import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_STYLES = {
    "mlp": {"color": "#2ca02c", "label": "MLP"},
    "mlp_large": {"color": "#9467bd", "label": "MLP-large"},
    "lstm": {"color": "#d62728", "label": "LSTM"},
    "lstm_large": {"color": "#ff7f0e", "label": "LSTM-large"},
}


def load_lstm_series(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    total_iterations = float(data["total_iterations"])
    eval_iterations = np.array(data["eval_iterations"], dtype=float)
    # Map logged PPO iterations to approximate training timesteps.
    timesteps = eval_iterations * (1_000_000.0 / total_iterations)
    returns = np.array(data["eval_means"], dtype=float)
    return timesteps, returns


def load_mlp_series(path: Path):
    with np.load(path) as data:
        timesteps = np.array(data["timesteps"], dtype=float)
        results = np.array(data["results"], dtype=float)
    if results.ndim == 1:
        returns = results
    else:
        returns = results.mean(axis=1)
    return timesteps, returns


def aggregate_model(files, loader):
    if not files:
        return None
    loaded = [loader(path) for path in sorted(files)]
    x0, _ = loaded[0]
    for x, _ in loaded[1:]:
        if x.shape != x0.shape or not np.allclose(x, x0):
            raise ValueError("Inconsistent x-axis across seeds for the same model.")
    stacked = np.vstack([y for _, y in loaded])
    return {
        "steps": x0,
        "mean": stacked.mean(axis=0),
        "std": stacked.std(axis=0),
        "n": stacked.shape[0],
    }


def parse_folder_name(folder: Path):
    name = folder.name
    if not name.endswith("_ppo"):
        raise ValueError(f"Unsupported folder name: {name}")
    stem = name[:-4]
    if stem.endswith("_full"):
        return stem[:-5], "full"
    if stem.endswith("_po"):
        return stem[:-3], "po"
    raise ValueError(f"Unsupported experiment suffix in: {name}")


def prettify_task(task_slug: str):
    return {
        "half_cheetah": "HalfCheetah",
        "inverted_pendulum": "InvertedPendulum",
        "swimmer": "Swimmer",
    }.get(task_slug, task_slug.replace("_", " ").title())


def plot_folder(folder: Path, output: Path | None = None):
    task_slug, experiment = parse_folder_name(folder)
    model_files = {
        "mlp": sorted(folder.glob("mlp_*_seed*.npz")),
        "mlp_large": sorted(folder.glob("mlp_large_*_seed*.npz")),
        "lstm": sorted(folder.glob("lstm_*_seed*.json")),
        "lstm_large": sorted(folder.glob("lstm_large_*_seed*.json")),
    }

    # Exclude large variants from the small-variant globs.
    model_files["mlp"] = [p for p in model_files["mlp"] if not p.name.startswith("mlp_large_")]
    model_files["lstm"] = [p for p in model_files["lstm"] if not p.name.startswith("lstm_large_")]

    aggregated = {
        "mlp": aggregate_model(model_files["mlp"], load_mlp_series),
        "mlp_large": aggregate_model(model_files["mlp_large"], load_mlp_series),
        "lstm": aggregate_model(model_files["lstm"], load_lstm_series),
        "lstm_large": aggregate_model(model_files["lstm_large"], load_lstm_series),
    }

    if not any(aggregated.values()):
        raise ValueError(f"No plottable files found in {folder}")

    plt.figure(figsize=(9, 5.5))
    for model in ("mlp", "mlp_large", "lstm", "lstm_large"):
        result = aggregated[model]
        if result is None:
            continue
        style = MODEL_STYLES[model]
        plt.plot(result["steps"], result["mean"], color=style["color"], linewidth=2.5, label=style["label"])
        plt.fill_between(
            result["steps"],
            result["mean"] - result["std"],
            result["mean"] + result["std"],
            color=style["color"],
            alpha=0.20,
        )

    plt.tick_params(axis="both", labelsize=11)
    plt.xlabel("Training timesteps")
    plt.ylabel("Mean return over 50 eval episodes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output is None:
        output = folder / f"ppo_comparison_{task_slug}_{experiment}.png"
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate MLP/MLP-large/LSTM/LSTM-large PPO comparison plots."
    )
    parser.add_argument(
        "folders",
        nargs="*",
        type=Path,
        help="Specific *_ppo folders to plot. Defaults to all runs/*_ppo folders.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    folders = args.folders if args.folders else sorted(Path("runs").glob("*_ppo"))
    for folder in folders:
        if not folder.is_dir():
            continue
        try:
            output = plot_folder(folder)
        except ValueError as exc:
            if "Unsupported experiment suffix" in str(exc) or "Unsupported folder name" in str(exc):
                continue
            raise
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
