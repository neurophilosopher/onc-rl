import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_STYLES = {
    "onc": {"color": "#1f77b4", "label": "ONC"},
    "lstm": {"color": "#d62728", "label": "LSTM"},
    "mlp": {"color": "#2ca02c", "label": "MLP"},
}


def load_onc_series(path: Path):
    points = OrderedDict()
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        for row in reader:
            if len(row) < 2:
                continue
            points[int(row[0])] = float(row[1])
    return np.array(list(points.keys())), np.array(list(points.values()))


def load_lstm_series(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    x = np.array(data["eval_iterations"], dtype=float)
    y = np.array(data["eval_means"], dtype=float)
    return x, y


def load_mlp_series(path: Path):
    with np.load(path) as data:
        x = np.array(data["timesteps"], dtype=float)
        y = np.array(data["results"], dtype=float).squeeze()
    if y.ndim > 1:
        y = y.mean(axis=1)
    return x, y


def common_grid(series_list):
    grid = set(series_list[0][0].tolist())
    for x, _ in series_list[1:]:
        grid &= set(x.tolist())
    return np.array(sorted(grid), dtype=float)


def align_to_grid(x, y, grid):
    lookup = {float(step): float(value) for step, value in zip(x, y)}
    return np.array([lookup[float(step)] for step in grid], dtype=float)


def aggregate_model(files, loader):
    if not files:
        return None
    loaded = [loader(path) for path in sorted(files)]
    grid = common_grid(loaded)
    aligned = np.vstack([align_to_grid(x, y, grid) for x, y in loaded])
    return {
        "steps": grid,
        "mean": aligned.mean(axis=0),
        "std": aligned.std(axis=0),
    }


def prettify_task(task_slug: str):
    return {
        "half_cheetah": "HalfCheetah",
        "inverted_pendulum": "InvertedPendulum",
        "swimmer": "Swimmer",
    }.get(task_slug, task_slug.replace("_", " ").title())


def parse_folder_name(folder: Path):
    name = folder.name
    if not name.endswith("_ars"):
        raise ValueError(f"Unsupported folder name: {name}")
    stem = name[:-4]
    if stem.endswith("_full"):
        return stem[:-5], "full"
    if stem.endswith("_po"):
        return stem[:-3], "po"
    raise ValueError(f"Unsupported experiment suffix in: {name}")


def plot_folder(folder: Path, output: Path | None = None):
    task_slug, experiment = parse_folder_name(folder)
    model_files = {
        "onc": sorted(folder.glob("onc_ARS_*.log")),
        "lstm": sorted(folder.glob("lstm_ARS_*.json")),
        "mlp": sorted(folder.glob("mlp_ARS_*.npz")),
    }

    aggregated = {
        "onc": aggregate_model(model_files["onc"], load_onc_series),
        "lstm": aggregate_model(model_files["lstm"], load_lstm_series),
        "mlp": aggregate_model(model_files["mlp"], load_mlp_series),
    }

    if not any(aggregated.values()):
        raise ValueError(f"No plottable files found in {folder}")

    plt.figure(figsize=(9, 5.5))
    for model in ("onc", "lstm", "mlp"):
        result = aggregated[model]
        if result is None:
            continue
        style = MODEL_STYLES[model]
        steps = result["steps"]
        mean = result["mean"]
        std = result["std"]
        plt.plot(steps, mean, color=style["color"], linewidth=2.5, label=style["label"])
        plt.fill_between(steps, mean - std, mean + std, color=style["color"], alpha=0.20)

    plt.tick_params(axis="both", labelsize=11)
    plt.xlabel("Search iterations")
    plt.ylabel("Mean return over 50 eval episodes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output is None:
        output = folder / f"ars_comparison_{task_slug}_{experiment}.png"
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate ONC/LSTM/MLP ARS comparison plots for experiment folders."
    )
    parser.add_argument(
        "folders",
        nargs="*",
        type=Path,
        help="Specific *_ars folders to plot. Defaults to all runs/*_ars folders.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.folders:
        folders = args.folders
    else:
        folders = sorted(Path("runs").glob("*_ars"))

    for folder in folders:
        if not folder.is_dir():
            continue
        output = plot_folder(folder)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
