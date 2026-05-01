import argparse
import csv
import re
from pathlib import Path

TEXT_LOG_PATTERN = re.compile(
    r"^(Improvement|Reevaluate) after: (\d+) steps, with return ([^,]+)"
)


def load_csv_series(csv_path: Path):
    steps = []
    avg_returns = []
    performance_returns = []

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        for row in reader:
            if len(row) < 3:
                continue
            steps.append(int(row[0]))
            avg_returns.append(float(row[1]))
            performance_returns.append(float(row[2]))

    return steps, avg_returns, performance_returns


def load_text_events(text_path: Path):
    steps = []
    event_returns = []
    event_labels = []

    with text_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = TEXT_LOG_PATTERN.match(line.strip())
            if not match:
                continue
            event_labels.append(match.group(1))
            steps.append(int(match.group(2)))
            event_returns.append(float(match.group(3)))

    return steps, event_returns, event_labels


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot return vs. timestep from ONC-RL result logs."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Path to a semicolon-delimited csvlog_*.log file.",
    )
    parser.add_argument(
        "--text-log",
        type=Path,
        help="Path to a textlog_*.log file with improvement/reevaluation events.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("return_vs_timesteps.svg"),
        help="Output SVG path.",
    )
    parser.add_argument(
        "--title",
        default="Return vs. Timesteps",
        help="Plot title.",
    )
    return parser


def _scale(value, src_min, src_max, dst_min, dst_max):
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def write_svg(output_path: Path, series, title: str):
    width = 1000
    height = 600
    margin_left = 90
    margin_right = 30
    margin_top = 50
    margin_bottom = 70

    all_x = [x for _, points, _, _ in series for x, _ in points]
    all_y = [y for _, points, _, _ in series for _, y in points]
    if not all_x or not all_y:
        raise SystemExit("No plottable data found in the provided logs.")

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom

    def px(x):
        return _scale(x, x_min, x_max, plot_left, plot_right)

    def py(y):
        return _scale(y, y_min, y_max, plot_bottom, plot_top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="22" font-family="Arial">{title}</text>',
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#333" stroke-width="2"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#333" stroke-width="2"/>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="16" font-family="Arial">Timesteps</text>',
        f'<text x="25" y="{height / 2}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 25 {height / 2})">Return</text>',
    ]

    for tick in range(6):
        xt = x_min + (x_max - x_min) * tick / 5 if x_max != x_min else x_min
        yt = y_min + (y_max - y_min) * tick / 5 if y_max != y_min else y_min
        x_pos = px(xt)
        y_pos = py(yt)
        parts.append(
            f'<line x1="{x_pos}" y1="{plot_top}" x2="{x_pos}" y2="{plot_bottom}" stroke="#ddd" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{plot_left}" y1="{y_pos}" x2="{plot_right}" y2="{y_pos}" stroke="#ddd" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x_pos}" y="{plot_bottom + 22}" text-anchor="middle" font-size="12" font-family="Arial">{int(round(xt))}</text>'
        )
        parts.append(
            f'<text x="{plot_left - 10}" y="{y_pos + 4}" text-anchor="end" font-size="12" font-family="Arial">{yt:.1f}</text>'
        )

    legend_x = plot_right - 220
    legend_y = plot_top + 10
    for index, (label, points, color, style) in enumerate(series):
        y = legend_y + index * 22
        if style == "line":
            parts.append(
                f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>'
            )
        else:
            parts.append(
                f'<circle cx="{legend_x + 12}" cy="{y}" r="4" fill="{color}"/>'
            )
        parts.append(
            f'<text x="{legend_x + 34}" y="{y + 4}" font-size="13" font-family="Arial">{label}</text>'
        )

    for _, points, color, style in series:
        if style == "line":
            path = " ".join(
                f"{'M' if i == 0 else 'L'} {px(x):.2f} {py(y):.2f}"
                for i, (x, y) in enumerate(points)
            )
            parts.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>'
            )
        else:
            for x, y in points:
                parts.append(
                    f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="2.5" fill="{color}" fill-opacity="0.75"/>'
                )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    args = build_parser().parse_args()

    if args.csv is None and args.text_log is None:
        raise SystemExit("Provide at least one of --csv or --text-log.")

    series = []

    if args.csv is not None:
        steps, avg_returns, performance_returns = load_csv_series(args.csv)
        if steps:
            series.append(
                ("Average return", list(zip(steps, avg_returns)), "#0b6e4f", "line")
            )
            series.append(
                (
                    "Mean sampled return",
                    list(zip(steps, performance_returns)),
                    "#c84c09",
                    "line",
                )
            )

    if args.text_log is not None:
        steps, event_returns, _ = load_text_events(args.text_log)
        if steps:
            series.append(
                (
                    "Improvement/Reevaluation events",
                    list(zip(steps, event_returns)),
                    "#3b5bdb",
                    "points",
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_svg(args.output, series, args.title)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
