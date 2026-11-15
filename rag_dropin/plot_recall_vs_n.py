#!/usr/bin/env python3
"""Create a dependency-free SVG of state-weighted G Recall@5 versus budget."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/utility/g_recall_at5.csv")
    parser.add_argument("--out", default="results/figures/fig_g_recall_at5_vs_n.svg")
    args = parser.parse_args()

    groups: dict[str, dict[int, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups[row["env"]][int(row["budget"])].append(
                (int(row["n_states"]), float(row["recall_at5"]))
            )
    if not groups:
        raise SystemExit(f"No Recall@5 rows in {args.input}")

    labels = {"alfworld": "ALFWorld", "sciworld": "ScienceWorld test",
              "sciworld_seen": "ScienceWorld seen", "bytesized32": "ByteSized32"}
    colors = {"alfworld": "#0072B2", "sciworld": "#D55E00",
              "sciworld_seen": "#009E73", "bytesized32": "#CC79A7"}
    width, height, left, right, top, bottom = 760, 470, 85, 25, 30, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    budgets = [32, 64, 128, 256, 512, 1024]
    x = lambda budget: left + budgets.index(budget) * plot_w / (len(budgets) - 1)
    y = lambda value: top + (1 - value) * plot_h
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#222}.tick{font-size:13px}.label{font-size:15px}.legend{font-size:14px}</style>']
    for tick in range(6):
        value, yy = tick / 5, y(tick / 5)
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#ddd"/>')
        out.append(f'<text class="tick" x="{left-10}" y="{yy+5:.1f}" text-anchor="end">{value:.1f}</text>')
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>')
    out.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>')
    for budget in budgets:
        xx = x(budget)
        out.append(f'<line x1="{xx:.1f}" y1="{height-bottom}" x2="{xx:.1f}" y2="{height-bottom+5}" stroke="#222"/>')
        out.append(f'<text class="tick" x="{xx:.1f}" y="{height-bottom+23}" text-anchor="middle">{budget}</text>')
    for index, (env, by_budget) in enumerate(sorted(groups.items())):
        points = []
        for budget in budgets:
            rows = by_budget.get(budget)
            if rows:
                points.append((x(budget), y(sum(n * value for n, value in rows) / sum(n for n, _ in rows))))
        color = colors.get(env, "#333")
        out.append('<polyline fill="none" stroke="%s" stroke-width="3" points="%s"/>' %
                   (color, " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in points)))
        out.extend(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{color}"/>' for xx, yy in points)
        lx, ly = left + index * 155, 18
        out.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+20}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        out.append(f'<text class="legend" x="{lx+26}" y="{ly+5}">{labels.get(env, env)}</text>')
    out.append(f'<text class="label" x="{left+plot_w/2:.1f}" y="{height-20}" text-anchor="middle">Training transitions (n)</text>')
    out.append(f'<text class="label" x="20" y="{top+plot_h/2:.1f}" text-anchor="middle" transform="rotate(-90 20 {top+plot_h/2:.1f})">Recall@5 (state-weighted)</text>')
    out.append('</svg>')
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
