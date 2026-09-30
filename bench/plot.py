"""Plot how much each agent changes MMLU accuracy from the no-communication baseline,
overall and in each MMLU category, and what that accuracy costs in time.

    uv run --group bench bench/plot.py bench/results/<model>

delta.png shows the paired accuracy difference from the baseline on the same questions,
with its 95% confidence interval. A bar right of zero means the agent helped in that
category, and a bar left of zero means it hurt.

cost.png shows each agent's overall accuracy, with its 95% Wilson interval, against the
mean wall-clock seconds it took to answer a question.
"""

import statistics
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import mmlu
from evaluate import RESULTS, load, paired, wilson


def main():
    parser = ArgumentParser(description="Plot accuracy change vs. baseline by category.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[RESULTS],
        help="results files or directories to read (default: bench/results)",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        metavar="AGENT",
        help="only plot these agents (the baseline is always kept)",
    )
    parser.add_argument(
        "--baseline",
        default="single",
        help="agent the others are compared with (default: single)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="compare on every question each agent shares with the baseline, instead of "
        "only on the questions that every agent answered",
    )
    parser.add_argument(
        "--cost-out",
        type=Path,
        help="image file to write the accuracy vs. time plot to "
        "(default: <first path>/cost.png)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="image file to write the category plot to (default: <first path>/delta.png)",
    )
    args = parser.parse_args()

    groups = load(args.paths)
    if args.agents:
        keep = {*args.agents, args.baseline}
        groups = {k: g for k, g in groups.items() if k[1] in keep}
    models = sorted({k[0] for k in groups})
    if len(models) != 1:
        parser.error(f"expected results for one model, found {models or 'none'}")
    bases = sorted(k for k in groups if k[1] == args.baseline)
    if not bases:
        parser.error(f"no results for the baseline {args.baseline!r}")
    baseline = groups[bases[0]]

    if not args.all:
        common = set.intersection(*(set(g) for g in groups.values()))
        if not common:
            parser.error("the agents have no questions in common; pass --all")
        groups = {k: {i: g[i] for i in common} for k, g in groups.items()}
        baseline = groups[bases[0]]

    def subset(records: dict[int, dict], category: str | None) -> dict[int, dict]:
        if category is None:
            return records
        return {
            i: r for i, r in records.items() if mmlu.CATEGORY[r["subject"]] == category
        }

    panels = [None, *mmlu.CATEGORIES]
    agents = [k for k in groups if k != bases[0]]
    # {agent: [(delta, ci, n) per panel]} in percentage points
    deltas = {}
    for key in agents:
        row = []
        for category in panels:
            a, b = subset(groups[key], category), subset(baseline, category)
            delta, ci = paired(a, b)
            row.append((100 * delta, 100 * ci, len(a.keys() & b.keys())))
        deltas[key] = row
    # best overall change at the top
    agents.sort(key=lambda k: deltas[k][0][0])

    ranks = {k[2] for k in agents}
    labels = [k[1] if len(ranks) == 1 else f"{k[1]}/{k[2]}" for k in agents]

    fig, axes = plt.subplots(
        1,
        len(panels),
        sharex=True,
        sharey=True,
        figsize=(3 * len(panels), 0.35 * len(agents) + 1.5),
    )
    y = range(len(agents))
    for ax, (j, category) in zip(axes, enumerate(panels)):
        values = [deltas[k][j][0] for k in agents]
        errors = [deltas[k][j][1] for k in agents]
        colors = ["tab:green" if v > 0 else "tab:red" for v in values]
        ax.barh(y, values, xerr=errors, color=colors, capsize=2)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{category or 'Overall'} (n={deltas[agents[0]][j][2]})")
        ax.set_xlabel(f"Δ accuracy vs {args.baseline} (pp)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(list(y), labels)

    fig.suptitle(f"MMLU · {models[0]}")
    fig.tight_layout()

    folder = args.paths[0] if args.paths[0].is_dir() else args.paths[0].parent
    out = args.out or folder / "delta.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")

    # accuracy vs. mean time to answer a question, one point per agent
    fig, ax = plt.subplots(figsize=(7, 5))
    for key in [bases[0], *agents]:
        rs = list(groups[key].values())
        correct = sum(r["correct"] for r in rs)
        lo, hi = wilson(correct, len(rs))
        acc = 100 * correct / len(rs)
        seconds = statistics.fmean(r["seconds"] for r in rs)
        color = "black" if key == bases[0] else "tab:blue"
        ax.errorbar(
            seconds,
            acc,
            yerr=[[acc - 100 * lo], [100 * hi - acc]],
            fmt="o",
            color=color,
            capsize=2,
        )
        label = key[1] if len(ranks) == 1 else f"{key[1]}/{key[2]}"
        ax.annotate(
            label, (seconds, acc), xytext=(4, 4), textcoords="offset points", fontsize=8
        )
    ax.axhline(
        100 * sum(r["correct"] for r in baseline.values()) / len(baseline),
        color="black",
        linewidth=0.8,
        linestyle="--",
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.set_xlabel("mean time per question (s, log scale)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"MMLU · {models[0]} (n={len(baseline)})")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()

    out = args.cost_out or folder / "cost.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
