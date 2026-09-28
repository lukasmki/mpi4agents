"""Score the results written by harness.py: accuracy per agent with a 95% confidence
interval, the change from a no-communication baseline, accuracy per MMLU category, and what
each question cost.

    uv run --group bench bench/evaluate.py bench/results/<model>
"""

import csv
import json
import math
import statistics
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

import mmlu

RESULTS = Path(__file__).parent / "results"

Key = tuple[str, str, int]  # (model, agent, ranks)


def load(paths: list[Path]) -> dict[Key, dict[int, dict]]:
    """Read every results file under paths into {(model, agent, ranks): {id: record}}. A
    later record of a question replaces an earlier one, and partial lines are skipped."""
    groups: dict[Key, dict[int, dict]] = defaultdict(dict)
    files = [
        f for p in paths for f in ([p] if p.is_file() else sorted(p.rglob("*.jsonl")))
    ]
    for f in files:
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            groups[r["model"], r["agent"], r["ranks"]][r["id"]] = r
    return groups


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n"""
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def paired(a: dict[int, dict], b: dict[int, dict]) -> tuple[float, float]:
    """Accuracy of a minus accuracy of b on the questions both answered, with the half-width
    of its 95% confidence interval"""
    diffs = [a[i]["correct"] - b[i]["correct"] for i in a.keys() & b.keys()]
    if len(diffs) < 2:
        return statistics.fmean(diffs) if diffs else math.nan, math.nan
    return statistics.fmean(diffs), 1.96 * statistics.stdev(diffs) / math.sqrt(
        len(diffs)
    )


def accuracy(records: list[dict]) -> float | None:
    return statistics.fmean(r["correct"] for r in records) if records else None


def summarize(
    key: Key, records: dict[int, dict], baseline: dict[int, dict] | None
) -> dict:
    """One summary row: accuracy overall and per category, extraction and error rates, and
    the mean cost of a question"""
    rs = list(records.values())
    correct = sum(r["correct"] for r in rs)
    lo, hi = wilson(correct, len(rs))
    delta, delta_ci = paired(records, baseline) if baseline else (None, None)
    row = {
        "model": key[0],
        "agent": key[1],
        "ranks": key[2],
        "n": len(rs),
        "accuracy": correct / len(rs),
        "ci_low": lo,
        "ci_high": hi,
        "delta": delta,
        "delta_ci": delta_ci,
        "no_answer": statistics.fmean(r["pred"] is None for r in rs),
        "errors": statistics.fmean(bool(r["errors"]) for r in rs),
        "calls": statistics.fmean(r["usage"]["requests"] for r in rs),
        "input_tokens": statistics.fmean(r["usage"]["input_tokens"] for r in rs),
        "output_tokens": statistics.fmean(r["usage"]["output_tokens"] for r in rs),
        "seconds": statistics.fmean(r["seconds"] for r in rs),
    }
    for category in mmlu.CATEGORIES:
        row[category] = accuracy(
            [r for r in rs if mmlu.CATEGORY[r["subject"]] == category]
        )
    return row


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def tokens(x: float) -> str:
    return f"{x / 1000:.1f}k" if x >= 1000 else f"{x:.0f}"


def main():
    parser = ArgumentParser(description="Score the MMLU results written by harness.py.")
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
        help="only score these agents (the baseline is always kept)",
    )
    parser.add_argument(
        "--baseline",
        default="single",
        help="agent the others are compared with (default: single)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="score each agent on every question it answered, instead of only on the "
        "questions that every agent answered",
    )
    parser.add_argument(
        "--by-subject", action="store_true", help="also show accuracy per subject"
    )
    parser.add_argument("--csv", type=Path, help="also write the summary to a CSV file")
    args = parser.parse_args()

    groups = load(args.paths)
    if args.agents:
        keep = {*args.agents, args.baseline}
        groups = {k: g for k, g in groups.items() if k[1] in keep}
    if not groups:
        parser.error(f"no results found in {', '.join(map(str, args.paths))}")

    caption = "Each agent is scored on every question it answered."
    if not args.all:
        common = set.intersection(*(set(g) for g in groups.values()))
        if not common:
            parser.error("the agents have no questions in common; pass --all")
        groups = {k: {i: g[i] for i in common} for k, g in groups.items()}
        some = next(iter(groups.values()))
        sizes = Counter(mmlu.CATEGORY[some[i]["subject"]] for i in common)
        caption = (
            f"Scored on the {len(common)} questions every agent answered "
            f"({', '.join(f'{c} {sizes[c]}' for c in mmlu.CATEGORIES)}); "
            "pass --all to score each agent on all of its questions."
        )

    # compare each group with the baseline run on the same model, preferring fewer ranks
    baselines: dict[str, Key] = {}
    for key in sorted(k for k in groups if k[1] == args.baseline):
        baselines.setdefault(key[0], key)
    rows = []
    for key, records in groups.items():
        base = baselines.get(key[0])
        baseline = groups[base] if base and base != key else None
        rows.append(summarize(key, records, baseline))
    rows.sort(key=lambda r: (r["model"], -r["accuracy"], r["agent"], r["ranks"]))

    models = sorted({r["model"] for r in rows})
    title = f"MMLU · {models[0]}" if len(models) == 1 else "MMLU"
    label = [] if len(models) == 1 else ["Model"]

    def cells(row: dict) -> list[str]:
        model = [] if len(models) == 1 else [row["model"]]
        return [*model, row["agent"], str(row["ranks"])]

    summary = Table(title=title, caption=caption)
    for name in [*label, "Agent", "Ranks"]:
        summary.add_column(name)
    for name in ["N", "Acc %", "95% CI", f"Δ vs {args.baseline}", "No answer %"]:
        summary.add_column(name, justify="right")
    for name in ["Errors %", "Calls/q", "Tokens in/out", "Sec/q"]:
        summary.add_column(name, justify="right")
    for row in rows:
        if row["delta"] is None:
            delta = "—"
        elif math.isnan(row["delta_ci"]):
            delta = f"{100 * row['delta']:+.1f}"
        else:
            delta = f"{100 * row['delta']:+.1f} ± {100 * row['delta_ci']:.1f}"
        summary.add_row(
            *cells(row),
            str(row["n"]),
            f"[bold]{pct(row['accuracy'])}[/bold]",
            f"{pct(row['ci_low'])}–{pct(row['ci_high'])}",
            delta,
            pct(row["no_answer"]),
            pct(row["errors"]),
            f"{row['calls']:.1f}",
            f"{tokens(row['input_tokens'])} / {tokens(row['output_tokens'])}",
            f"{row['seconds']:.1f}",
        )

    categories = Table(title=f"{title} · accuracy % by category")
    for name in [*label, "Agent", "Ranks"]:
        categories.add_column(name)
    for category in mmlu.CATEGORIES:
        categories.add_column(category, justify="right")
    for row in rows:
        categories.add_row(*cells(row), *(pct(row[c]) for c in mmlu.CATEGORIES))

    console = Console()
    console.print(summary)
    console.print(categories)

    if args.by_subject:
        keys = [(r["model"], r["agent"], r["ranks"]) for r in rows]
        subjects = Table(title=f"{title} · accuracy % by subject")
        subjects.add_column("Subject")
        for model, agent, ranks in keys:
            name = (
                f"{agent}/{ranks}" if len(models) == 1 else f"{model} {agent}/{ranks}"
            )
            subjects.add_column(name, justify="right")
        for names in mmlu.CATEGORIES.values():
            for subject in names:
                accs = [
                    accuracy([r for r in groups[k].values() if r["subject"] == subject])
                    for k in keys
                ]
                if any(a is not None for a in accs):
                    subjects.add_row(subject, *map(pct, accs))
            subjects.add_section()
        console.print(subjects)

    if args.csv:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        console.print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
