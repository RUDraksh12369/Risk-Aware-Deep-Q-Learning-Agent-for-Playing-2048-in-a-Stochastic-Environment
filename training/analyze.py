"""Generate descriptive plots and a report from saved comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = {
    "mean_score": "Evaluation score",
    "tile_2048_rate": "2048 achievement rate",
    "mean_episode_length": "Episode length (moves)",
    "mean_board_risk": "Mean board risk",
    "mean_empty_cells": "Mean empty cells",
    "truncation_rate": "Truncation rate",
}
LABELS = {"random": "Random", "greedy": "Greedy", "dqn": "DQN",
          "double_dqn": "Double\nDQN", "dueling_dqn": "Dueling\nDQN",
          "dueling_double_dqn": "Dueling\nDouble DQN", "ra_d3qn": "RA-D3QN"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError(f"CSV has no records: {path}")
    return records


def _artifact(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact paths must stay inside the comparison directory.")
    return path


def _number(value: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("Metrics must be finite; missing losses belong in the history only.")
    return number


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing episode mean; missing observations stay missing, never zero."""
    if window < 1:
        raise ValueError("window must be positive.")
    result = np.full(len(values), np.nan)
    for index, value in enumerate(values):
        if np.isfinite(value):
            previous = values[max(0, index - window + 1):index + 1]
            finite = previous[np.isfinite(previous)]
            result[index] = finite.mean()
    return result


def _figure(width: float, height: float):
    # Direct Agg canvas: no GUI/display required and no pyplot global figures.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def analyze_comparison(
    input_dir: str | Path, output_dir: str | Path | None = None, *, smoothing_window: int = 25
) -> Path:
    """Read existing data only; create plots/report in a new output directory.

    Error bars describe sample SD across run means, not confidence intervals.
    Episode scores are pooled descriptively and never treated as independent
    training replicates. No models are trained or evaluated by this command.
    """
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be positive.")
    root = Path(input_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = _read_csv(root / "results.csv")
    identities = [(row["agent"], row["seed"]) for row in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate agent/seed rows would inflate replication counts.")
    variants = list(dict.fromkeys(row["agent"] for row in records))
    values = {name: {metric: np.asarray([_number(row[metric]) for row in records if row["agent"] == name])
                     for metric in METRICS} for name in variants}
    histories = []
    episode_scores = {name: [] for name in variants}
    missing_episode_files = 0
    for row in records:
        if row.get("history"):
            history = _read_csv(_artifact(root, row["history"]))
            episodes = np.asarray([_number(item["episode"]) for item in history])
            if np.any(episodes < 1) or np.any(np.diff(episodes) <= 0):
                raise ValueError("Training episodes must be positive and strictly increasing.")
            curves = {key: np.asarray([np.nan if key == "loss" and item[key] == "" else _number(item[key])
                                      for item in history]) for key in ("score", "max_tile", "loss", "epsilon")}
            histories.append((row["agent"], episodes, curves))
        if row.get("evaluation_history"):
            evaluation = _read_csv(_artifact(root, row["evaluation_history"]))
            if len(evaluation) != int(_number(row["episodes"])):
                raise ValueError("Evaluation episode count does not match results.csv.")
            scores = [_number(item["score"]) for item in evaluation]
            if not np.isclose(np.mean(scores), _number(row["mean_score"])):
                raise ValueError("Evaluation episode scores do not match the aggregate mean.")
            episode_scores[row["agent"]].extend(scores)
        else:
            missing_episode_files += 1

    output = Path(output_dir).resolve() if output_dir is not None else root / "analysis"
    output.mkdir(parents=True, exist_ok=False)
    colors = {name: ("#C66B2B" if name == "ra_d3qn" else "#276C91") for name in variants}
    labels = [LABELS.get(name, name) for name in variants]
    figure = _figure(15, 9)
    for axis, (metric, title) in zip(figure.subplots(2, 3).flat, METRICS.items()):
        for index, name in enumerate(variants):
            data = values[name][metric]
            std = float(data.std(ddof=1)) if len(data) > 1 else None
            axis.bar(index, data.mean(), color=colors[name], alpha=0.65, yerr=std, capsize=4)
            offsets = np.linspace(-0.14, 0.14, len(data)) if len(data) > 1 else [0]
            axis.scatter(index + np.asarray(offsets), data, color="#182C39", s=16, zorder=3)
        axis.set_title(title, fontsize=11)
        axis.set_xticks(range(len(variants)), labels, fontsize=8)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle("Evaluation comparison\nBars: run means; whiskers: sample SD across runs; dots: individual runs", fontsize=14)
    figure.savefig(output / "evaluation.png", dpi=140)
    figure.clear()

    if histories:
        figure = _figure(13, 8)
        axes = figure.subplots(2, 2).flat
        palette = ["#276C91", "#38856A", "#79569B", "#B08628", "#C66B2B"]
        learned = list(dict.fromkeys(name for name, _, _ in histories))
        for axis, key in zip(axes, ("score", "max_tile", "loss", "epsilon")):
            labeled = set()
            for name, episodes, curves in histories:
                curve = trailing_mean(curves[key], smoothing_window)
                if not np.isfinite(curve).any():
                    continue
                axis.plot(episodes, curve, marker=".", markersize=3, alpha=0.7,
                          color=palette[learned.index(name) % len(palette)],
                          label=name if name not in labeled else None)
                labeled.add(name)
            axis.set_title({"score": "Training score", "max_tile": "Training max tile",
                            "loss": "Training loss", "epsilon": "Exploration epsilon"}[key])
            axis.set_xlabel("Training episode")
            axis.grid(alpha=0.2)
            if labeled:
                axis.legend(fontsize=7)
            else:
                axis.text(0.5, 0.5, "No finite observations (warm-up)", ha="center", transform=axis.transAxes)
        figure.suptitle(f"Training curves: trailing {smoothing_window}-episode mean\nEach line is one training run; missing values remain gaps", fontsize=13)
        figure.savefig(output / "training.png", dpi=140)
        figure.clear()

    available = [name for name in variants if episode_scores[name]]
    if available:
        figure = _figure(12, 6)
        axis = figure.subplots()
        for index, name in enumerate(available):
            data = np.asarray(episode_scores[name])
            offsets = np.linspace(-0.2, 0.2, len(data)) if len(data) > 1 else [0]
            axis.scatter(index + np.asarray(offsets), data, color=colors[name], s=18, alpha=0.55)
            axis.hlines(np.median(data), index - 0.28, index + 0.28, color="#182C39", linewidth=2)
        axis.set_xticks(range(len(available)), [LABELS.get(name, name) for name in available], fontsize=9)
        axis.set_ylabel("Episode score")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.2)
        axis.set_title("Evaluation episodes pooled descriptively across runs\nDots: episodes; horizontal lines: pooled medians (not independent training replicates)", fontsize=12)
        figure.savefig(output / "episode_scores.png", dpi=140)
        figure.clear()

    status = manifest.get("status", "unknown")
    report = ["# Comparison analysis", "",
              f"Source: `{root}`", f"Run status: **{status}**; recorded agent/seed runs: **{len(records)}**.", "",
              "This report is descriptive. It does not establish statistical significance, convergence, or a winning agent.", "",
              "## Recorded settings", "", "```json",
              json.dumps({key: manifest.get(key) for key in ("training", "evaluation", "risk", "seeds")}, indent=2),
              "```", "", "## Evaluation means", "",
              "Values are means of run-level metrics +/- sample SD across runs. A single run has no estimated SD.", "",
              "| Agent | Runs | Score | Episode length | Board risk | Truncation rate |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in variants:
        cells = []
        for metric in ("mean_score", "mean_episode_length", "mean_board_risk", "truncation_rate"):
            data = values[name][metric]
            spread = f"{data.std(ddof=1):.3f}" if len(data) > 1 else "SD not estimated"
            cells.append(f"{data.mean():.3f} +/- {spread}")
        report.append(f"| {name} | {len(values[name]['mean_score'])} | " + " | ".join(cells) + " |")
    report.extend(["", "## Interpretation limits", "",
                   "- Error bars are sample standard deviations, not confidence intervals.",
                   "- Evaluation episodes share trained policies; pooled episodes are not independent training replicates.",
                   "- Greedy repeats across seeds are deterministic repeats, not independent training samples.",
                   "- Training curves use episode indices; survival differences imply different numbers of steps/updates.",
                   "- Risk metrics are pooled over decision states within each run, then averaged across runs here.",
                   "- Missing warm-up losses remain missing. Short/smoke runs cannot demonstrate convergence."])
    if status != "completed":
        report.append("- **Partial comparison:** runs are missing; do not treat these tables as a complete comparison.")
    if any(np.any(values[name]["truncation_rate"] > 0) for name in variants):
        report.append("- **Truncated evaluation episodes are present:** scores and episode lengths are capped observations, not complete-game outcomes.")
    if missing_episode_files:
        report.append(f"- {missing_episode_files} runs lack episode exports (older format); their aggregate metrics are retained.")
    report.extend(["", "## Figures", "", "![Evaluation comparison](evaluation.png)"])
    if histories:
        report.extend(["", "![Training curves](training.png)"])
    if available:
        report.extend(["", "![Evaluation episode scores](episode_scores.png)"])
    report.extend(["", "## Next experimental work", "",
                   "Run substantive multi-seed training, inspect learning curves, and then evaluate on held-out seeds. Tune risk settings on validation runs, not the final test results.", ""])
    report_path = output / "report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoothing-window", type=int, default=25)
    args = parser.parse_args()
    try:
        report = analyze_comparison(args.input_dir, args.output_dir, smoothing_window=args.smoothing_window)
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
    print(f"Analysis saved to {report}")


if __name__ == "__main__":
    main()
