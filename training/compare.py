"""Train/evaluate explicit 2048 baselines across seeds and retain run artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import platform
from typing import Any, Sequence

import numpy as np
import torch

from rl.agent import DQNAgent, DQNConfig
from rl.baselines import GreedyAgent, RandomAgent
from rl.doubledqn import DoubleDQNAgent
from rl.risk import RiskConfig
from rl.risk_aware import RiskAwareConfig, RiskAwareDQNAgent
from training.evaluate import evaluate
from training.train import TrainingConfig, train


# Values are (double_dqn, dueling). Never rely on DQNConfig's enhanced defaults.
LEARNED_VARIANTS = {
    "dqn": (False, False),
    "double_dqn": (True, False),
    "dueling_dqn": (False, True),
    "dueling_double_dqn": (True, True),
    "ra_d3qn": (True, True),
}
VARIANTS = ("random", "greedy", *LEARNED_VARIANTS)


def create_agent(
    variant: str, config: DQNConfig, risk: RiskAwareConfig | None = None
) -> DQNAgent | RandomAgent | GreedyAgent:
    """Construct a fresh policy; do not mutate the shared base configuration."""
    if variant == "random":
        return RandomAgent(seed=config.seed)
    if variant == "greedy":
        return GreedyAgent()
    if variant not in LEARNED_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    double, dueling = LEARNED_VARIANTS[variant]
    configured = replace(config, double_dqn=double, dueling=dueling)
    if variant == "ra_d3qn":
        return RiskAwareDQNAgent(configured, risk)
    if double:
        return DoubleDQNAgent(configured)
    return DQNAgent(configured)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            # Warm-up loss is NaN in the existing trainer: export it as missing,
            # never as a fabricated zero or a nonstandard JSON NaN literal.
            writer.writerow({key: None if isinstance(value, float) and not math.isfinite(value) else value
                             for key, value in record.items()})


def summarize_results(
    records: list[dict[str, Any]], metric_names: Sequence[str]
) -> list[dict[str, Any]]:
    """Mean and sample standard deviation of per-run evaluation metrics.

    These are across training/policy seeds, not confidence intervals over
    independent test episodes. A single run has no estimated sample std.
    """
    summary = []
    for variant in dict.fromkeys(record["agent"] for record in records):
        runs = [record for record in records if record["agent"] == variant]
        row: dict[str, Any] = {"agent": variant, "runs": len(runs)}
        for name in metric_names:
            values = np.asarray([run[name] for run in runs], dtype=float)
            row[f"{name}_mean"] = float(values.mean())
            row[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        summary.append(row)
    return summary


def run_comparison(
    output_dir: str | Path,
    *, variants: Sequence[str] = VARIANTS, seeds: Sequence[int] = (42, 43, 44),
    training: TrainingConfig | None = None, dqn: DQNConfig | None = None,
    risk: RiskAwareConfig | None = None, evaluation_episodes: int = 100,
    evaluation_seed: int = 1000, evaluation_max_steps: int | None = None,
    high_risk_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Reuse shared training/evaluation with equal episode budgets and held-out seeds.

    Checkpoints and histories are isolated per variant/seed. The output folder
    must not already exist. Partial results and failure status are retained.
    Seeds control training initialization/environment and random-policy RNG;
    all runs use the same evaluation environment seed range. A deterministic
    greedy agent is repeated for table alignment, not independent replication.
    """
    variants, seeds = tuple(variants), tuple(seeds)
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError("variants must be nonempty, unique, and supported.")
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("seeds must be unique integers in [0, 2**32).")
    if evaluation_episodes < 1 or evaluation_seed < 0:
        raise ValueError("evaluation_episodes must be positive and evaluation_seed nonnegative.")
    if evaluation_max_steps is not None and evaluation_max_steps < 1:
        raise ValueError("evaluation_max_steps must be positive or None.")
    if not 0.0 <= high_risk_threshold <= 1.0:
        raise ValueError("high_risk_threshold must be in [0, 1].")
    if any(evaluation_seed <= seed < evaluation_seed + evaluation_episodes for seed in seeds):
        raise ValueError("Training/policy seeds must not overlap the evaluation seed range.")
    training, dqn, risk = training or TrainingConfig(), dqn or DQNConfig(), risk or RiskAwareConfig()
    if training.episodes < 1 or training.max_steps_per_episode < 1 or training.checkpoint_every < 1:
        raise ValueError("Training episode, step, and checkpoint limits must be positive.")
    if not 0 <= training.epsilon_start <= 1 or not 0 <= training.epsilon_end <= 1 or training.epsilon_decay_steps < 1:
        raise ValueError("Invalid exploration schedule.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "variants": list(variants), "seeds": list(seeds),
        "training": {key: value for key, value in asdict(training).items() if key not in ("seed", "checkpoint_path")},
        "base_dqn": asdict(dqn), "risk": asdict(risk),
        "evaluation": {"episodes": evaluation_episodes, "seed": evaluation_seed,
                       "max_steps_per_episode": evaluation_max_steps, "high_risk_threshold": high_risk_threshold},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "notes": ["Fresh training; equal episode/step caps, not equal realized update counts.",
                  "Evaluation environment seed range is shared across all variants and training seeds.",
                  "Greedy rows repeat a deterministic policy and are not independent training replicates.",
                  "RA-D3QN is policy-only; rewards and Double DQN targets are unchanged.",
                  "Summary std is sample std across run metrics; blank for a single run."],
    }
    _write_json(output / "manifest.json", manifest)
    records: list[dict[str, Any]] = []
    try:
        for variant in variants:
            for seed in seeds:
                print(f"agent={variant} seed={seed}", flush=True)
                run_dir = output / variant / f"seed_{seed}"
                run_dir.mkdir(parents=True)
                agent = create_agent(variant, replace(dqn, seed=seed), risk)
                checkpoint_path, history_path = None, None
                training_steps, updates = 0, 0
                if isinstance(agent, DQNAgent):
                    checkpoint_path = run_dir / "checkpoint.pt"
                    history_path = run_dir / "history.csv"
                    history = train(replace(training, seed=seed, checkpoint_path=str(checkpoint_path)),
                                    agent.config, agent=agent)
                    _write_csv(history_path, history)
                    training_steps = int(sum(record["steps"] for record in history))
                    updates = agent.learn_steps
                    # Evaluate the saved artifact, including its saved risk policy.
                    agent = DQNAgent.load(checkpoint_path, device=str(agent.device))
                episode_records = []
                metrics = evaluate(agent, episodes=evaluation_episodes, seed=evaluation_seed,
                                   max_steps_per_episode=evaluation_max_steps, high_risk_threshold=high_risk_threshold,
                                   episode_records=episode_records)
                evaluation_path = run_dir / "evaluation.csv"
                _write_csv(evaluation_path, episode_records)
                row = {
                    "agent": variant, "seed": seed, "evaluation_seed": evaluation_seed,
                    "training_steps": training_steps, "learning_updates": updates,
                    "device": str(agent.device) if isinstance(agent, DQNAgent) else "cpu",
                    "parameters": sum(p.numel() for p in agent.online_net.parameters()) if isinstance(agent, DQNAgent) else 0,
                    "checkpoint": str(checkpoint_path.relative_to(output)) if checkpoint_path else None,
                    "history": str(history_path.relative_to(output)) if history_path else None,
                    "evaluation_history": str(evaluation_path.relative_to(output)),
                    "double_dqn": agent.config.double_dqn if isinstance(agent, DQNAgent) else None,
                    "dueling": agent.config.dueling if isinstance(agent, DQNAgent) else None,
                    "risk_coefficient": risk.risk_coefficient if variant == "ra_d3qn" else 0.0,
                    **metrics,
                }
                _write_json(run_dir / "metrics.json", row)
                records.append(row)
                _write_csv(output / "results.csv", records)
                _write_csv(output / "summary.csv", summarize_results(records, list(metrics)))
        manifest["status"] = "completed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["completed_runs"] = len(records)
        _write_json(output / "manifest.json", manifest)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--agents", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps-per-episode", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--eval-max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DQNConfig.batch_size)
    parser.add_argument("--min-replay-size", type=int, default=DQNConfig.min_replay_size)
    parser.add_argument("--device", default=None)
    parser.add_argument("--risk-coefficient", type=float, default=0.1)
    parser.add_argument("--mobility-weight", type=float, default=1.0)
    parser.add_argument("--crowding-weight", type=float, default=1.0)
    parser.add_argument("--game-over-weight", type=float, default=1.0)
    parser.add_argument("--high-risk-threshold", type=float, default=0.5)
    parser.add_argument("--smoke", action="store_true", help="Override budgets: 1x32 training, 2x32 evaluation, CPU, batch/warmup 4, epsilon 0.")
    args = parser.parse_args()
    training = TrainingConfig(episodes=1 if args.smoke else args.episodes,
                              max_steps_per_episode=32 if args.smoke else args.max_steps_per_episode)
    if args.smoke:
        training = replace(training, epsilon_start=0.0, epsilon_end=0.0)
    try:
        run_comparison(
            args.output_dir, variants=args.agents, seeds=args.seeds or ([42] if args.smoke else [42, 43, 44]),
            training=training,
            dqn=DQNConfig(batch_size=4 if args.smoke else args.batch_size,
                          min_replay_size=4 if args.smoke else args.min_replay_size,
                          device="cpu" if args.smoke else args.device),
            risk=RiskAwareConfig(args.risk_coefficient, RiskConfig(
                args.mobility_weight, args.crowding_weight, args.game_over_weight)),
            evaluation_episodes=2 if args.smoke else args.eval_episodes, evaluation_seed=args.eval_seed,
            evaluation_max_steps=32 if args.smoke else args.eval_max_steps,
            high_risk_threshold=args.high_risk_threshold,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
