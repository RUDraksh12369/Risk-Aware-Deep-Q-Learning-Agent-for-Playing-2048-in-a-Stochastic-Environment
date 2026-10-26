"""Standalone Double Dueling DQN using the project's shared pipelines.

Train: python -m rl.double_dueling_dqn train --episodes 2000
Evaluate: python -m rl.double_dueling_dqn evaluate --episodes 100
Resume: python -m rl.double_dueling_dqn train --resume models/double_dueling_dqn_2048.pt --checkpoint models/double_dueling_dqn_resumed.pt --episodes 3000

Resume episodes is the total desired episode count, as in training.train.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from rl.agent import DQNAgent, DQNConfig
from training.evaluate import evaluate
from training.train import TrainingConfig, train


DEFAULT_CHECKPOINT = Path("models/double_dueling_dqn_2048.pt")


class DoubleDuelingDQNAgent(DQNAgent):
    """Always use dueling heads and online-selection/target-evaluation targets.

    The shared DQN feature extractor matches rl.duelingDQN.DuelingDQN:
    log2 board / 16 -> shared MLP -> separate value and advantage heads,
    Q(s,a) = V(s) + A(s,a) - mean_a A(s,a).

    Inherited learn() uses the masked online argmax to select a*, then
    gathers Q_target(s', a*) under no_grad. Terminal or empty-mask targets
    equal r. Replay, reward scaling, loss, and target updates stay shared.
    """

    def __init__(self, config: DQNConfig | None = None) -> None:
        # Copy so callers cannot accidentally disable either algorithm, and
        # their original configuration is never changed.
        super().__init__(replace(config or DQNConfig(), dueling=True, double_dqn=True))

    def _checkpoint_metadata(self) -> dict[str, Any]:
        # Keep DQNAgent.load and the existing evaluation CLI compatible.
        return {"format": "double_dueling_dqn_v1"}

    @classmethod
    def _from_checkpoint(cls, config: DQNConfig, checkpoint: dict[str, Any]) -> "DoubleDuelingDQNAgent":
        if not config.dueling or not config.double_dqn:
            raise ValueError("Checkpoint must use both dueling heads and Double DQN targets.")
        return cls(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    training.add_argument("--resume", type=Path)
    training.add_argument("--episodes", type=int, default=TrainingConfig.episodes)
    training.add_argument("--max-steps-per-episode", type=int, default=TrainingConfig.max_steps_per_episode)
    training.add_argument("--checkpoint-every", type=int, default=TrainingConfig.checkpoint_every)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", default=None)
    for name in ("batch_size", "min_replay_size", "replay_capacity"):
        training.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(DQNConfig, name))
    for name in ("learning_rate", "gamma", "target_tau"):
        training.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(DQNConfig, name))
    for name in ("epsilon_start", "epsilon_end", "epsilon_decay_steps"):
        training.add_argument("--" + name.replace("_", "-"), type=int if name.endswith("steps") else float,
                              default=getattr(TrainingConfig, name))
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    evaluation.add_argument("--episodes", type=int, default=100)
    evaluation.add_argument("--seed", type=int, default=1000)
    evaluation.add_argument("--device", default=None)
    evaluation.add_argument("--max-steps-per-episode", type=int, default=None)
    args = parser.parse_args()
    if args.command == "evaluate":
        agent = DoubleDuelingDQNAgent.load(args.checkpoint, args.device)
        metrics = evaluate(agent, args.episodes, args.seed, max_steps_per_episode=args.max_steps_per_episode)
        for name, value in metrics.items():
            print(f"{name}: {value:.3f}")
        return

    if args.episodes < 1 or args.max_steps_per_episode < 1 or args.checkpoint_every < 1:
        parser.error("episodes, max-steps-per-episode and checkpoint-every must be positive")
    if not 1 <= args.batch_size <= args.min_replay_size <= args.replay_capacity:
        parser.error("require 1 <= batch-size <= min-replay-size <= replay-capacity")
    if not 0 < args.learning_rate < float("inf") or not 0 <= args.gamma <= 1 or not 0 < args.target_tau <= 1:
        parser.error("require finite positive learning-rate, gamma in [0,1], target-tau in (0,1]")
    if not 0 <= args.epsilon_end <= args.epsilon_start <= 1 or args.epsilon_decay_steps < 1:
        parser.error("require 0 <= epsilon-end <= epsilon-start <= 1 and positive epsilon-decay-steps")
    # Refuse to overwrite any pre-existing output: resuming writes a new path.
    if args.checkpoint.exists():
        parser.error("Output checkpoint already exists; choose a new --checkpoint path")
    resume = None
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload.get("format") != "double_dueling_dqn_v1":
            parser.error("--resume requires a Double Dueling DQN checkpoint")
        resume = payload.get("extra", {})
        if not all(key in resume for key in ("episode", "global_step", "history", "training_config", "training_state")):
            parser.error("Checkpoint lacks resumable training metadata")
        agent = DoubleDuelingDQNAgent.load(args.resume, args.device)
        config = TrainingConfig(**resume["training_config"])
        config.episodes = args.episodes
        config.checkpoint_path = str(args.checkpoint)
        config.checkpoint_every = args.checkpoint_every
    else:
        agent = DoubleDuelingDQNAgent(DQNConfig(
            batch_size=args.batch_size, min_replay_size=args.min_replay_size,
            replay_capacity=args.replay_capacity, learning_rate=args.learning_rate,
            gamma=args.gamma, target_tau=args.target_tau, seed=args.seed, device=args.device,
        ))
        config = TrainingConfig(
            episodes=args.episodes, max_steps_per_episode=args.max_steps_per_episode,
            checkpoint_path=str(args.checkpoint), checkpoint_every=args.checkpoint_every,
            seed=args.seed, epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
            epsilon_decay_steps=args.epsilon_decay_steps,
        )
    train(config, agent.config, agent=agent, resume=resume, resumable=True)


if __name__ == "__main__":
    main()
