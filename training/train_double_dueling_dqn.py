"""Train Double Dueling DQN with the canonical DQN command-line interface."""

from __future__ import annotations

import argparse

from rl.agent import DQNConfig
from rl.double_dueling_dqn import DoubleDuelingDQNAgent
from training.train import TrainingConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes)
    parser.add_argument("--checkpoint", default="models/double_dueling_dqn_2048.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda, or cuda:0")
    parser.add_argument("--batch-size", type=int, default=DQNConfig.batch_size)
    parser.add_argument("--min-replay-size", type=int, default=DQNConfig.min_replay_size)
    args = parser.parse_args()
    config = DQNConfig(
        batch_size=args.batch_size,
        min_replay_size=args.min_replay_size,
        seed=args.seed,
        device=args.device,
    )
    agent = DoubleDuelingDQNAgent(config)
    train(
        TrainingConfig(
            episodes=args.episodes,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
        ),
        agent.config,
        agent=agent,
    )


if __name__ == "__main__":
    main()
