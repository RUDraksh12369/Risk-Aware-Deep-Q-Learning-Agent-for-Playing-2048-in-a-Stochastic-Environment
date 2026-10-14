"""Train Double DQN from scratch using the existing DQN architecture."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl.agent import DQNConfig
from rl.doubledqn import DoubleDQNAgent
from training.train import TrainingConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/doubledqn_2048.pt"))
    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes)
    parser.add_argument("--max-steps-per-episode", type=int, default=TrainingConfig.max_steps_per_episode)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda, or cuda:0")
    parser.add_argument("--batch-size", type=int, default=DQNConfig.batch_size)
    parser.add_argument("--min-replay-size", type=int, default=DQNConfig.min_replay_size)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps_per_episode < 1:
        parser.error("episodes and max-steps-per-episode must be positive")
    agent = DoubleDQNAgent(DQNConfig(
        batch_size=args.batch_size, min_replay_size=args.min_replay_size,
        seed=args.seed, device=args.device,
    ))
    train(
        TrainingConfig(
            episodes=args.episodes, max_steps_per_episode=args.max_steps_per_episode,
            checkpoint_path=str(args.checkpoint), seed=args.seed,
        ),
        agent.config,
        agent=agent,
    )


if __name__ == "__main__":
    main()
