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
    parser.add_argument("--dueling", action=argparse.BooleanOptionalAction, default=DQNConfig.dueling,
                        help="Use dueling heads (default); --no-dueling selects plain Double DQN")
    parser.add_argument("--learning-rate", type=float, default=DQNConfig.learning_rate)
    parser.add_argument("--gamma", type=float, default=DQNConfig.gamma)
    parser.add_argument("--replay-capacity", type=int, default=DQNConfig.replay_capacity)
    parser.add_argument("--target-tau", type=float, default=DQNConfig.target_tau)
    parser.add_argument("--epsilon-start", type=float, default=TrainingConfig.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=TrainingConfig.epsilon_end)
    parser.add_argument("--epsilon-decay-steps", type=int, default=TrainingConfig.epsilon_decay_steps)
    parser.add_argument("--checkpoint-every", type=int, default=TrainingConfig.checkpoint_every)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps_per_episode < 1:
        parser.error("episodes and max-steps-per-episode must be positive")
    if not 0 < args.learning_rate < float("inf"):
        parser.error("learning-rate must be positive and finite")
    if not 0 <= args.gamma <= 1 or not 0 < args.target_tau <= 1:
        parser.error("gamma must be in [0, 1] and target-tau in (0, 1]")
    if not 1 <= args.batch_size <= args.min_replay_size <= args.replay_capacity:
        parser.error("require 1 <= batch-size <= min-replay-size <= replay-capacity")
    if not 0 <= args.epsilon_end <= args.epsilon_start <= 1:
        parser.error("require 0 <= epsilon-end <= epsilon-start <= 1")
    if args.epsilon_decay_steps < 1 or args.checkpoint_every < 1:
        parser.error("epsilon-decay-steps and checkpoint-every must be positive")
    agent = DoubleDQNAgent(DQNConfig(
        batch_size=args.batch_size, min_replay_size=args.min_replay_size,
        seed=args.seed, device=args.device,
        dueling=args.dueling, learning_rate=args.learning_rate, gamma=args.gamma,
        replay_capacity=args.replay_capacity, target_tau=args.target_tau,
    ))
    train(
        TrainingConfig(
            episodes=args.episodes, max_steps_per_episode=args.max_steps_per_episode,
            checkpoint_path=str(args.checkpoint), seed=args.seed,
            epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
            epsilon_decay_steps=args.epsilon_decay_steps, checkpoint_every=args.checkpoint_every,
        ),
        agent.config,
        agent=agent,
    )


if __name__ == "__main__":
    main()
