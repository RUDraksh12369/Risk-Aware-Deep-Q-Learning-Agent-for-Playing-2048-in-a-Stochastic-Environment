"""Train policy-only RA-D3QN from scratch with unchanged score rewards."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl.agent import DQNConfig
from rl.risk import RiskConfig
from rl.risk_aware import RiskAwareConfig, RiskAwareDQNAgent
from training.train import TrainingConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/ra_d3qn_2048.pt"))
    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes)
    parser.add_argument("--max-steps-per-episode", type=int, default=TrainingConfig.max_steps_per_episode)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda, or cuda:0")
    parser.add_argument("--batch-size", type=int, default=DQNConfig.batch_size)
    parser.add_argument("--min-replay-size", type=int, default=DQNConfig.min_replay_size)
    parser.add_argument("--risk-coefficient", type=float, default=0.1)
    parser.add_argument("--mobility-weight", type=float, default=1.0)
    parser.add_argument("--crowding-weight", type=float, default=1.0)
    parser.add_argument("--game-over-weight", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=TrainingConfig.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=TrainingConfig.epsilon_end)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps_per_episode < 1:
        parser.error("episodes and max-steps-per-episode must be positive")
    if not 0.0 <= args.epsilon_start <= 1.0 or not 0.0 <= args.epsilon_end <= 1.0:
        parser.error("epsilon-start and epsilon-end must be in [0, 1]")
    try:
        risk = RiskAwareConfig(args.risk_coefficient, RiskConfig(
            args.mobility_weight, args.crowding_weight, args.game_over_weight
        ))
        agent = RiskAwareDQNAgent(DQNConfig(
            batch_size=args.batch_size, min_replay_size=args.min_replay_size,
            seed=args.seed, device=args.device,
        ), risk)
    except ValueError as error:
        parser.error(str(error))
    train(
        TrainingConfig(
            episodes=args.episodes, max_steps_per_episode=args.max_steps_per_episode,
            checkpoint_path=str(args.checkpoint), seed=args.seed,
            epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
        ),
        agent.config, agent=agent,
    )


if __name__ == "__main__":
    main()
