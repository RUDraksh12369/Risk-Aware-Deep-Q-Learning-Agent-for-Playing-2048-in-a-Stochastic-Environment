"""Train or resume the policy-only Risk-Aware Dueling Double DQN."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rl.agent import DQNConfig
from rl.risk import RiskConfig
from rl.risk_aware import RiskAwareConfig, RiskAwareDQNAgent
from training.train import TrainingConfig, train


DEFAULT_CHECKPOINT = Path("models/ra_d3qn_2048.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Output checkpoint (default: resume source or models/ra_d3qn_2048.pt)")
    parser.add_argument("--fresh", action="store_true", help="Start over even if the checkpoint exists")
    parser.add_argument("--resume", type=Path, help="Resume another RA-D3QN checkpoint; fail if missing")
    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes)
    parser.add_argument("--max-steps-per-episode", type=int, default=TrainingConfig.max_steps_per_episode)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda, or cuda:0")
    for name in ("batch_size", "min_replay_size", "replay_capacity"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(DQNConfig, name))
    for name in ("learning_rate", "gamma", "target_tau"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(DQNConfig, name))
    for name in ("epsilon_start", "epsilon_end", "epsilon_decay_steps"):
        parser.add_argument("--" + name.replace("_", "-"), type=int if name.endswith("steps") else float,
                            default=getattr(TrainingConfig, name))
    parser.add_argument("--risk-coefficient", type=float, default=RiskAwareConfig.risk_coefficient)
    parser.add_argument("--mobility-weight", type=float, default=RiskConfig.mobility_weight)
    parser.add_argument("--crowding-weight", type=float, default=RiskConfig.crowding_weight)
    parser.add_argument("--game-over-weight", type=float, default=RiskConfig.game_over_weight)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint or args.resume or DEFAULT_CHECKPOINT
    if args.episodes < 1 or args.max_steps_per_episode < 1 or args.checkpoint_every < 1:
        parser.error("episodes, max-steps-per-episode and checkpoint-every must be positive")
    if not 1 <= args.batch_size <= args.min_replay_size <= args.replay_capacity:
        parser.error("require 1 <= batch-size <= min-replay-size <= replay-capacity")
    if not 0 < args.learning_rate < float("inf") or not 0 <= args.gamma <= 1 or not 0 < args.target_tau <= 1:
        parser.error("require finite positive learning-rate, gamma in [0,1], target-tau in (0,1]")
    if not 0 <= args.epsilon_end <= args.epsilon_start <= 1 or args.epsilon_decay_steps < 1:
        parser.error("require 0 <= epsilon-end <= epsilon-start <= 1 and positive epsilon-decay-steps")
    try:
        risk = RiskAwareConfig(args.risk_coefficient, RiskConfig(
            args.mobility_weight, args.crowding_weight, args.game_over_weight
        ))
    except ValueError as error:
        parser.error(str(error))
    if args.fresh and args.resume:
        parser.error("--fresh and --resume cannot be combined")
    source = args.resume or args.checkpoint
    if args.resume and not source.is_file():
        parser.error(f"Resume checkpoint not found: {source}")

    if source.exists() and not args.fresh:
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if payload.get("agent_type") != "ra_d3qn":
            parser.error("Resume requires an RA-D3QN checkpoint")
        extra = payload.get("extra", {})
        if not all(key in extra for key in ("episode", "global_step", "history")):
            parser.error("Checkpoint lacks episode/step/history metadata required for resume")
        agent = RiskAwareDQNAgent.load(source, device=args.device)
        saved_training = extra.get("training_config", {})
        if "training_state" not in extra:
            print("Warning: checkpoint has no replay/RNG state; replay warm-up restarts.")
        training = TrainingConfig(**saved_training) if saved_training else TrainingConfig(
            max_steps_per_episode=args.max_steps_per_episode, seed=args.seed,
            epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
            epsilon_decay_steps=args.epsilon_decay_steps,
        )
        training.episodes = args.episodes
        training.checkpoint_path = str(args.checkpoint)
        training.checkpoint_every = args.checkpoint_every
        print(f"Resuming {source} after episode {extra['episode']}, step {extra['global_step']}. "
              "Saved model, risk, and available training settings take precedence over CLI settings.")
        train(training, agent.config, agent=agent, resume=extra, resumable=True)
        return

    agent = RiskAwareDQNAgent(DQNConfig(
        batch_size=args.batch_size, min_replay_size=args.min_replay_size,
        replay_capacity=args.replay_capacity, learning_rate=args.learning_rate,
        gamma=args.gamma, target_tau=args.target_tau, seed=args.seed, device=args.device,
    ), risk)
    train(
        TrainingConfig(
            episodes=args.episodes, max_steps_per_episode=args.max_steps_per_episode,
            checkpoint_path=str(args.checkpoint), checkpoint_every=args.checkpoint_every,
            seed=args.seed, epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
            epsilon_decay_steps=args.epsilon_decay_steps,
        ),
        agent.config, agent=agent, resumable=True,
    )


if __name__ == "__main__":
    main()
