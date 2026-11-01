"""Shared command-line configuration and checkpoint lifecycle for model trainers."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import torch

from rl.agent import DQNAgent, DQNConfig
from rl.double_dueling_dqn import DoubleDuelingDQNAgent
from rl.doubledqn import DoubleDQNAgent
from rl.risk import RiskConfig
from rl.risk_aware import RiskAwareConfig, RiskAwareDQNAgent
from training.train import TrainingConfig, train


VARIANT_FLAGS = {
    "dqn": (False, False),
    "double_dqn": (True, False),
    "dueling_dqn": (False, True),
    "dueling_double_dqn": (True, True),
    "ra_d3qn": (True, True),
}
DEFAULT_CHECKPOINTS = {
    "dqn": Path("models/dqn_baseline_2048.pt"),
    "double_dqn": Path("models/doubledqn_2048.pt"),
    "dueling_dqn": Path("models/dueling_dqn_shared.pt"),
    "dueling_double_dqn": Path("models/dueling_double_dqn_2048.pt"),
    "ra_d3qn": Path("models/ra_d3qn_2048.pt"),
}


def _target_tau(value: str) -> float | None:
    if value.lower() in {"none", "hard"}:
        return None
    return float(value)


def add_training_arguments(
    parser: argparse.ArgumentParser, *, variant: str, checkpoint: Path | None = None
) -> None:
    if variant not in VARIANT_FLAGS:
        raise ValueError(f"Unsupported training variant: {variant}")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help=f"Output path (default: {checkpoint or DEFAULT_CHECKPOINTS[variant]})")
    parser.add_argument("--fresh", action="store_true", help="Start from new weights, even if the output exists")
    parser.add_argument("--resume", type=Path, help="Resume from a checkpoint; output defaults to this path")
    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes,
                        help="Total target episode count, including episodes in a resumed checkpoint")
    parser.add_argument("--max-steps-per-episode", type=int, default=TrainingConfig.max_steps_per_episode)
    parser.add_argument("--checkpoint-every", type=int, default=TrainingConfig.checkpoint_every)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda, or cuda:0")
    parser.add_argument("--batch-size", type=int, default=DQNConfig.batch_size)
    parser.add_argument("--min-replay-size", type=int, default=DQNConfig.min_replay_size)
    parser.add_argument("--replay-capacity", type=int, default=DQNConfig.replay_capacity)
    parser.add_argument("--learning-rate", type=float, default=DQNConfig.learning_rate)
    parser.add_argument("--gamma", type=float, default=DQNConfig.gamma)
    parser.add_argument("--target-tau", type=_target_tau, default=DQNConfig.target_tau,
                        help="Soft update rate in (0,1], or 'none' for hard interval updates")
    parser.add_argument("--target-update-interval", type=int, default=DQNConfig.target_update_interval)
    parser.add_argument("--hard-target-updates", action="store_true",
                        help="Use periodic target copies instead of soft updates")
    parser.add_argument("--reward-scale", type=float, default=DQNConfig.reward_scale)
    parser.add_argument("--gradient-clip-norm", type=float, default=DQNConfig.gradient_clip_norm)
    parser.add_argument("--max-exponent", type=float, default=DQNConfig.max_exponent)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=DQNConfig.hidden_dims)
    parser.add_argument("--epsilon-start", type=float, default=TrainingConfig.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=TrainingConfig.epsilon_end)
    parser.add_argument("--epsilon-decay-steps", type=int, default=TrainingConfig.epsilon_decay_steps)
    if variant == "double_dqn":
        parser.add_argument("--dueling", action=argparse.BooleanOptionalAction, default=None,
                            help="Optional dueling-head Double DQN ablation")
    if variant == "ra_d3qn":
        parser.add_argument("--risk-coefficient", type=float, default=RiskAwareConfig.risk_coefficient)
        parser.add_argument("--mobility-weight", type=float, default=RiskConfig.mobility_weight)
        parser.add_argument("--crowding-weight", type=float, default=RiskConfig.crowding_weight)
        parser.add_argument("--game-over-weight", type=float, default=RiskConfig.game_over_weight)


def train_from_args(
    args: argparse.Namespace, *, variant: str, checkpoint: Path | None = None,
    agent_factory: Callable[[DQNConfig], DQNAgent] | None = None,
    agent_loader: Callable[..., DQNAgent] | None = None,
    train_fn: Callable[..., list[dict[str, float]]] = train,
) -> None:
    """Validate common settings, restore compatible checkpoints, and train."""
    parser = args._parser
    default_path = checkpoint or DEFAULT_CHECKPOINTS[variant]
    args.checkpoint = args.checkpoint or args.resume or default_path
    if args.episodes < 1 or args.max_steps_per_episode < 1 or args.checkpoint_every < 1:
        parser.error("episodes, max-steps-per-episode and checkpoint-every must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    if not 1 <= args.batch_size <= args.min_replay_size <= args.replay_capacity:
        parser.error("require 1 <= batch-size <= min-replay-size <= replay-capacity")
    if not 0 < args.learning_rate < float("inf"):
        parser.error("learning-rate must be positive and finite")
    if not 0 <= args.gamma <= 1 or (args.target_tau is not None and not 0 < args.target_tau <= 1):
        parser.error("gamma must be in [0,1] and target-tau in (0,1] or none")
    if args.target_update_interval < 1:
        parser.error("target-update-interval must be positive")
    if not 0 <= args.epsilon_end <= args.epsilon_start <= 1 or args.epsilon_decay_steps < 1:
        parser.error("require 0 <= epsilon-end <= epsilon-start <= 1 and positive epsilon-decay-steps")
    if args.fresh and args.resume:
        parser.error("--fresh and --resume cannot be combined")
    source = args.resume or args.checkpoint
    if args.resume and not source.is_file():
        parser.error(f"Resume checkpoint not found: {source}")

    expected_double, expected_dueling = VARIANT_FLAGS[variant]
    if variant == "double_dqn" and getattr(args, "dueling", None) is not None:
        expected_dueling = args.dueling
    risk = None
    if variant == "ra_d3qn":
        try:
            risk = RiskAwareConfig(args.risk_coefficient, RiskConfig(
                args.mobility_weight, args.crowding_weight, args.game_over_weight
            ))
        except ValueError as error:
            parser.error(str(error))

    if source.exists() and not args.fresh:
        payload = torch.load(source, map_location="cpu", weights_only=False)
        config_data = payload.get("config", {})
        saved_dueling = bool(config_data.get("dueling", False))
        if variant == "double_dqn" and getattr(args, "dueling", None) is None:
            expected_dueling = saved_dueling
        if (bool(config_data.get("double_dqn", False)) != expected_double
                or saved_dueling != expected_dueling
                or (payload.get("agent_type") == "ra_d3qn") != (variant == "ra_d3qn")):
            parser.error(f"Resume checkpoint is not compatible with model variant '{variant}'")
        extra = payload.get("extra", {})
        if not all(key in extra for key in ("episode", "global_step", "history")):
            parser.error("Checkpoint lacks episode/step/history metadata required for resume")
        agent = (agent_loader or DQNAgent.load)(source, device=args.device)
        saved_training = extra.get("training_config", {})
        if "training_state" not in extra:
            print("Warning: legacy checkpoint has no replay/RNG state; replay warm-up restarts.")
        training = TrainingConfig(**saved_training) if saved_training else TrainingConfig(
            max_steps_per_episode=args.max_steps_per_episode, seed=args.seed,
            epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
            epsilon_decay_steps=args.epsilon_decay_steps,
        )
        training.episodes = args.episodes
        training.checkpoint_path = str(args.checkpoint)
        training.checkpoint_every = args.checkpoint_every
        try:
            training.validate()
        except ValueError as error:
            parser.error(str(error))
        print(f"Resuming {source} after episode {extra['episode']}, step {extra['global_step']}. "
              "Saved model, risk, and available training settings take precedence; --episodes, --device, "
              "--checkpoint and --checkpoint-every use the supplied values.")
        train_fn(training, agent.config, agent=agent, resume=extra, resumable=True)
        return

    config = DQNConfig(
        batch_size=args.batch_size, min_replay_size=args.min_replay_size,
        replay_capacity=args.replay_capacity, learning_rate=args.learning_rate,
        gamma=args.gamma, target_tau=args.target_tau,
        target_update_interval=args.target_update_interval,
        reward_scale=args.reward_scale, gradient_clip_norm=args.gradient_clip_norm,
        max_exponent=args.max_exponent, hidden_dims=tuple(args.hidden_dims),
        seed=args.seed, device=args.device,
        double_dqn=expected_double,
        dueling=(args.dueling if variant == "double_dqn" and args.dueling is not None else expected_dueling),
    )
    if args.hard_target_updates:
        config.target_tau = None
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    if variant == "ra_d3qn":
        agent = RiskAwareDQNAgent(config, risk)
    elif variant == "dueling_double_dqn":
        agent = DoubleDuelingDQNAgent(config)
    elif agent_factory is not None:
        agent = agent_factory(config)
    elif expected_double:
        agent = DoubleDQNAgent(config)
    else:
        agent = DQNAgent(config)
    training = TrainingConfig(
        episodes=args.episodes, max_steps_per_episode=args.max_steps_per_episode,
        checkpoint_path=str(args.checkpoint), checkpoint_every=args.checkpoint_every,
        seed=args.seed, epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
    )
    try:
        training.validate()
    except ValueError as error:
        parser.error(str(error))
    train_fn(training, agent.config, agent=agent, resumable=True)


def parse_and_train(
    argv: list[str] | None = None, *, variant: str, checkpoint: Path | None = None,
    description: str | None = None,
    agent_factory: Callable[[DQNConfig], DQNAgent] | None = None,
    agent_loader: Callable[..., DQNAgent] | None = None,
    train_fn: Callable[..., list[dict[str, float]]] = train,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    add_training_arguments(parser, variant=variant, checkpoint=checkpoint)
    args = parser.parse_args(argv)
    args._parser = parser
    train_from_args(args, variant=variant, checkpoint=checkpoint, agent_factory=agent_factory,
                    agent_loader=agent_loader, train_fn=train_fn)
