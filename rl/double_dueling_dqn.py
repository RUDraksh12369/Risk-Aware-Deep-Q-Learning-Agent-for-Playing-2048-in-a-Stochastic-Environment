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
from rl.replay_buffer import ReplayBuffer
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
    training.add_argument("--hard-target-updates", action="store_true")
    training.add_argument("--target-update-interval", type=int, default=DQNConfig.target_update_interval)
    training.add_argument("--reward-scale", type=float, default=DQNConfig.reward_scale)
    training.add_argument("--gradient-clip-norm", type=float, default=DQNConfig.gradient_clip_norm)
    training.add_argument("--max-exponent", type=float, default=DQNConfig.max_exponent)
    training.add_argument("--hidden-dims", type=int, nargs="+", default=DQNConfig.hidden_dims)
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
        if args.episodes < 1 or (args.max_steps_per_episode is not None and args.max_steps_per_episode < 1):
            parser.error("episodes and max-steps-per-episode must be positive")
        agent = DoubleDuelingDQNAgent.load(args.checkpoint, args.device)
        metrics = evaluate(agent, args.episodes, args.seed, max_steps_per_episode=args.max_steps_per_episode)
        for name, value in metrics.items():
            print(f"{name}: {value:.3f}")
        return

    # Refuse to overwrite any pre-existing output: resuming writes a new path.
    if args.checkpoint.exists():
        parser.error("Output checkpoint already exists; choose a new --checkpoint path")
    resume = None
    if args.resume:
        if not args.resume.is_file():
            parser.error(f"Resume checkpoint not found: {args.resume}")
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != "double_dueling_dqn_v1":
            parser.error("--resume requires a Double Dueling DQN checkpoint")
        resume = payload.get("extra", {})
        if not isinstance(resume, dict) or not all(key in resume for key in ("episode", "global_step", "history", "training_config", "training_state")):
            parser.error("Checkpoint lacks resumable training metadata")
        for key in ("episode", "global_step"):
            value = resume[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                parser.error(f"Checkpoint {key} must be a nonnegative integer")
        if not isinstance(resume["history"], list):
            parser.error("Checkpoint history must be a list")
        state = resume["training_state"]
        state_keys = {"replay", "policy_rng", "environment_rng", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"}
        if not isinstance(state, dict) or not state_keys.issubset(state) or not isinstance(state["replay"], ReplayBuffer):
            parser.error("Checkpoint lacks replay/RNG state required for resume")
        try:
            config = TrainingConfig(**resume["training_config"])
        except (TypeError, ValueError) as error:
            parser.error(f"Invalid saved training configuration: {error}")
        config.episodes = args.episodes
        config.checkpoint_path = str(args.checkpoint)
        config.checkpoint_every = args.checkpoint_every
        try:
            config.validate()
        except ValueError as error:
            parser.error(str(error))
        if args.episodes <= resume["episode"]:
            print(f"Checkpoint already completed {resume['episode']} episodes; nothing to train.")
            return
        agent = DoubleDuelingDQNAgent.load(args.resume, args.device)
        if state["replay"].capacity != agent.config.replay_capacity:
            parser.error("Saved replay capacity does not match model configuration")
    else:
        model_config = DQNConfig(
            batch_size=args.batch_size, min_replay_size=args.min_replay_size,
            replay_capacity=args.replay_capacity, learning_rate=args.learning_rate,
            gamma=args.gamma, target_tau=None if args.hard_target_updates else args.target_tau,
            target_update_interval=args.target_update_interval, reward_scale=args.reward_scale,
            gradient_clip_norm=args.gradient_clip_norm, max_exponent=args.max_exponent,
            hidden_dims=tuple(args.hidden_dims), seed=args.seed, device=args.device,
        )
        config = TrainingConfig(
            episodes=args.episodes, max_steps_per_episode=args.max_steps_per_episode,
            checkpoint_path=str(args.checkpoint), checkpoint_every=args.checkpoint_every,
            seed=args.seed, epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
            epsilon_decay_steps=args.epsilon_decay_steps,
        )
        try:
            model_config.validate()
            config.validate()
        except ValueError as error:
            parser.error(str(error))
        agent = DoubleDuelingDQNAgent(model_config)
    train(config, agent.config, agent=agent, resume=resume, resumable=True)


if __name__ == "__main__":
    main()
