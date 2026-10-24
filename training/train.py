"""Command-line trainer for the standard masked Double-DQN 2048 baseline."""

from __future__ import annotations

import argparse
import random
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from game.environment import Env2048
from rl.agent import DQNAgent, DQNConfig


@dataclass
class TrainingConfig:
    episodes: int = 2_000
    max_steps_per_episode: int = 5_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 300_000
    checkpoint_path: str = "models/dqn_2048.pt"
    checkpoint_every: int = 250
    seed: int | None = 42
    illegal_move_penalty: float = 0.0
    game_over_penalty: float = 0.0


def epsilon_by_step(step: int, config: TrainingConfig) -> float:
    progress = min(step / max(config.epsilon_decay_steps, 1), 1.0)
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def train(
    training: TrainingConfig, dqn: DQNConfig, *, agent: DQNAgent | None = None,
    resume: dict | None = None, resumable: bool = False
) -> list[dict[str, float]]:
    """Train a new or supplied agent and return metrics for this run.

    A supplied agent retains its weights, optimizer, replay, and update count.
    The episode history and exploration schedule start fresh unless resume is
    provided. Resumable checkpoints capture completed episode boundaries.
    """
    if agent is not None and agent.config != dqn:
        raise ValueError("dqn must match the supplied agent's config.")
    if training.seed is not None:
        random.seed(training.seed)
        np.random.seed(training.seed)
        torch.manual_seed(training.seed)
    env = Env2048(
        seed=training.seed,
        illegal_move_penalty=training.illegal_move_penalty,
        game_over_penalty=training.game_over_penalty,
    )
    if agent is None:
        agent = DQNAgent(dqn)
    global_step = 0
    history: list[dict[str, float]] = []
    checkpoint = Path(training.checkpoint_path)

    start_episode = 0
    if resume is not None:
        start_episode = int(resume["episode"])
        global_step = int(resume["global_step"])
        history = list(resume["history"])
        saved = resume.get("training_state")
        if saved is not None:
            agent.replay = saved["replay"]
            agent._rng.setstate(saved["policy_rng"])
            env.game._rng.setstate(saved["environment_rng"])
            random.setstate(saved["python_rng"])
            np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["torch_rng"].cpu())
            if agent.device.type == "cuda" and saved["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        if training.episodes <= start_episode:
            print(f"Checkpoint already completed {start_episode} episodes; nothing to train.")
            return history

    def save_progress(episode: int) -> None:
        extra = {"episode": episode, "global_step": global_step, "history": history}
        if not resumable:
            agent.save(checkpoint, extra)
            return
        extra["training_config"] = asdict(training)
        extra["training_state"] = {
            "replay": agent.replay, "policy_rng": agent._rng.getstate(),
            "environment_rng": env.game._rng.getstate(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if agent.device.type == "cuda" else None,
        }
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        # Keep the last complete checkpoint intact if writing is interrupted.
        with tempfile.NamedTemporaryFile(dir=checkpoint.parent, suffix=".pt", delete=False) as file:
            temporary = Path(file.name)
        try:
            agent.save(temporary, extra)
            with temporary.open("rb") as file:
                os.fsync(file.fileno())
            os.replace(temporary, checkpoint)
        finally:
            temporary.unlink(missing_ok=True)

    if resumable and resume is None:
        save_progress(0)

    for episode in range(start_episode + 1, training.episodes + 1):
        state = env.reset()
        total_reward = 0.0
        loss_values: list[float] = []
        steps_taken = 0
        for _ in range(training.max_steps_per_episode):
            legal_mask = env.legal_action_mask()
            if not legal_mask.any():
                break
            epsilon = epsilon_by_step(global_step, training)
            action = agent.select_action(state, legal_mask, epsilon, game=env.game)
            next_state, reward, done, _ = env.step(action)
            next_mask = np.zeros(env.n_actions, dtype=bool) if done else env.legal_action_mask()
            agent.observe(state, action, reward, next_state, done, next_mask)
            update = agent.learn()
            if update is not None:
                loss_values.append(update["loss"])
            state = next_state
            total_reward += reward
            global_step += 1
            steps_taken += 1
            if done:
                break

        record = {
            "episode": float(episode), "steps": float(steps_taken), "score": float(env.score),
            "max_tile": float(env.game.max_tile()), "reward": total_reward,
            "epsilon": epsilon_by_step(global_step, training),
            "loss": float(np.mean(loss_values)) if loss_values else float("nan"),
        }
        history.append(record)
        if episode == 1 or episode % 25 == 0:
            print("episode={episode:.0f} score={score:.0f} max_tile={max_tile:.0f} "
                  "steps={steps:.0f} epsilon={epsilon:.3f} loss={loss:.4f}".format(**record))
        if episode % training.checkpoint_every == 0:
            save_progress(episode)

    save_progress(training.episodes)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes)
    parser.add_argument("--checkpoint", default=TrainingConfig.checkpoint_path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda, or cuda:0")
    parser.add_argument("--batch-size", type=int, default=DQNConfig.batch_size)
    parser.add_argument("--min-replay-size", type=int, default=DQNConfig.min_replay_size)
    args = parser.parse_args()
    train(
        TrainingConfig(episodes=args.episodes, checkpoint_path=args.checkpoint, seed=args.seed),
        DQNConfig(batch_size=args.batch_size, min_replay_size=args.min_replay_size, seed=args.seed, device=args.device),
    )


if __name__ == "__main__":
    main()
