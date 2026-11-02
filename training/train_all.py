"""Train all five 2048 agents sequentially with the shared trainer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from training.train_cli import DEFAULT_CHECKPOINTS, add_training_arguments, train_from_args


VARIANTS = (
    "dqn",
    "double_dqn",
    "dueling_dqn",
    "dueling_double_dqn",
    "ra_d3qn",
)


def _train_variant(variant: str, episodes: int, device: str | None) -> Path:
    """Train one fresh variant to a temporary checkpoint, then publish it."""
    checkpoint = DEFAULT_CHECKPOINTS[variant]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(f".{checkpoint.stem}.train_all.tmp{checkpoint.suffix}")
    temporary.unlink(missing_ok=True)

    parser = argparse.ArgumentParser(prog=f"training.train_all ({variant})")
    add_training_arguments(parser, variant=variant)
    args = parser.parse_args(["--episodes", str(episodes)])
    args._parser = parser
    args.checkpoint = temporary
    args.resume = None
    args.fresh = True
    args.seed = None
    args.device = device

    try:
        train_from_args(args, variant=variant)
        if not temporary.is_file():
            raise RuntimeError("trainer completed without writing a checkpoint")
        payload = torch.load(temporary, map_location="cpu", weights_only=False)
        saved_episode = payload.get("extra", {}).get("episode")
        if saved_episode != episodes:
            raise RuntimeError(
                f"checkpoint reports episode {saved_episode!r}; expected {episodes}"
            )
        os.replace(temporary, checkpoint)
        return checkpoint
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2_000,
                        help="Number of episodes to train each agent (default: 2000)")
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda, or cuda:0")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("episodes must be positive")

    for index, variant in enumerate(VARIANTS, start=1):
        print(f"[{index}/{len(VARIANTS)}] Starting {variant}: {args.episodes} episodes", flush=True)
        try:
            checkpoint = _train_variant(variant, args.episodes, args.device)
        except Exception as error:
            print(f"[{index}/{len(VARIANTS)}] FAILED {variant}: {error}", flush=True)
            raise
        print(f"[{index}/{len(VARIANTS)}] COMPLETED {variant}: {checkpoint}", flush=True)

    print(f"All {len(VARIANTS)} agents completed {args.episodes} episodes.", flush=True)


if __name__ == "__main__":
    main()
