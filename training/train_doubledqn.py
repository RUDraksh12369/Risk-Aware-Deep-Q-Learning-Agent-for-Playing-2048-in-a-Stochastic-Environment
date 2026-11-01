"""Train or resume Double DQN using the existing DQN architecture."""

from __future__ import annotations

from pathlib import Path

from rl.doubledqn import DoubleDQNAgent
from training.train import train
from training.train_cli import parse_and_train


def main() -> None:
    parse_and_train(
        variant="double_dqn",
        checkpoint=Path("models/doubledqn_2048.pt"),
        description=__doc__,
        agent_factory=DoubleDQNAgent,
        agent_loader=lambda *args, **kwargs: DoubleDQNAgent.load(*args, **kwargs),
        train_fn=train,
    )


if __name__ == "__main__":
    main()
