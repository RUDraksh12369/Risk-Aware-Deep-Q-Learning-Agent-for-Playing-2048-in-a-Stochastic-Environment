"""Train the Dueling DQN research baseline with the shared trainer."""

from pathlib import Path

from training.train_cli import parse_and_train


def main() -> None:
    parse_and_train(
        variant="dueling_dqn",
        checkpoint=Path("models/dueling_dqn_shared.pt"),
        description=__doc__,
    )


if __name__ == "__main__":
    main()
