"""Train the Dueling Double DQN baseline with the shared trainer."""

from pathlib import Path

from training.train_cli import parse_and_train


def main() -> None:
    parse_and_train(
        variant="dueling_double_dqn",
        checkpoint=Path("models/dueling_double_dqn_2048.pt"),
        description=__doc__,
    )


if __name__ == "__main__":
    main()
