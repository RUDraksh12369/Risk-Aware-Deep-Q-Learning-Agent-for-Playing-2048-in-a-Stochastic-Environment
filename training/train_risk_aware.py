"""Train or resume the policy-only Risk-Aware Dueling Double DQN."""

from pathlib import Path

from training.train_cli import parse_and_train


def main() -> None:
    parse_and_train(
        variant="ra_d3qn",
        checkpoint=Path("models/ra_d3qn_2048.pt"),
        description=__doc__,
    )


if __name__ == "__main__":
    main()
