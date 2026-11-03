"""Resume or sequentially train every supported 2048 model variant."""

from __future__ import annotations

import argparse

from training.train_cli import DEFAULT_CHECKPOINTS, add_training_arguments, train_from_args
from training.registry import update_model


# Existing project order is retained.
VARIANTS = ("dqn", "double_dqn", "dueling_dqn", "dueling_double_dqn", "ra_d3qn")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2_000,
                        help="Total target episodes per model (default: 2000)")
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda, or cuda:0")
    parser.add_argument("--profile", choices=("personal", "organization", "auto"), default="personal")
    parser.add_argument("--fresh-run", action="store_true",
                        help="Explicitly archive prior checkpoints and start a new experiment")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("episodes must be positive")

    for index, variant in enumerate(VARIANTS, start=1):
        checkpoint = DEFAULT_CHECKPOINTS[variant]
        print(f"[{index}/{len(VARIANTS)}] Resume-or-continue {variant}: target={args.episodes}", flush=True)
        variant_parser = argparse.ArgumentParser(prog=f"training.train_all ({variant})")
        add_training_arguments(variant_parser, variant=variant)
        cli = ["--episodes", str(args.episodes), "--checkpoint", str(checkpoint),
               "--profile", args.profile]
        if args.device:
            cli += ["--device", args.device]
        if args.fresh_run:
            cli.append("--fresh-run")
        variant_args = variant_parser.parse_args(cli)
        variant_args._parser = variant_parser
        try:
            train_from_args(variant_args, variant=variant)
        except Exception as error:
            update_model(variant, status="failed", total_episodes=args.episodes,
                         latest_checkpoint=str(checkpoint), error=str(error))
            print(f"[{index}/{len(VARIANTS)}] FAILED {variant}: {error}", flush=True)
            raise
        update_model(variant, status="completed", completed_episodes=args.episodes,
                     total_episodes=args.episodes, latest_checkpoint=str(checkpoint),
                     completion_status="complete")
        print(f"[{index}/{len(VARIANTS)}] COMPLETE {variant}: {checkpoint}", flush=True)

    print(f"All {len(VARIANTS)} agents reached the {args.episodes}-episode target.", flush=True)


if __name__ == "__main__":
    main()
