"""Shared CLI defaults and checkpoint resume for the five model variants."""

import argparse
import sys

import pytest

pytest.importorskip("torch")

from training.train_cli import add_training_arguments

ENTRYPOINTS = {
    "dqn": "training.train",
    "double_dqn": "training.train_doubledqn",
    "dueling_dqn": "training.train_duelingdqn",
    "dueling_double_dqn": "training.train_double_dueling_dqn",
    "ra_d3qn": "training.train_risk_aware",
}


VARIANTS = (
    "dqn", "double_dqn", "dueling_dqn", "dueling_double_dqn", "ra_d3qn",
)
COMMON = (
    "episodes", "checkpoint", "seed", "device", "batch_size", "min_replay_size",
    "replay_capacity", "learning_rate", "gamma", "target_tau", "target_update_interval",
    "epsilon_start", "epsilon_end", "epsilon_decay_steps", "max_steps_per_episode",
    "checkpoint_every", "fresh", "resume",
)


def test_all_five_training_clis_share_common_defaults():
    parsed = []
    for variant in VARIANTS:
        parser = argparse.ArgumentParser()
        add_training_arguments(parser, variant=variant)
        parsed.append(vars(parser.parse_args([])))
    for key in COMMON:
        values = [entry[key] for entry in parsed]
        assert all(value == values[0] for value in values), (key, values)
    assert {entry["checkpoint"] for entry in parsed} == {None}


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_standardized_trainer_saves_and_resumes_its_variant(tmp_path, monkeypatch, variant):
    from importlib import import_module

    entrypoint = import_module(ENTRYPOINTS[variant])
    path = tmp_path / f"{variant}.pt"
    options = [
        "--checkpoint", str(path), "--device", "cpu", "--batch-size", "2",
        "--min-replay-size", "2", "--replay-capacity", "8",
        "--max-steps-per-episode", "4", "--epsilon-start", "0", "--epsilon-end", "0",
        "--episodes", "1", "--seed", "12", "--checkpoint-every", "1",
    ]
    if variant == "ra_d3qn":
        options += ["--risk-coefficient", "0.4", "--mobility-weight", "2"]
    monkeypatch.setattr(sys, "argv", [ENTRYPOINTS[variant], *options])
    entrypoint.main()
    import torch

    first = torch.load(path, map_location="cpu", weights_only=False)
    assert first["extra"]["episode"] == 1
    assert first["extra"]["global_step"] == 4
    config = first["config"]
    assert config["double_dqn"] == (variant in {"double_dqn", "dueling_double_dqn", "ra_d3qn"})
    assert config["dueling"] == (variant in {"dueling_dqn", "dueling_double_dqn", "ra_d3qn"})
    if variant == "ra_d3qn":
        assert first["risk_config"]["risk_coefficient"] == 0.4

    # Resume uses the checkpoint's model and risk settings. The episode count
    # remains a total target, and a rerun at the completed target is a no-op.
    monkeypatch.setattr(sys, "argv", [ENTRYPOINTS[variant], "--checkpoint", str(path),
                        "--episodes", "2", "--device", "cpu"])
    entrypoint.main()
    second = torch.load(path, map_location="cpu", weights_only=False)
    assert second["extra"]["episode"] == 2
    assert second["extra"]["global_step"] == 8
    if variant == "ra_d3qn":
        assert second["risk_config"]["risk_coefficient"] == 0.4

