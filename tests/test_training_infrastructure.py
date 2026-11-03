"""Persistence and hardware-profile behavior for shared training commands."""

import argparse

import pytest

torch = pytest.importorskip("torch")


def test_hardware_profile_supports_cpu_fallback(monkeypatch):
    from training import hardware

    monkeypatch.setattr(hardware, "detect_hardware", lambda: hardware.HardwareInfo(
        os="Windows test", cpu="test CPU", logical_processors=8, ram_gb=16,
        cuda_available=False, gpu=None, gpu_memory_gb=None, compute_capability=None,
        torch_version="test", cuda_version=None,
    ))
    resolved, info = hardware.resolve_profile("organization")
    assert resolved == "organization:cpu"
    assert info.gpu is None
    with pytest.raises(ValueError, match="CUDA was requested"):
        hardware.resolve_profile("auto", "cuda")


def test_train_all_does_not_force_fresh_and_forwards_resume_configuration(monkeypatch, tmp_path):
    from training import train_all

    checkpoints = {name: tmp_path / f"{name}.pt" for name in train_all.VARIANTS}
    monkeypatch.setattr(train_all, "DEFAULT_CHECKPOINTS", checkpoints)
    calls = []
    monkeypatch.setattr(train_all, "train_from_args", lambda args, *, variant: calls.append((variant, args)))
    monkeypatch.setattr(train_all, "update_model", lambda *a, **k: None)
    train_all.main(["--episodes", "7", "--profile", "auto"])
    assert [variant for variant, _ in calls] == list(train_all.VARIANTS)
    assert all(args.fresh is False and args.episodes == 7 and args.profile == "auto"
               for _, args in calls)


def test_default_cli_refuses_corrupt_checkpoint_instead_of_reinitializing(tmp_path):
    from training.train_cli import parse_and_train

    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"partial checkpoint")
    with pytest.raises(SystemExit):
        parse_and_train(["--checkpoint", str(checkpoint), "--device", "cpu"], variant="dqn")
    assert checkpoint.read_bytes() == b"partial checkpoint"


def test_fresh_run_archives_existing_checkpoint(tmp_path, monkeypatch):
    from training.train_cli import parse_and_train

    checkpoint = tmp_path / "existing.pt"
    checkpoint.write_bytes(b"old checkpoint")

    def noop_train(training, config, **kwargs):
        return []

    monkeypatch.chdir(tmp_path)
    parse_and_train(["--checkpoint", str(checkpoint), "--fresh-run", "--episodes", "1",
                     "--max-steps-per-episode", "1", "--device", "cpu"],
                    variant="dqn", train_fn=noop_train)
    archived = list((tmp_path / "archive").glob("existing.*.pt"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == b"old checkpoint"

