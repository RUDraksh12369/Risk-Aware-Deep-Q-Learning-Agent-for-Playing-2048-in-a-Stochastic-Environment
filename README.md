# Risk-Aware Deep Q-Learning for 2048

This repository studies deep Q-learning for 2048, where random tile spawns make
long-horizon decisions uncertain. It contains a 2048 engine, a Gym-like
environment, five DQN variants, a separate Monte Carlo Tree Search (MCTS)
player, evaluation and comparison tools, and a Pygame demo. The implementation
is complete enough to run experiments; substantive multi-seed comparisons and
research conclusions remain future work.

## Project Overview

The agent observes a 4×4 board encoded as 16 log2 tile values, and chooses one
of four moves. The environment rewards score gain and reports legal actions.
Training, evaluation, inference, and game rendering are separate workflows.

## Research Objectives

- Compare DQN variants under matched training budgets and seeded evaluation.
- Measure board-level risk from mobility, crowding, and immediate danger after
  the next tile spawn.
- Study RA-D3QN, which penalizes action values by a configurable risk score.
- Use held-out seeds, ablations, and substantive multi-seed runs before drawing
  conclusions about playing strength.

## Algorithms Implemented

| Variant | Description |
| --- | --- |
| DQN | Standard value network and target-network learning. |
| Double DQN | Online network selects the next action; target network evaluates it. |
| Dueling DQN | Separates state value and action advantage estimates. |
| Dueling Double DQN | Combines the dueling head and Double DQN target. |
| RA-D3QN | Dueling Double DQN with risk-adjusted behavior action selection. |

The RA-D3QN risk penalty changes action selection during behavior and inference;
rewards and Bellman targets remain unchanged. MCTS is a separate planning demo,
not a sixth learned model or part of the training suite.

## Key Features

- Legal-action masking during learned action selection and bootstrapping.
- Resumable checkpoints that include training state; `--episodes` is a total
  target, including completed episodes in a resumed checkpoint.
- Shared seeded evaluation, episode exports, comparison manifests and CSVs,
  and descriptive plots and reports.
- Human, random, heuristic, MCTS, and supplied-agent GUI modes.

## Project Architecture

- `game/`: 2048 rules and the RL-facing environment.
- `rl/`: networks, agents, replay, risk calculations, and baseline policies.
- `training/`: trainers, checkpoint/evaluation utilities, comparisons, and
  analysis.
- `inference/`: checkpoint-backed policies and inference-only comparisons.
- `mcts/`: independent Monte Carlo Tree Search player.
- `gui/`: demo-only Pygame interface.
- `models/`: included model checkpoints and optional `.previous` backups.
- `results/`, `artifacts/`: experiment outputs and reports.
- `tests/`: unit and integration tests.

See [handbook.md](handbook.md) for algorithm details, checkpoint behavior,
risk definitions, evaluation metrics, and historical verification notes.

## Installation and Environment Setup

Run commands from the repository root. Python 3.10 or later is recommended.
Create and activate a virtual environment, then install the pinned-by-range
dependencies in `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`. For CUDA training,
use a PyTorch installation compatible with the installed CUDA runtime and GPU;
the trainer does not install or upgrade packages automatically.

## Training All Models

`training.train_all` trains the five variants sequentially, using their default
checkpoints. Existing checkpoints resume automatically. The default target is
2,000 episodes per model.

```powershell
python -m training.train_all --episodes 2000 --device cpu
```

Choose `--device cuda` for a CUDA-enabled PyTorch setup, or omit `--device` to
use the current profile's device selection. Supported profiles are `personal`,
`organization`, and `auto`; `organization` and `auto` select CUDA when
available, otherwise CPU. Add `--fresh-run` only to start a new experiment;
existing checkpoints are archived before replacement.

```powershell
python -m training.train_all --episodes 50000 --profile auto
python -m training.train_all --episodes 2000 --fresh-run --device cpu
```

## Training Individual Models

Each command below trains one variant. All support `--episodes`, `--checkpoint`,
`--seed`, `--device`, `--profile`, `--fresh`/`--fresh-run`, and `--resume`.
Common learning options include `--batch-size`, `--min-replay-size`,
`--learning-rate`, `--gamma`, `--epsilon-start`, `--epsilon-end`,
`--epsilon-decay-steps`, and `--max-steps-per-episode`. RA-D3QN also accepts
`--risk-coefficient`, `--mobility-weight`, `--crowding-weight`, and
`--game-over-weight`.

```powershell
python -m training.train --episodes 2000 --seed 42
python -m training.train_doubledqn --episodes 2000 --seed 42
python -m training.train_duelingdqn --episodes 2000 --seed 42
python -m training.train_double_dueling_dqn --episodes 2000 --seed 42
python -m training.train_risk_aware --episodes 2000 --seed 42 --risk-coefficient 0.1
```

By default, a trainer resumes its default checkpoint if present. `--resume PATH`
selects a source checkpoint; absent an explicit `--checkpoint`, it also remains
the output path. Saved model and training settings take precedence on resume.

## Running the GUI

The Pygame demo runs from the repository root. Launch human play with:

```powershell
python -m gui.pygame_demo --mode human
```

To demonstrate the trained models in sequence, run the block below. Each
checkpoint must already exist. The GUI identifies the selected model; quit a
demo with Q or Esc to start the next one.

```powershell
$env:DQN_CHECKPOINT = "models\dqn_baseline_2048.pt"
python -m gui.pygame_demo --mode agent --agent inference.autoplay:choose_action

$env:DQN_CHECKPOINT = "models\doubledqn_2048.pt"
python -m gui.pygame_demo --mode agent --agent inference.autoplay:choose_action

$env:DQN_CHECKPOINT = "models\dueling_dqn_shared.pt"
python -m gui.pygame_demo --mode agent --agent inference.autoplay:choose_action

$env:DQN_CHECKPOINT = "models\dueling_double_dqn_2048.pt"
python -m gui.pygame_demo --mode agent --agent inference.autoplay:choose_action

$env:DQN_CHECKPOINT = "models\ra_d3qn_2048.pt"
$env:DQN_INFERENCE_POLICY = "original"
python -m gui.pygame_demo --mode agent --agent inference.autoplay:choose_action
```

For RA-D3QN's inference-time optimized policy, follow the PowerShell example in
[how to run.txt](how%20to%20run.txt). It applies a configurable risk penalty and
heuristic tie-breaker without changing the checkpoint.

## Available Game Modes

The supported `--mode` choices are `human`, `random`, `heuristic`, `mcts`, and
`agent`. Human play uses arrow keys or WASD; R restarts and Q or Esc quits.
Autoplay modes also support Space to pause/resume, +/- to change speed, and R to
restart. Separate “Vs Bot” and “Pass and Play” modes are not implemented.

```powershell
python -m gui.pygame_demo --mode random --seed 42
python -m gui.pygame_demo --mode heuristic
python -m gui.pygame_demo --mode mcts --mcts-simulations 200 --mcts-rollout-depth 36
```

MCTS is a standalone gameplay policy and does not load a neural checkpoint.

## Running Trained Models

Evaluate a checkpoint with greedy, seeded episodes. The evaluator reports score,
maximum tile, episode length, termination/truncation, and board-risk metrics.
Use `--max-steps-per-episode` for a bounded evaluation; use `--seed` to set the
first environment seed.

```powershell
python -m training.evaluate --checkpoint models/ra_d3qn_2048.pt --episodes 100 --seed 1000
```

To play through the GUI, set `DQN_CHECKPOINT` to any supported checkpoint and
run the agent command in “Running the GUI.” The inference loader selects
`models/ra_d3qn_2048.pt` by default when it exists; set the environment variable
to make model selection explicit. GUI inference does not update model weights.

## Model Checkpoints

The five standard trainer defaults are:

| Variant | Default checkpoint |
| --- | --- |
| DQN | `models/dqn_baseline_2048.pt` |
| Double DQN | `models/doubledqn_2048.pt` |
| Dueling DQN | `models/dueling_dqn_shared.pt` |
| Dueling Double DQN | `models/dueling_double_dqn_2048.pt` |
| RA-D3QN | `models/ra_d3qn_2048.pt` |

All five primary checkpoint files are present in this checkout. Four also have
`.pt.previous` fallback files; DQN does not. Checkpoints are binary model data,
not documentation or source, and are not tracked as regular Git files here.
Training can create additional checkpoints and archive older ones under
`models/archive/` when explicitly started fresh.

## Experiments and Evaluation

`training.compare` runs matched training and seeded evaluations for the five
learned variants and the `random` and `greedy` references. Its output directory
must be new. `--smoke` uses tiny budgets to check the pipeline; it is not a
performance experiment.

```powershell
python -m training.compare --smoke --seeds 42 43 --output-dir results/comparison_smoke_new
python -m training.analyze --input-dir results/comparison_smoke_new --smoothing-window 25
python -m training.evaluate --checkpoint models/ra_d3qn_2048.pt --episodes 100 --episodes-csv results/ra_evaluation.csv
```

Comparison data is written beneath the chosen output directory, including
`manifest.json`, `results.csv`, `summary.csv`, and per-run metrics, histories,
checkpoints, and `evaluation.csv` exports. `training.analyze` reads saved output
without further training or game evaluation and writes a descriptive report and
plots. The repository currently contains comparison outputs under
`results/`; Dueling transfer/training artifacts are under `artifacts/dueling_dqn/`.

## Results and Research Findings

The checked-in `results/comparison_smoke/analysis/report.md` records a
two-seed, one-episode-per-agent smoke comparison with evaluations capped at 32
moves. All evaluated episodes were truncated by that cap. The observed mean
scores were random 194, greedy 204, DQN 169, Double DQN 175, Dueling DQN 144,
Dueling Double DQN 144, and RA-D3QN 146. These are pipeline-check observations,
not complete-game outcomes, convergence evidence, or a reliable model ranking.
The report includes sample standard deviations and other descriptive metrics.

The handbook also preserves historical test and smoke-run records. They describe
earlier verification snapshots, not tests run as part of this documentation
update. Multi-seed training, ablations, risk sensitivity, and robustness studies
remain necessary before claiming that risk-aware action selection improves play.

## Project Structure

```text
game/       game rules and RL environment
rl/         DQN agents, replay, risk scoring, and reference policies
training/   training, evaluation, comparison, and analysis CLIs
inference/  checkpoint inference policies and policy studies
mcts/       Monte Carlo Tree Search agent
gui/        Pygame demonstration
models/     model checkpoints and fallback backups
results/    experiment outputs and generated analysis
artifacts/  Dueling transfer and training records
tests/      unit and integration tests
```

## Testing

Run the test suite from the repository root after installing dependencies:

```powershell
python -m pytest -q
```

The most recent verification record in the handbook is dated 2026-11-01 and
reports 180 passed tests plus 63 subtests. This documentation task did not run
tests or training.

## Limitations

- The recorded comparison is a short, truncated smoke run; substantive
  multi-seed performance results are not available in that report.
- RA-D3QN changes the behavior policy only. Its Q targets still estimate ordinary
  score return, and its coefficient and risk weights have not been established
  as optimal.
- MCTS is a demo baseline, not an optimal solver or a trained model.
- The analysis reports descriptive statistics, not significance tests or
  automatic model rankings.

## Contributors and Acknowledgements

The demo-only Pygame interface adapts visual layout and event-loop ideas from
[pygame-2048 by Arbel Amram](https://github.com/arbelamram/pygame-2048). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the license notice.
