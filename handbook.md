# Project Handbook - 2048 Risk-Aware RL

## Purpose and current position

This repository is building a research-style 2048 agent for studying
long-horizon decisions under stochastic tile spawns. The foundation, planning
baseline, DQN baseline, training workflow, inference adapter, and presentation
GUI are implemented. `rl/risk.py` now provides board/afterstate risk and the
proposal's `Q_RA = Q - lambda * R` action-evaluation utility. The first,
policy-only RA-D3QN variant is integrated into training, evaluation, and
checkpoint/GUI inference. A multi-seed comparison runner now covers the five
learned variants plus random and immediate-score greedy reference policies.
Full comparative training/experiments and the final report are still pending.

In short: the project has a working and tested **standard DQN baseline** and
an **MCTS comparison/demo agent**; it is ready for proper training runs and
the risk-aware research phase.

## What has been completed

| Area | Status | What is present |
| --- | --- | --- |
| 2048 rules engine | Complete | Tile spawning, directional movement, merges, score tracking, terminal detection, cloning, move simulation, and legal-action discovery. |
| RL environment | Complete | Gym-like `reset`/`step`, log2 state encoding, rewards, legal-action masks, and non-mutating afterstate simulation. |
| Tests | Implemented features covered | Game rules, environment behaviour, MCTS, replay, DQN/Double DQN targets, checkpoint compatibility, and risk scoring have dedicated tests. See verification below for run results. |
| MCTS baseline/demo | Complete | Stochastic rollouts, UCB action selection, simulated tile spawns, and board heuristics. |
| DQN baseline | Complete | Dueling MLP, Double-DQN targets, replay buffer, target network, reward scaling, gradient clipping, checkpoint save/load, and legal-action masking. |
| Risk evaluation | Implemented; experimental design unvalidated | Configurable mobility, crowding, and immediate spawn-danger score; action-specific afterstates; masked Q-minus-risk utility. |
| RA-D3QN integration | Implemented; performance unvalidated | Risk-aware behaviour policy, unchanged Double DQN targets, separate trainer, saved risk settings, and automatic policy restoration. |
| Comparison runner | Implemented; full experiments pending | Explicit baseline flags, random/greedy references, paired evaluation seed ranges, isolated checkpoints/histories, manifests and CSV summaries. |
| Analysis tooling | Implemented; substantive results pending | Per-episode evaluation exports, per-seed training curves, evaluation/episode plots, and a descriptive Markdown report. |
| Training and evaluation | Complete | CLI trainer with epsilon schedule/checkpoints and deterministic greedy batch evaluation. |
| Inference and GUI | Complete | Checkpoint-loading action callable and Pygame modes for human, random, heuristic, MCTS, and supplied-agent play. |
| Saved artefacts | Present | A trained checkpoint is included in `models/dqn_2048.pt`; the merged checkpoint records 1,672,322 learning updates. |

## How the implemented system works

### 1. Game and environment

`game/game.py` is the source of truth for 2048 mechanics. A move compresses
and merges tiles once per row/column, adds the merge value to the score, and
then spawns a random tile (90% 2, 10% 4 by default). It also exposes
`simulate_move`, which permits agents to inspect a move without changing the
live game.

`game/environment.py` keeps RL concerns separate from game rules. It converts
the 4x4 board into 16 float32 log2 tile exponents (`0 -> 0`, `2 -> 1`, `4 -> 2`,
and so on), returns score gain as the default reward, and exposes a boolean
legal-action mask. Optional illegal-move, game-over, and custom reward hooks
are already available for later experiments.

### 2. Planning baseline

`mcts/agent.py` implements an independent Monte Carlo Tree Search player. For
each legal move it simulates the random spawn, uses UCB to balance exploration
and exploitation, performs bounded rollouts, and backs up accumulated return.
Rollouts prefer immediate gains, empty cells, corner placement of the maximum
tile, and monotonic rows/columns. It is a useful playable baseline, but it is
not a risk-aware algorithm or an optimal 2048 solver.

### 3. DQN baseline

The learning agent is split into three parts:

- `rl/dqn.py` defines a 16-input, two-hidden-layer (256/256) MLP. It
  normalises the log2 encoding and uses a dueling value/advantage head.
- `rl/replay_buffer.py` stores fixed-capacity transitions and samples random
  mini-batches to decorrelate updates.
- `rl/agent.py` performs epsilon-greedy action selection, never chooses a
  masked/illegal move, trains with Smooth L1 loss, clips gradients, and keeps a
  softly updated target network.

The update is Double DQN: the online network chooses the next legal action and
the target network evaluates it. Terminal or no-legal-action transitions do
not bootstrap. Raw game rewards are multiplied by `0.01` only when written to
replay, keeping reported scores in normal 2048 units while improving Q-target
scale.

`rl/doubledqn.py` exposes a dedicated `DoubleDQNAgent` and the Double DQN
next-action calculation. `training/train_doubledqn.py` starts fresh or resumes its checkpoint
and defaults to a separate `models/doubledqn_2048.pt` output. Dueling is enabled
by default, so this is a Dueling Double DQN unless configured otherwise.
Use `--no-dueling` for plain Double DQN. The dedicated CLI exposes learning
rate, gamma, replay capacity, soft target tau, epsilon schedule, and checkpoint
frequency, and rejects invalid settings before constructing the agent.

Training-ready command (from the repository root):

```bash
python3 -m training.train_doubledqn --no-dueling --episodes 2000 --seed 42 --checkpoint models/doubledqn_2048.pt
python3 -m training.evaluate --checkpoint models/doubledqn_2048.pt --episodes 100
```

Double DQN now automatically resumes an existing output checkpoint. Its default
save interval is one completed episode; writes replace the previous checkpoint
atomically. Replay, policy/environment/Python/NumPy/Torch RNG states, history,
episode/global-step counters and training settings are included alongside model
and optimizer state. `--episodes` means the total target; rerunning the same
command continues toward that target. An interrupted partial episode is replayed
from the last saved boundary, not resumed mid-step. `--fresh` deliberately starts
over; `--resume PATH` requires that source and also saves there unless another
output is specified. Saved settings take precedence except for total episodes,
device, output path and save interval. Legacy checkpoints resume weights and
counters with a warning: missing replay/RNG cannot be recovered and exploration
settings come from the CLI. Other trainers retain their existing save behaviour.
No additional dependencies are needed beyond `requirements.txt`.

Training-readiness verification: 30 dedicated tests passed in 1.78s; the full
suite passed 144 tests in 30.99s. A fresh
CPU plain-Double-DQN smoke run completed 32 steps and 29 learning updates,
reloaded with `double_dqn=True, dueling=False`, and evaluated two episodes
capped at 32 steps (mean score 152). This is a runtime check, not a performance
benchmark. The checkpoint was written to a temporary directory.

The merged standalone trainer `rl/duelingDQN.py` uses its own
`dueling_dqn_v1` checkpoint format. The shared loader now recognizes that
format, restores dueling heads and the saved input normalization, and supports
these checkpoints in `training.evaluate` and GUI inference. Use the standalone
trainer for its replay-based continuation; the shared loader restores model and
optimizer state, not its saved replay buffer. The included DQN and standalone
Dueling checkpoints both have Double DQN enabled, so their filenames alone do
not identify plain-DQN versus Double-DQN experimental baselines.

### 3a. Risk evaluation foundation

The proposal calls for action/value adjustment while preserving score rewards.
`rl/risk.py` now provides `RiskConfig`, `board_risk`, `action_risks`, and
`risk_adjusted_q_values`. For a legal action's deterministic afterstate:

```text
mobility risk = 1 - number of legal moves / 4
crowding risk = 1 - number of empty cells / number of board cells
spawn danger = P(no legal moves immediately after the next random tile spawn)
R(s, a) = weighted mean(mobility risk, crowding risk, spawn danger)
Q_RA(s, a) = Q(s, a) - lambda * R(s, a)
```

The first two features describe the board before spawning. Spawn danger is
computed exactly for the next spawn: with one empty cell the two possible tile
values are checked; at least two empty cells guarantee room remains. A full
board retains its current terminal status. The implementation reuses the game
rules without changing the live board, score, move count, or RNG state.

Equal weights are a provisional design choice because the proposal does not
specify them. The composite is a bounded heuristic, not a calibrated long-term
failure probability. Lambda uses the network's Q-value units, including reward
scaling. Illegal moves are excluded even at lambda zero.

### 3b. Policy-only RA-D3QN

`rl/risk_aware.py` provides `RiskAwareConfig` and `RiskAwareDQNAgent`, reusing
the Dueling network and Double DQN learning loop. Exploitation chooses the
legal action maximizing `Q - lambda * R`; epsilon exploration remains uniform
over legal actions. The raw board is supplied explicitly, so risk assessment
does not attempt to reconstruct potentially clipped tiles from the encoding.

This initial experimental variant changes **only the behaviour policy**.
Rewards and Double DQN targets are intentionally unchanged: next actions in
the Bellman target maximize ordinary online Q and are evaluated by target Q.
Thus Q still estimates ordinary score return, not risk-adjusted return. This
choice isolates the effect of policy adjustment and makes lambda zero an exact
baseline ablation; it does not establish optimality for a risk objective.

`training/train_risk_aware.py` starts fresh and saves to
`models/ra_d3qn_2048.pt` by default. Risk coefficient (default 0.1) and component
weights are saved alongside the existing model/optimizer state. The generic
`DQNAgent.load` restores the risk-aware class when appropriate; old checkpoints
remain baseline agents. The evaluator and GUI use this same restoration path.

### 4. Running, saving, and evaluating

`training/train.py` creates an environment and agent, linearly decays epsilon
from 1.0 to 0.05 over 300,000 steps, records per-episode score/tile/loss data,
and saves checkpoints every 250 episodes plus at the end. Defaults include a
2,000-transition warm-up, batch size 128, gamma 0.995, and soft target update
tau 0.005.

`training/evaluate.py` loads a checkpoint and plays greedy episodes with fresh,
seeded environments. It reports mean/median/best score, mean maximum tile, and
rates of reaching tiles 128, 256, 512, and 2048. It also reports episode length,
maximum tile, termination versus time-limit truncation, mean legal moves,
empty cells, board risk, and low-mobility/high-risk state rates.
`inference/autoplay.py` provides the same loaded policy to the GUI.

Risk metrics pool nonterminal decision states, using equal component weights
for every policy. Low mobility means <=1 legal move; high risk means composite
score >=0.5 by default. `--high-risk-threshold` changes that threshold and
`--max-steps-per-episode` optionally caps evaluation. Without a cap, complete
games naturally have game-over rate 1; comparative conclusions must also use
episode length, score, and risk metrics.

### 5. Controlled comparison runner

`training/compare.py` constructs fresh DQN (`double_dqn=False, dueling=False`),
Double DQN (`True, False`), Dueling DQN (`False, True`), Dueling Double DQN
(`True, True`), and RA-D3QN (`True, True` plus risk policy) agents. It explicitly
overrides enhanced defaults to avoid mislabeled baselines. `rl/baselines.py`
adds uniform-legal random and immediate-merge-score greedy policies; the latter
breaks ties by action id and does not use the GUI heuristic.

All runs share training settings and evaluation environment seed ranges;
training/policy seeds cannot overlap the evaluation seed range. Learned models
are initialized fresh, trained using the existing loop, and reloaded from
their saved checkpoint before evaluation. Episode/step budgets are matched,
but actual steps/updates vary with survival and are recorded. Dueling heads
change parameter counts despite matching hidden widths; counts are recorded.

The output folder must not already exist. `manifest.json` records settings,
versions, and completion/failure status. Each `<agent>/seed_<seed>/` stores
metrics, and learned runs also store checkpoints and training history CSVs.
`results.csv` holds per-run results; `summary.csv` holds means and sample
standard deviations across run metrics. Single-run std and warm-up loss are
blank, not zero. Partial completed results survive a subsequent run failure.

Random seeds govern its own policy RNG. Greedy repeats across seeds are
deterministic repeats, not independent training samples. Summary std is not
a confidence interval. `--smoke` uses CPU, 1x32 training, 2x32 evaluation,
batch/warmup 4, and epsilon zero; these checks are not research results.

### 6. Per-episode exports and descriptive analysis

The evaluator optionally collects individual episode records without changing
its aggregate return or policy randomness. Comparison runs always save these
as `evaluation.csv` and reference them in `results.csv`. Standalone evaluation
supports `--episodes-csv` and refuses to overwrite an existing export. Records
include seed, score, max tile, steps, termination/truncation, decision-state
counts, and board-risk measurements. Aggregate risk metrics remain weighted
by decision-state counts across episodes.

`training/analyze.py` reads saved comparison files without training/evaluation
and writes plots plus `report.md` in a new analysis directory. Training plots
show individual runs with trailing-episode smoothing; missing warm-up losses
remain gaps. Evaluation bars summarize run means with sample-standard-deviation
whiskers; individual evaluation scores are pooled descriptively in a separate
plot, not treated as independent training replicates. The report explicitly
flags partial comparisons, truncated games, unavailable legacy episode exports,
and single-run standard deviation. It does not declare significance or rank
trained agents automatically. Source data and checkpoints are not modified.

## Verification completed

After adding the comparison runner, `python3 -m pytest -q` passed **124 tests**
(3.10 seconds), including **24 comparison cases**. A temporary CPU CLI smoke
comparison completed **7 policies x 2 seeds = 14 runs**. The 10 learned runs
each performed 32 steps and 29 updates (**320 steps, 290 updates total**).
Each run evaluated two episodes capped at 32 steps. All 10 checkpoints
reloaded with the intended model flags; CSV/manifest outputs were checked and
existing model hashes were unchanged. Smoke artifacts were temporary; these
results validate the pipeline, not an agent performance ranking.

After RA-D3QN integration, `python3 -m pytest -q` passed **100 tests**
(3.20 seconds), including **16 integration cases**. A fresh CPU CLI smoke run
with epsilon zero and lambda 0.1 completed **32 steps and 29 updates**, with
mean loss **0.0010826458892552182**. Generic checkpoint loading restored the
risk policy/settings; two seeded evaluations capped at 128 steps per episode
were repeatable; the evaluation CLI and GUI inference adapter also passed.
Existing checkpoint files were verified unchanged. These bounded checks are
runtime verification, not evidence of comparative performance improvement.

After adding the risk-evaluation foundation, `python3 -m pytest -q` passed
**84 tests**, including **37 risk tests**. A bounded CPU integration smoke
test exercised risk-adjusted action selection with the existing environment,
replay, and Double DQN learning: **32 steps, 29 updates**, finite losses,
lambda-zero baseline equivalence, unchanged RNG state during risk evaluation,
and a successful temporary checkpoint save/load. This does not establish
playing-strength improvements or constitute a trained RA-D3QN model.

The original baseline workspace was checked with:

```powershell
pytest -q
```

Result: **25 passed**. A five-episode CPU smoke evaluation of the bundled
`models/dqn_2048.pt` also ran successfully (mean score 1112.8; 3/5 games
reached tile 128). This is a runtime sanity check only, not a statistically
reliable benchmark.

## Episode export and analysis verification (2026-10-21)

- Focused analysis and comparison tests: **34 passed in 33.50s**.
- Full suite (`python3 -m pytest -q`): **134 passed in 36.96s**.
- Generated `results/comparison_smoke/analysis/report.md` and three plots from
  the existing completed smoke comparison, without further training.
- Visually checked evaluation, training, and episode-score plots for readability.
- These runs use one training episode and a 32-step evaluation cap; they verify
  the pipeline and do not establish comparative performance or convergence.

## Remaining work

The following research milestones remain before the repository matches its
"risk-aware" title:

1. Run the implemented baseline comparison runner with substantive training
   budgets and multiple seeds; inspect convergence before drawing conclusions.
2. Evaluate the policy-only risk design and tune lambda/weights; any change to
   risk-adjusted Bellman target selection should be a separately tested ablation.
3. Use the implemented episode exports and descriptive reports to design
   appropriate statistical comparisons after substantive multi-seed training.
4. Run repeatable, multi-seed training for the standard and risk-aware agents;
   retain histories, checkpoints, and evaluation tables.
5. Perform ablations, lambda sensitivity, and robustness/generalisation experiments.
6. Produce plots, conclusions, and the final report/presentation. A CNN-DQN
   comparison is optional.

## Practical commands

```powershell
# Install dependencies in an activated virtual environment
pip install -r requirements.txt

# Validate the implementation
pytest -q

# Small comparison pipeline check (choose a new output directory each time)
python -m training.compare --smoke --seeds 42 43 --output-dir results/comparison_smoke

# Plot saved results without any further training (analysis directory must be new)
python -m training.analyze --input-dir results/comparison_smoke --smoothing-window 25

# Train a baseline checkpoint
python -m training.train --episodes 2000 --checkpoint models/dqn_2048_enhanced.pt

# Short RA-D3QN smoke run with risk-aware exploitation enabled
python -m training.train_risk_aware --episodes 1 --max-steps-per-episode 32 --batch-size 4 --min-replay-size 4 --epsilon-start 0 --epsilon-end 0 --risk-coefficient 0.1 --device cpu --checkpoint models/ra_d3qn_smoke.pt

# Bounded evaluation automatically restores the saved risk settings
python -m training.evaluate --checkpoint models/ra_d3qn_smoke.pt --episodes 2 --max-steps-per-episode 128 --device cpu

# Evaluate a trained checkpoint
python -m training.evaluate --checkpoint models/dqn_2048_enhanced.pt --episodes 100

# Demo the agent or MCTS
python -m gui.pygame_demo --mode agent --agent inference.autoplay:choose_action
python -m gui.pygame_demo --mode mcts --mcts-simulations 200
```

## Important caveats

- The included checkpoint is evidence that the pipeline runs; it should not be
  presented as the final trained or risk-aware result without multi-seed
  evaluation.
- Standalone trainers print/store histories in checkpoint metadata; the new
  comparison runner additionally exports history, metric, and episode CSVs.
  Analysis plots are available; plots from substantive trained runs remain pending.
- Some README structure entries still describe planned GUI/report components;
  do not infer implementation from the illustrative tree alone.
