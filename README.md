# Risk-Aware Deep Reinforcement Learning for Long-Horizon Decision Making under Stochastic Uncertainty
### An Empirical Study Using the 2048 Game Environment

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running tests

```bash
pytest -q
# or, if pytest isn't available:
PYTHONPATH=. python3 tests/test_game.py
```

## Quick smoke tests

```bash
python -m game.game          # pure mechanics demo (a few scripted moves)
python -m game.environment   # RL-wrapper demo (one full random episode)
python -m training.train --episodes 2000 --checkpoint models/dqn_2048_enhanced.pt
python -m training.evaluate --checkpoint models/dqn_2048_enhanced.pt --episodes 100
```

## Demo GUI

The pygame GUI is for presentation/demo use only. It renders the project's
existing `Game2048` engine and is not used by training or tests.

```bash
python -m gui.pygame_demo --mode human
python -m gui.pygame_demo --mode heuristic --delay-ms 180
python -m gui.pygame_demo --mode mcts --delay-ms 140 --mcts-simulations 200
python -m gui.pygame_demo --mode random --seed 42
```

For a trained agent demo, expose a callable that returns one of
`game.game.Action` or action ids `0..3`, then run:

```bash
python -m gui.pygame_demo --mode agent --agent inference.autoplay:choose_action
```

`training.train` uses a masked Dueling Double DQN: the MLP receives the 16
log2 tile exponents, experience replay decorrelates updates, score rewards are
scaled for stable Q-targets, and the target network is softly synchronized.
Illegal moves are excluded in both behaviour and bootstrapping, which is
important because the environment otherwise permits them. Training runs for
the requested number of episodes. A checkpoint is saved after training (and
every 250 episodes by default).

The GUI automatically prefers `models/dqn_2048_enhanced.pt` when it exists,
falling back to the original `models/dqn_2048.pt` checkpoint. Set
`DQN_CHECKPOINT` to select a specific saved model.

### Separate Double DQN training

`rl/doubledqn.py` provides `DoubleDQNAgent` and the Double DQN next-action
calculation. It reuses the existing DQN network, replay buffer, learning loop,
and evaluation interface. The online network selects the best legal next
action and the target network evaluates it; terminal states do not bootstrap.
The original `DQNAgent` and its `double_dqn=False` standard-DQN option remain
available. Algorithm tests live in `tests/test_doubledqn.py`.

The dedicated trainer initializes fresh weights; it does not load the saved
DQN checkpoint. Its default output is `models/doubledqn_2048.pt`.
The existing dueling architecture remains the default; use `--no-dueling` for
plain Double DQN. Both reuse the same network implementation. Run commands from
the repository root with the dependencies in `requirements.txt` installed.
`--help` lists learning-rate, gamma, replay, target-update, exploration, and
checkpoint-frequency controls. Checkpoints contain model/optimizer state and
training history; they do not contain replay or RNG state for exact resumption.
Choose a distinct checkpoint path for each run: training overwrites that path.

```bash
# Short CPU smoke run with learning enabled
python -m training.train_doubledqn --episodes 1 --max-steps-per-episode 32 --batch-size 4 --min-replay-size 4 --device cpu --checkpoint models/doubledqn_smoke.pt

# Plain Double DQN training from scratch, when ready
python -m training.train_doubledqn --no-dueling --episodes 2000 --seed 42 --checkpoint models/doubledqn_2048.pt

# Evaluate using the existing evaluator
python -m training.evaluate --checkpoint models/doubledqn_2048.pt --episodes 100
```

### Risk-aware action evaluation (proposal objective 4)

`rl/risk.py` implements a configurable risk score for each legal action's
deterministic afterstate, using the existing game rules without modifying the
live board or consuming random numbers:

```text
mobility risk = 1 - legal moves / 4
crowding risk = 1 - empty cells / board cells
spawn danger = probability that the next tile spawn immediately ends the game
R(s, a) = weighted mean of these three components
Q_RA(s, a) = Q(s, a) - lambda * R(s, a)
```

Mobility and empty cells are measured before the random spawn. Spawn danger
uses the configured probability of a 2 versus a 4: with one empty cell it
checks both outcomes; with at least two empty cells a single spawn cannot end
a reachable game. This is a local calculation, with no rollouts or tree search.
The composite score is a heuristic in [0, 1], not a probability of eventual
failure. Equal default weights are an initial design choice, not a result
established by the proposal or experiments. `RiskConfig` allows ablations.

```python
import numpy as np
from game.environment import Env2048
from rl.doubledqn import DoubleDQNAgent
from rl.risk import risk_adjusted_q_values

env = Env2048(seed=42)
state = env.reset()
agent = DoubleDQNAgent()  # Fresh model; this example is not a trained policy.
values = risk_adjusted_q_values(env.game, agent.q_values(state), risk_coefficient=0.1)
action = int(np.argmax(values))
next_state, reward, done, info = env.step(action)
```

Lambda is in the network's Q-value units, which already reflect reward
scaling. Zero lambda reproduces the masked baseline; illegal moves remain
excluded. The utility leaves rewards and learning targets unchanged. The
policy-only RA-D3QN integration below now uses it during training and inference.

Run the focused tests with `python3 -m pytest tests/test_risk.py -q`.

### RA-D3QN training, evaluation, and inference

`rl/risk_aware.py` provides `RiskAwareDQNAgent`: a Dueling Double DQN with
`Q - lambda * R` greedy action selection. Epsilon exploration still samples
uniformly from legal actions. This first variant adjusts the **behaviour
policy only**. It retains the existing score rewards, reward scaling, replay,
and ordinary Double DQN targets (`argmax Q_online`, evaluated by `Q_target`).
The network therefore learns the ordinary score-return objective; it does not
learn a risk-adjusted return. Lambda zero reproduces the baseline behaviour
and updates. The coefficient and risk weights are saved in the checkpoint.

```bash
# Short CPU smoke test; epsilon zero ensures it exercises the risk policy.
python3 -m training.train_risk_aware --episodes 1 --max-steps-per-episode 32 --batch-size 4 --min-replay-size 4 --epsilon-start 0 --epsilon-end 0 --risk-coefficient 0.1 --device cpu --checkpoint models/ra_d3qn_smoke.pt

# Evaluate the smoke checkpoint with a bounded episode length.
python3 -m training.evaluate --checkpoint models/ra_d3qn_smoke.pt --episodes 2 --max-steps-per-episode 128 --device cpu

# Full training from scratch, when ready; normal epsilon decay is the default.
python3 -m training.train_risk_aware --episodes 2000 --risk-coefficient 0.1 --checkpoint models/ra_d3qn_2048.pt

# Demo with the saved risk policy (macOS/Linux).
DQN_CHECKPOINT=models/ra_d3qn_2048.pt python3 -m gui.pygame_demo --mode agent --agent inference.autoplay:choose_action
```

`--mobility-weight`, `--crowding-weight`, and `--game-over-weight` set the
relative risk weights. The default coefficient 0.1 and equal weights are
provisional, not experimentally validated. Use `--risk-coefficient 0` for
the no-risk ablation. Training always initializes fresh weights; the default
output `models/ra_d3qn_2048.pt` is separate from the baseline checkpoint.

The existing evaluator and GUI loader automatically restore the correct agent
and saved risk settings. Evaluation now includes reaching 2048, maximum tile,
episode length, termination/truncation rates, and board-risk statistics.
Risk statistics pool **pre-action, nonterminal** states and always use equal
weights for comparable measurement across policies. Low mobility means at
most one legal move; high risk uses `--high-risk-threshold` (default 0.5).
Game-over excludes step-limit truncation. Without a step limit, completed 2048
episodes naturally have game-over rate 1; that alone cannot measure robustness.

Run the integration tests with `python3 -m pytest tests/test_risk_aware.py -q`.
Multi-seed training, comparisons with all proposed baselines, and lambda/weight
ablations remain necessary before claiming improved gameplay.

### Reproducible baseline comparisons

`training/compare.py` reuses the shared trainer/evaluator and explicitly
configures every learned baseline:

| Agent name | Double DQN | Dueling | Risk-aware policy |
| --- | --- | --- | --- |
| `dqn` | No | No | No |
| `double_dqn` | Yes | No | No |
| `dueling_dqn` | No | Yes | No |
| `dueling_double_dqn` | Yes | Yes | No |
| `ra_d3qn` | Yes | Yes | Yes |

`random` samples legal moves uniformly. `greedy` maximizes immediate merge
score and breaks ties by action id; it is distinct from the GUI heuristic.
Both are non-learning reference policies defined in `rl/baselines.py`.

```bash
# Small pipeline check for all seven policies, using two seeds.
python3 -m training.compare --smoke --seeds 42 43 --output-dir results/comparison_smoke

# Full experiment, when ready (potentially expensive; not a smoke test).
python3 -m training.compare --episodes 2000 --seeds 42 43 44 --eval-episodes 100 --device cpu --output-dir results/comparison_full

# Focused zero-risk ablation, using a NEW output directory.
python3 -m training.compare --smoke --agents dueling_double_dqn ra_d3qn --risk-coefficient 0 --output-dir results/zero_risk_smoke
```

`--smoke` overrides training to 1 episode capped at 32 steps, evaluation to
2 episodes capped at 32 steps, CPU, batch/warmup size 4, and epsilon zero to
exercise greedy/risk policies. It does not validate playing strength. Without
`--smoke`, normal training defaults and epsilon decay apply. Optional
`--eval-max-steps` caps evaluation; uncapped evaluation is the default.

The output directory must be new. It contains `manifest.json` with settings,
software versions, and completion/failure status; `results.csv` with one row
per agent/seed; and `summary.csv` with means and sample standard deviations
across run metrics. Each learned run retains its checkpoint and `history.csv`
in `<agent>/seed_<seed>/`; every run saves `metrics.json`. Warm-up losses are
blank, not zero. Completed partial results survive a later run failure.

All policies use the same evaluation environment seed range, separate from
the supplied training/policy seeds. Random-policy seeds control action RNG;
greedy rows repeat a deterministic policy and are not independent replicates.
Single-run summary standard deviations are blank. These statistics are not
confidence intervals. Training uses equal episode/step limits, not equal
realized updates; actual steps, updates, model flags, parameter counts, and
devices are recorded. Dueling changes parameter count despite common hidden
widths. Risk measurement uses the shared evaluator's equal-weight definition.

Run `python3 -m pytest tests/test_comparison.py -q` for comparison tests.

### Episode exports and analysis

New comparison runs save `evaluation.csv` beside each run's metrics and
checkpoint. It records episode/environment seed, score, maximum tile, move
count, termination versus truncation, and pre-action board-risk measurements.
Aggregate metrics remain unchanged; decision-state risk averages use the
number of moves, not equal weighting of unequal-length episodes.

```bash
# Analyze saved comparison data without training or playing any more games.
python3 -m training.analyze --input-dir results/comparison_smoke --smoothing-window 25

# Export individual episodes when evaluating a single checkpoint.
python3 -m training.evaluate --checkpoint models/ra_d3qn_2048.pt --episodes 100 --episodes-csv results/ra_evaluation.csv
```

Analysis writes a new `analysis/` directory containing `report.md`,
`evaluation.png`, `training.png` (when training histories exist), and
`episode_scores.png` (when episode exports exist). Use `--output-dir` for a
different location; existing analysis directories and episode CSVs are never
overwritten. Matplotlib, already in `requirements.txt`, renders headlessly.

Training curves show a separate line per seed and a trailing episode mean;
missing warm-up losses remain missing. Evaluation error bars show sample
standard deviation across **run means**, not confidence intervals. Episode
scores are pooled descriptively and are not independent training replicates.
The report labels partial comparisons, capped evaluations, missing older-format
episode exports, and single-run standard deviations. It does not automatically
declare significance, convergence, or a winning agent. Smoke charts are not
evidence of trained-model performance.

Run `python3 -m pytest tests/test_analysis.py -q` for export/analysis tests.

## Repository structure

```
2048-risk-aware-rl/
├── game/
│   ├── game.py          # pure 2048 rules engine (board, moves, merges, spawn, game-over)
│   └── environment.py   # RL-facing wrapper: encoding, reward, Gym-like step/reset, action simulation
├── rl/
│   ├── dqn.py            # DQN network (MLP)
│   ├── replay_buffer.py  # experience replay
│   ├── agent.py           # epsilon-greedy agent, training loop glue
│   └── risk.py            # Board Risk Index + action-specific risk (Section 9-12)
├── gui/
│   ├── pygame_ui.py       # human-play + AI autoplay rendering
│   └── visualizer.py      # Q-value / risk analysis overlay
├── training/
│   ├── train.py           # training entrypoint (Standard DQN / Risk-Aware DQN)
│   └── evaluate.py         # batch evaluation across many episodes, metrics (Section 22)
├── inference/
│   └── autoplay.py         # load trained_model.pth, run/greedy-play, no learning
├── models/                 # saved .pth checkpoints
├── tests/
│   └── test_game.py         # unit tests for game mechanics + environment
└── requirements.txt
```

## Status

- [x] Step 1: 2048 environment (`game/game.py`, `game/environment.py`) + unit tests
- [x] Step 2: Random & immediate-score greedy evaluation baselines
- [x] Step 3: Pure Monte Carlo Tree Search demo agent
- [x] Step 4: Evaluation framework and multi-seed comparison runner (full experiments pending)
- [x] Step 5: Standard DQN (MLP) + replay buffer + target network + training loop
- [x] Step 6: Board Risk Index + action-specific risk module
- [x] Step 7: Policy-only RA-D3QN (risk-adjusted action selection; unchanged rewards)
- [ ] Step 8: Ablation study
- [ ] Step 9: Risk-sensitivity (λ sweep) experiments
- [ ] Step 10: Robustness / generalization experiments
- [ ] Step 11: Pygame GUI (human mode, autoplay, analysis overlay)
- [ ] Step 12: Optional CNN-DQN comparison
- [ ] Step 13: Final report / presentation

See project context doc for full research questions, risk formulation, and
the 10–12 week plan / group division of labor.
