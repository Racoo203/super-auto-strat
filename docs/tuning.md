# Hyperparameter tuning

The workflow: **search** (`make tune`) → **validate** the top few with more seeds
(`make validate`) → **launch** the winner as a long run (`EXTRA_ARGS=--from-trial ...`). Every
command runs on the server, in `<repo-dir>`.

## Studies

All studies live in `runs/optuna.db`:

| Study | Written by | One trial is |
|---|---|---|
| `ppo` | `run.py` / `train()` / `make up` | one launch of a run (each restart opens a new trial) |
| `ppo-tune` | `make tune` (`agents/ppo/tune.py`) | one tuning trial, run name `ppo-tune-NNN` |
| `ppo-validate` | `make validate` (`agents/ppo/validate.py`) | one seeded rerun, run name `val-ppo-tune-tNNN-sS` |

They are kept separate so that the pruner only compares tuning trials with each other.

## The objective

A trial's value is its **best `eval/battle_win_rate`**: battles won ÷ battles fought in the
evaluation games against random teams. `best.pt` follows the same metric.

The old objective was `eval/win_rate`, the share of whole games won. It saturates: against random
teams it sits at 0.9–1.0 from about update 175 onward. At p = 0.9 with 20 games its standard error
is about 0.07, so trials couldn't be told apart, and `best.pt` stopped updating once a run hit 1.0.
`eval/battle_win_rate` keeps improving after that. Evaluations are also larger now
(`eval_episodes` 50, up from 20), which makes the objective less noisy.

Runs started before this change (e.g. `server-selfplay-01`) have `eval/win_rate` as their Optuna
value. Compare them with new runs through the `eval/battle_win_rate` curve in TensorBoard (logged
by both), not by the Optuna value.

## 1. Search: `make tune`

```bash
make stop          # train must not be running: tune shares the GPU and the CPU env workers
make tune          # starts container sas-tune in the background (plus the dashboards)
make tune-logs     # follow it (Ctrl+C stops following, not tuning)
make tune-stop     # graceful stop: the running trial is marked FAIL and saves its latest.pt
```

- Arguments come from `TUNE_ARGS` in `.env`, e.g. `TUNE_ARGS=--trials 20 --updates 200`. Without
  that line it uses the `tune.py` defaults, which are the same. `make tune` always passes
  `--n-envs $N_ENVS` first, and a `--n-envs` in `TUNE_ARGS` overrides it. Other flags:
  `--eval-every` (default 20), `--eval-episodes` (default 50), `--seed`, `--study`, `--storage`
  (`python -m agents.ppo.tune --help`).
- It searches `lr` (1e-5–1e-3, log scale), `entropy_coef` (0–0.05), `clip_eps` (0.1–0.3),
  `gae_lambda` (0.9–0.99) and `n_epochs` (2–8).
- **Pruning.** `MedianPruner`: after 3 complete trials, a trial whose objective falls below the
  median of earlier trials at the same update is stopped early (marked PRUNED). There's no pruning
  in the first `2 × eval_every` updates.
- **Rerunning continues the study.** `make tune` again with the same study name loads it
  (`load_if_exists`) and runs `--trials` **more** trials, using everything learned so far.
- **Failures.** A trial that crashes is marked FAIL and the search moves on to the next trial. An
  Optuna storage error doesn't end the search either. Interrupted trials are not resumed. They
  stay FAIL and the next run starts a new trial.
- `make up` refuses to start while `sas-tune` or `sas-validate` is running, and `make tune` refuses
  while `train` is running.
- Rough cost: 200 updates at ~1.6 s (≈ 5 min) plus 10 evals at up to a minute each, so roughly
  10–15 min per unpruned trial and a few hours for 20 trials (pruned ones stop early). Check the
  real timings in `make tune-logs` and lower `--eval-episodes` if evals dominate.

## 2. Validate the top configs: `make validate`

Tuning trials are single-seed (seed 0) and short (200 updates), so the top of the leaderboard is
partly luck. Before committing to a multi-hour run, rerun the best few for longer with other seeds:

```bash
make validate         # VALIDATE_ARGS=--top 3 --seeds 1 2 3 --updates 500 by default
make validate-logs    # per-run progress, then the summary table
make validate-stop    # graceful stop; `make validate` again continues where it stopped
```

This takes the top `--top` COMPLETE trials of `--study` (default `ppo-tune`) and trains each one
with every seed in `--seeds`. Runs that already finished are skipped and unfinished ones resume.
The table at the end ranks configs by the mean, over seeds, of each run's **last 3 evals** (less
luck-driven than the single best eval), with the standard deviation and the best eval
(illustrative numbers):

```
eval/battle_win_rate: mean of each run's last 3 evals, then mean ± std over seeds
      t007  ppo-tune:7       0.842 ± 0.012   (best eval 0.861, 3/3 seeds)
      t012  ppo-tune:12      0.829 ± 0.031   (best eval 0.858, 3/3 seeds)
   default  default          0.801 ± 0.018   (best eval 0.823, 3/3 seeds)

launch it: EXTRA_ARGS=--from-trial ppo-tune:7   (in .env, with a new RUN_NAME)
```

Add `--baseline` to `VALIDATE_ARGS` to include the default `PPOConfig` as a reference row. To see
the plan and the summary so far without training anything, run
`make shell`, then `python -m agents.ppo.validate --dry-run` (with the same flags).

Pick a config that is ahead by more than about one standard deviation. If the top two overlap,
prefer the lower std, or the one closer to the defaults. 3 configs × 3 seeds × 500 updates is 9
runs of roughly 15 min training plus 20 evals each: plan for an evening (4–5 h), or use
`--seeds 1 2` / `--top 2` to halve it.

Doing it by hand instead works too: one run per (trial, seed),
`python -m agents.ppo.run --name val-t7-s1 --updates 500 --seed 1 --from-trial ppo-tune:7`.

## 3. Launch the final run with the tuned params

Edit `.env`: give it a **new** `RUN_NAME` (a new name means a fresh run, not a resume of the
baseline) and put the trial in `EXTRA_ARGS`:

```ini
RUN_NAME=server-tuned-01
UPDATES=5000
EXTRA_ARGS=--from-trial ppo-tune:7
```

Then `make up` and `make logs`. The first lines show
`hyperparameters from ppo-tune:7 (COMPLETE, value 0.861): {...}`. Other ways to set them:

```ini
# explicit flags (they win over --from-trial if you combine them)
EXTRA_ARGS=--lr 1.3e-4 --entropy-coef 0.021 --clip-eps 0.17 --gae-lambda 0.94 --n-epochs 6
# a JSON file under runs/ (mounted in the container), any PPOConfig fields
EXTRA_ARGS=--params-json runs/final-params.json
```

No quotes in `.env`. Compose passes the value to `run.py` as-is. `--from-trial` reads
`runs/optuna.db` on every (re)start, so **don't delete that trial or study** while the run is
going: the next restart would fail to find it. If you need to prune the database, switch to
explicit flags first.
