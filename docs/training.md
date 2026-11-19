# Training the PPO agent

`agents/ppo/` is a from-scratch PPO; [PPO.ipynb](../PPO.ipynb) walks through the code. This page
covers running it: from Python, from the `run.py` CLI, what a run writes, resuming, and inspecting
how the agent plays. For the server see [deploy/README.md](../deploy/README.md). For hyperparameter
search see [tuning.md](tuning.md).

## From Python

```python
from agents.ppo import PPOConfig, train

result = train(PPOConfig(n_updates=300), "baseline")                   # tracked run -> runs/baseline/
train(PPOConfig(n_updates=300, n_policies=2), "duel")                 # two agents trained against each other
train(PPOConfig(n_updates=300, opponent="random"), "random")          # the original: random opponent teams
result.best_objective   # best eval/battle_win_rate seen (the Optuna trial value)
```

Opponents are **ghosts** by default. This is self-play as in Super Auto Pets: each player fights a
snapshot of a team another player built at the same round (Context.md §9). `PPOConfig` in
`agents/ppo/config.py` holds every hyperparameter, with comments.

Async envs spawn subprocesses, so on Windows scripts need an `if __name__ == "__main__":` guard.
Use `run.py` (below) rather than a notebook for anything long.

## `run.py`: one long, resumable run

```
python -m agents.ppo.run --name final --updates 5000
python -m agents.ppo.run --name final --updates 5000 --resume       # after a crash/interrupt
python -m agents.ppo.run --name final --updates 5000 --from-trial ppo-tune:7
python -m agents.ppo.run --help                                     # all flags
```

| Flag | `PPOConfig` field | Default |
|---|---|---|
| `--name` | run directory `runs/<name>/` | a timestamp |
| `--resume` | continue `runs/<name>/checkpoints/latest*.pt` | off |
| `--updates` | `n_updates` | 1000 |
| `--lr` | `lr` | 2.5e-4 |
| `--entropy-coef` | `entropy_coef` | 0.01 |
| `--clip-eps` | `clip_eps` | 0.2 |
| `--gae-lambda` | `gae_lambda` | 0.95 |
| `--n-epochs` | `n_epochs` | 4 |
| `--n-envs` | `n_envs` (env worker processes, ~ physical cores) | 8 |
| `--n-policies` | `n_policies` (P > 1: P agents trained against each other) | 1 |
| `--opponent` | `opponent`: `ghosts` (self-play) or `random` | ghosts |
| `--seed` | `seed` | 0 |
| `--eval-every` | `eval_every` (updates between evaluations, 0 = never) | 25 |
| `--eval-episodes` | `eval_episodes` (games per evaluation) | 50 |
| `--checkpoint-every` | `checkpoint_every` | 50 |
| `--keep-checkpoints` | `keep_checkpoints` (numbered checkpoints kept, 0 = all) | 3 |
| `--runs-dir` | `runs_dir` | runs |
| `--from-trial STUDY:N` | the tuned params (`lr, entropy_coef, clip_eps, gae_lambda, n_epochs`) of Optuna trial N | – |
| `--params-json JSON\|FILE` | any `PPOConfig` fields, inline (`'{"lr": 1e-4}'`) or from a `.json` file | – |
| `--storage` | Optuna storage `--from-trial` reads | `sqlite:///<runs-dir>/optuna.db` |

**Precedence**, lowest to highest: `PPOConfig` defaults, then `--from-trial`, then `--params-json`,
then explicit flags. So `--from-trial ppo-tune:7 --lr 1e-4` uses trial 7's params, except `lr`.
`--from-trial` only copies the five tuned params. The trial's own `n_updates`, `n_envs`, etc. are
not copied.

**Evaluation cost.** Each evaluation plays `eval_episodes` games against random teams. With ghosts
it also plays `eval_episodes` games against the live ghost pool, so an eval with the default of 50
plays 100 games. An untrained network takes about 0.3–0.4 s per game on a laptop CPU, and trained
agents survive more rounds, so expect an eval to take about a minute. That is comparable to the
25 updates between evals. If that is too slow, use `--eval-every 50` or `--eval-episodes 30`. Eval
games use fixed seeds (`seed + 10000 + i`), so evaluations of one run are comparable over time.

## What a run writes

```
runs/
  optuna.db                     every run is a trial (study "ppo"); tuning uses "ppo-tune"
  <run>/
    config.json                 the PPOConfig
    metrics.jsonl               one JSON line per update
    tb/                         TensorBoard event files
    replays/update_XXXX.jsonl   the evaluation games, every decision recorded
    checkpoints/
      latest.pt                 newest weights + optimizer (each eval and on exit)
      best.pt                   weights at the best eval/battle_win_rate so far
      update_XXXX.pt            every checkpoint_every updates, newest keep_checkpoints kept
      latest_p1.pt, best_p1.pt  the other policies, with n_policies > 1
```

Logged metrics are listed in the `agents/ppo/train.py` docstring. Which ones to read is covered in
[monitoring.md](monitoring.md).

## Resume semantics

- `--resume` loads each policy's `latest.pt` (model, optimizer, update counter) and continues at
  the next update. With no checkpoint it starts fresh. Once the run has reached `--updates` it
  exits at once with "nothing to do". That is why the server always passes `--resume`.
- `latest.pt` records the last update whose PPO step **finished**. An update cut off mid-rollout
  (Ctrl+C, SIGTERM, a crash) is redone on resume, not skipped.
- `latest.pt` is written at every evaluation and on exit. After a hard kill (power cut, OOM-kill)
  you lose up to `eval_every` updates.
- Not persisted: the self-play ghost pool (rebuilt within a few updates) and the RNG state.
- Writes append to the same `runs/<name>/` directory. Each launch opens a **new** Optuna trial,
  because Optuna can't resume a trial. One long run with restarts shows up as several trials.
- Hyperparameter flags are re-read on every launch. Keep them the same for the whole run. To change
  them, start a new `--name`.
- SIGTERM (`docker stop`, `systemctl stop`) is handled like Ctrl+C: `latest.pt` is saved, the trial
  is marked FAIL and the exit code is 130, so a restart-on-failure supervisor brings it back.
- An Optuna storage error after startup (the study deleted in the dashboard, a locked `optuna.db`)
  logs a warning and training continues. Checkpoints, metrics and TensorBoard don't depend on Optuna.

## Inspecting play

| Question | Tool |
|---|---|
| How did it play in the last eval? | `python -m agents.ppo.replay runs/<run>` summarises the latest evaluation games. Add `-e 0 --play [--battles-only] [--delay 0.5]` to step through game 0 |
| A specific eval? | `python -m agents.ppo.replay runs/<run>/replays/update_0099.jsonl` |
| How does it play now? | `python -m agents.ppo.watch runs/<run> --delay 0.5` plays a fresh game: board, last battle, top actions with V(s). Options: `--greedy`, `--episodes N`, `--seed S`, `--battles-only`, `--no-clear` |
| Against another policy? | `python -m agents.ppo.watch runs/<run> --vs runs/duel/checkpoints/latest_p1.pt` fights that policy's ghosts |
| An untrained baseline? | `python -m agents.ppo.watch --random` |
| Who beats whom? | TensorBoard `eval/vs_ghosts/*`, `eval/h2h_p0_vs_p1/*`. In code: `head_to_head(model_a, model_b, 20)` |

On the server, run these inside `make shell`. Or copy the run to the laptop first:
`scp -r sas:<repo-dir>/runs/<run> runs/`.

## Analysing runs in `PPO.ipynb` or a notebook

```python
from agents.ppo import load_runs, load_metrics, load_checkpoint

runs = load_runs()                            # study "ppo" as a DataFrame: value, state, params_*, user_attrs_*
tuning = load_runs(study_name="ppo-tune")
m = load_metrics("runs/server-selfplay-01")   # metrics.jsonl, indexed by update
m["eval/battle_win_rate"].dropna().plot()
model, cfg, ckpt = load_checkpoint("runs/server-selfplay-01/checkpoints/best.pt")
```
