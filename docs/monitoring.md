# Monitoring training

Two dashboards run next to training on the server (`make up` starts them, and `make tune` /
`make validate` start them too). Both are bound to `127.0.0.1` on the server and reached through
the SSH tunnel: `ssh sas` (or `ssh -N sas`) from the laptop, see
[remote-access.md](remote-access.md). Then:

| Dashboard | URL (laptop) | Shows |
|---|---|---|
| TensorBoard | http://localhost:6006 | per-update curves of every run in `runs/` |
| Optuna dashboard | http://localhost:8080 | every run/trial in `runs/optuna.db`: params, value, state, user attrs |

Locally: `tensorboard --logdir runs` and `optuna-dashboard sqlite:///runs/optuna.db`.

On the server, `make logs` (or `make tune-logs` / `make validate-logs`) prints one line per update
and an `EVAL` line at each evaluation, `EVAL win 98% battles 81% ...`.

## Which metrics matter

| Metric | Read it as |
|---|---|
| `eval/battle_win_rate` | **the objective.** Battles won vs random teams on fixed seeds. Comparable over time and across runs |
| `eval/ret` | mean episode return in the eval games. Still discriminates when win rates saturate |
| `eval/win_rate` | whole games won vs random teams. **Saturates** at 0.9–1.0 early (~update 175), so it can't rank runs. Kept as a sanity check |
| `eval/rounds`, `eval/wins`, `eval/losses` | how long games last, and how they're won or lost |
| `eval/invalid_rate` | share of masked-out actions attempted. Should stay ~0 |
| `eval/vs_ghosts/battle_win_rate` | vs the live self-play pool. Tends toward ~50% by design (it's playing itself), so a flat line is fine |
| `eval/h2h_p0_vs_p1/*` | with `n_policies` > 1: policy 0 vs policy 1's ghosts |
| `train/*` | self-play games: drift toward 50%, so they say little about strength |
| `loss/entropy` | falls slowly. A collapse to ~0 early means the policy stopped exploring (raise `entropy_coef`) |
| `loss/approx_kl`, `loss/clip_frac` | step size. Persistently high KL (≫ 0.02) or clip_frac (> 0.3) means updates are too big (lower `lr` / `n_epochs`) |
| `loss/explained_var` | how well V(s) predicts returns. Should rise toward 1 |
| `ghosts/pool_size`, `ghosts/random_frac` | self-play pool health |
| `perf/sps` | environment steps per second |

In the Optuna dashboard a trial's **value** is its best `eval/battle_win_rate`. Runs from before the
objective change have `eval/win_rate` there instead (see [tuning.md](tuning.md#the-objective)).
User attrs record `git_commit`, `run_name`, `best_update`, `objective`, `resumed_from_update` and,
for tuned launches, `params_from_trial`.

## Rules for the Optuna dashboard

- **Never delete a study or a trial while a run is writing to it.** That is what caused the
  `KeyError: 'Record does not exist.'` crash at update 99 of `server-selfplay-01`. Training now
  survives it with a warning, but the rest of that run is untracked in Optuna, and a run launched
  with `--from-trial` needs its source trial on every restart. Delete only when nothing is running
  (`make ps`), and `make backup` first.
- **Studies:** `ppo` = plain runs (every launch or restart of a run is a new trial), `ppo-tune` =
  tuning trials, `ppo-validate` = seeded reruns. See [tuning.md](tuning.md#studies).
- **Smoke runs and tests never appear here.** `make smoke` writes to `/tmp/smoke` inside a throwaway
  container, and the tests use temporary directories. Anything you see in the dashboard is real
  and lives in `runs/`.
- One long run with restarts shows up as several `ppo` trials with the same `run_name`. The later
  ones have `resumed_from_update` set.
