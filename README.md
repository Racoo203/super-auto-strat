# Super Auto Strat

A text-based auto-battler inspired by Super Auto Pets, a Gymnasium environment for it, and a
from-scratch PPO agent that learns it through self-play. The agent trains on a GPU server in Docker,
which you manage over SSH from a laptop. Rules and design notes are in [Context.md](Context.md).

## Repo layout

| Path | What's there |
|---|---|
| `main.py`, `sap/` | the game: engine (`game.py`, `battle.py`, `pets.py`, `abilities.py`, `items.py`, `perks.py`), CLI, pet data (`sap/data/pets.csv`) |
| `sap/env/` | Gymnasium env `SuperAutoStrat-v0`: actions, masking, observation, text rendering |
| `sap/ghosts.py` | ghost pool: snapshots of teams the agents built, used as self-play opponents |
| `agents/ppo/` | PPO: `train.py` (loop), `run.py` (long run CLI), `tune.py` (Optuna search), `validate.py` (multi-seed recheck), `replay.py` / `watch.py` (inspect play). `agents/ppo/__init__.py` lists every module |
| `PPO.ipynb` | walkthrough of the PPO code, plus loading and plotting runs |
| `tests/` | pytest suite (CI runs it on every push: `.github/workflows/ci.yml`) |
| `Dockerfile`, `compose.yaml`, `Makefile`, `.env.example` | the server stack. `make help` lists the targets |
| `deploy/` | server setup guide and `check-host.sh` |
| `docs/` | the detailed guides below |
| `runs/` | output, git-ignored: `runs/<run>/` per run plus `runs/optuna.db` |

## Which guide do I need?

| I want to… | Read |
|---|---|
| play the game, or drive the engine / env from code | [docs/engine.md](docs/engine.md) |
| train on my laptop: `run.py` flags, run layout, resume, replays | [docs/training.md](docs/training.md) |
| set up the GPU server, deploy and operate it, troubleshoot | [deploy/README.md](deploy/README.md) |
| SSH from Windows/macOS: keys, `Host sas` alias, tunnels | [docs/remote-access.md](docs/remote-access.md) |
| watch training: TensorBoard, Optuna dashboard, which metrics matter | [docs/monitoring.md](docs/monitoring.md) |
| tune hyperparameters, validate the winner, launch the final run | [docs/tuning.md](docs/tuning.md) |

## Command cheat sheet

**Where:** *laptop* is your own machine (a prompt like `PS C:\...>` or `you@laptop %`). *server* is
the GPU PC after `ssh sas` (a prompt like `<user>@<host>:~$`), inside the repo directory
`<repo-dir>`. *server shell* means inside `make shell` on the server. The `sas` alias and its
tunnels are set up in [docs/remote-access.md](docs/remote-access.md).

### 1. Play and develop locally

| Command | Where | What it does |
|---|---|---|
| `uv sync` (or `python -m pip install -r requirements.txt`) | laptop | install the locked dependencies (Python 3.13) |
| `python main.py` | laptop | play the game in the terminal |
| `uv run python -m pytest -q` | laptop | run the test suite |

### 2. Train locally

| Command | Where | What it does |
|---|---|---|
| `python -m agents.ppo.run --name <run> --updates 300` | laptop | one tracked training run → `runs/<run>/` |
| `python -m agents.ppo.run --name <run> --updates 300 --resume` | laptop | continue it after an interrupt (no-op once finished) |
| `python -m agents.ppo.run --help` | laptop | every flag: `--lr`, `--entropy-coef`, `--clip-eps`, `--gae-lambda`, `--n-epochs`, `--eval-episodes`, `--from-trial`, `--params-json`, … |
| `python -m agents.ppo.replay runs/<run>` | laptop | summary of the latest evaluation games; `-e 0 --play` steps through one |
| `python -m agents.ppo.watch runs/<run> --delay 0.5` | laptop | watch the policy play a fresh game live |
| `tensorboard --logdir runs` | laptop | training curves at http://localhost:6006 |
| `optuna-dashboard sqlite:///runs/optuna.db` | laptop | every run as a trial at http://localhost:8080 |

### 3. Deploy to the server

| Command | Where | What it does |
|---|---|---|
| `git push` | laptop | publish the code; CI runs the tests on Ubuntu |
| `ssh sas` | laptop | log in to the server, with the dashboard tunnels open |
| `cd <repo-dir> && git pull` | server | fetch the new code |
| `cp .env.example .env && nano .env` | server | first time only: `RUN_NAME`, `UPDATES`, `N_ENVS`, … |
| `make check` | server | driver, Docker, GPU-in-container preflight |
| `make test` / `make gpu` / `make smoke` | server | tests in the image / which device / 2-update GPU run + resume |
| `make up` | server | build, start (or resume) training plus both dashboards |
| `make logs` | server | follow training (Ctrl+C stops following, not training) |
| `make ps` / `nvidia-smi` | server | container status / GPU load |
| `make stop` | server | stop training gracefully (saves `latest.pt`), dashboards stay up |
| `git pull && make restart` | server | deploy new code: resumes the **same** `RUN_NAME` with it |
| `make down` | server | stop everything, dashboards included |
| `make shell` | server | a shell in the image, for replay/watch and one-off scripts |

### 4. Monitor from the laptop

| Command | Where | What it does |
|---|---|---|
| `ssh sas` (or `ssh -N sas` for the tunnel only) | laptop | forwards ports 6006 and 8080 while it stays open |
| http://localhost:6006 | laptop browser | TensorBoard: read `eval/battle_win_rate` and `eval/ret` |
| http://localhost:8080 | laptop browser | Optuna dashboard. **Never delete a study or trial while something is writing to it** |

### 5. Tune hyperparameters

| Command | Where | What it does |
|---|---|---|
| `make stop` | server | free the GPU first: only one of train / tune / validate at a time |
| `make tune` | server | Optuna search in the background (study `ppo-tune`, args from `TUNE_ARGS`). Rerunning it continues the study |
| `make tune-logs` / `make tune-stop` | server | follow it / stop it gracefully |
| `make validate` | server | rerun the top 3 trials with seeds 1 2 3 for 500 updates (`VALIDATE_ARGS`) |
| `make validate-logs` / `make validate-stop` | server | follow it (summary table at the end) / stop it; rerun to continue |
| `python -m agents.ppo.validate --dry-run` | server shell | the validation plan and the summary so far, without training |

### 6. Final run

| Command | Where | What it does |
|---|---|---|
| `nano .env`: new `RUN_NAME`, `EXTRA_ARGS=--from-trial ppo-tune:<N>` | server | train with trial N's tuned hyperparameters |
| `make up` then `make logs` | server | start it; the first log line prints the params it took from the trial |

### 7. Analyse and back up

| Command | Where | What it does |
|---|---|---|
| `make backup` | server | tarball of `runs/` (checkpoints, metrics, `optuna.db`) in `backups/` |
| `scp -r sas:<repo-dir>/runs/<run> runs/` | laptop | copy one run to the laptop (or `rsync -avz` on macOS/Linux) |
| `scp sas:<repo-dir>/runs/optuna.db runs/` | laptop | copy the Optuna database (all studies) |
| `python -m agents.ppo.replay runs/<run>` / `python -m agents.ppo.watch runs/<run>` | laptop or server shell | inspect how it plays |
| `PPO.ipynb`: `load_runs()`, `load_metrics("runs/<run>")` | laptop | compare runs and plot curves as DataFrames |
