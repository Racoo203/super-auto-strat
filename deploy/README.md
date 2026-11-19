# Running PPO training on the Ubuntu server

Training runs in Docker on the server's NVIDIA GPU. Checkpoints, metrics, TensorBoard logs and
`optuna.db` are written to `runs/` on the host. If the process crashes or the machine reboots,
training resumes from the last checkpoint on its own. You manage it over SSH from the laptop.

```
laptop ── git push ──► GitHub (CI runs the tests on Ubuntu) ──► server: git pull, make up
   ▲                                                                      │
   └──── ssh tunnel: TensorBoard :6006, Optuna dashboard :8080 ◄──────────┘
```

## One-time server setup (Ubuntu 22.04 / 24.04)

**1. NVIDIA driver.** Driver 570 or newer is required, because torch 2.10 ships CUDA 12.8.

```bash
sudo ubuntu-drivers install && sudo reboot
nvidia-smi                       # should list the GPU
```

**2. Docker Engine and the NVIDIA Container Toolkit**

```bash
# Docker Engine: https://docs.docker.com/engine/install/ubuntu/ (apt repo method), then:
sudo usermod -aG docker $USER && newgrp docker
sudo systemctl enable docker     # start on boot, so training comes back after a reboot

# NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# (add NVIDIA's apt repo, then:)
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**3. SSH access.** Install `openssh-server` and use key-based login. Keep the dashboards bound
to `127.0.0.1` (the default in `.env.example`), because they have no authentication.

**4. Clone the repo and check the host**

```bash
sudo apt-get install -y git make
git clone https://github.com/Racoo203/super-auto-strat.git && cd super-auto-strat
cp .env.example .env && nano .env    # RUN_NAME, UPDATES, N_ENVS (~ physical cores)
make check                           # driver, docker, GPU in containers: all OK?
make test                            # build the image, run pytest inside it
make gpu                             # should print the GPU, not "NO GPU"
make smoke                           # 2 updates on the GPU, resume, checkpoints
```

## Day to day

| Task | Command |
|---|---|
| Start (or resume) training and the dashboards | `make up` |
| Follow the training log | `make logs` (Ctrl+C only stops following) |
| Status | `make ps`, `nvidia-smi` |
| Stop gracefully (saves `latest.pt`) | `make stop` |
| Deploy new code | `git pull && make restart` |
| Start a new run | change `RUN_NAME` in `.env`, then `make restart` |
| Replay or watch the agent | `make shell`, then `python -m agents.ppo.replay runs/<run>` / `python -m agents.ppo.watch runs/<run>` |
| Back up runs | `make backup` (writes a tarball to `backups/`) |

**Dashboards from the laptop.** Open a tunnel and keep it running:

```bash
ssh -N -L 6006:localhost:6006 -L 8080:localhost:8080 user@server
```

Then go to http://localhost:6006 (TensorBoard) and http://localhost:8080 (Optuna).

**Copy a run to the laptop** to analyse it in `PPO.ipynb`:

```bash
rsync -avz user@server:super-auto-strat/runs/<run>/ runs/<run>/
```

## What happens when something goes wrong

- **The process crashes** (OOM, an exception). Docker restarts it up to 5 times (`restart: on-failure:5`).
  The trainer always runs with `--resume`, so it picks up from `latest.pt`, which is saved at
  every evaluation and on exit.
- **The server reboots or loses power.** Docker starts on boot and brings the trainer back. With
  a power cut you lose up to `eval_every` updates of progress. Test this once after setup: run
  `make up`, wait for the first eval, `sudo reboot`, then check that `make ps` shows `train` running
  and `make logs` shows "resuming from update N".
- **You stop it** (`make stop` / `docker compose stop`). Docker sends SIGTERM. The trainer
  finishes cleanly, saves `latest.pt` and is not restarted. `make up` resumes it.
- **The run finishes.** Exit code 0, so Docker does not restart it. Running `make up` again with the
  same `RUN_NAME` is a no-op ("nothing to do").
- **Training diverges** (NaN loss). The trainer raises and Docker retries it a few times. Check
  `make logs`, change the config and start a new `RUN_NAME`.

Resuming restores the model, the optimizer and the update counter. The self-play ghost pool is
rebuilt within a few updates. Every restart opens a new Optuna trial on the same run directory.

## Reproducibility

- **Dependencies.** `pyproject.toml` pins the direct dependencies to the versions they were tested
  with, and `uv.lock` pins everything else. The image installs exactly that set (`uv sync --frozen`).
  To upgrade, edit the pins, run `uv lock`, test and commit.
- **Code version.** The image is tagged `super-auto-strat:<commit>`, and every run records
  `git_commit` in Optuna (`-dirty` if the checkout had local edits). Deploy committed code only.
- **Config.** Each run writes `runs/<run>/config.json`. Every hyperparameter is also an Optuna
  param, so runs can be compared in the dashboard or with `load_runs()`.
- **CI.** `.github/workflows/ci.yml` runs the tests and a short training smoke test on Ubuntu
  for every push.

## Without Docker (alternative)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen
uv run python -m pytest -q
tmux new -s train 'uv run python -m agents.ppo.run --resume --name <run> --updates 5000'
```

This uses the same locked dependencies. You lose automatic restarts and the container isolation.
