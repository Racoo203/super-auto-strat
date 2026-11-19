# Running PPO training on the Ubuntu server

Training runs in Docker on the server's NVIDIA GPU. Checkpoints, metrics, TensorBoard logs and
`optuna.db` are written to `runs/` on the host. If the process crashes or the machine reboots,
training resumes from the last checkpoint on its own. You manage it over SSH from the laptop.
SSH setup is in [docs/remote-access.md](../docs/remote-access.md). Every command is also listed in
the [README cheat sheet](../README.md#command-cheat-sheet).

```
laptop ── git push ──► GitHub (CI runs the tests on Ubuntu) ──► server: git pull, make up
   ▲                                                                      │
   └──── ssh tunnel: TensorBoard :6006, Optuna dashboard :8080 ◄──────────┘
```

Placeholders: `<user>`, `<pc-ip>`, and `<repo-dir>` (where you cloned the repo on the server, e.g.
`~/super-auto-strat` or `~/Documents/Development/super-auto-strat`). The commands below run on the
**server** unless they say laptop.

## One-time server setup (Ubuntu 22.04 / 24.04, also tested on 26.04)

**1. NVIDIA driver.** Driver 570 or newer is required, because torch 2.10 ships CUDA 12.8.

```bash
sudo ubuntu-drivers install && sudo reboot
nvidia-smi                       # should list the GPU and a driver version >= 570
```

**2. Docker Engine.** Docker's guide [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/),
section **"Install using the apt repository"**. As of writing, that is:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker $USER    # run docker without sudo...
sudo systemctl enable docker     # ...and start it on boot, so training comes back after a reboot
```

The group change only applies to **new** login sessions: log out and back in (or reconnect SSH).
`newgrp docker` only fixes the current shell. Check with `docker run --rm hello-world`. If Docker's
repo doesn't publish your Ubuntu codename yet, the linked page has the current workaround.

**3. NVIDIA Container Toolkit.** NVIDIA's [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
section **"Installing with Apt"**, then **"Configuring Docker"**. As of writing:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all ubuntu nvidia-smi    # the GPU, seen from a container
```

**4. SSH access.** Install `openssh-server` and set up key-based login from the laptop, a `Host sas`
alias with the dashboard tunnels, and a DHCP reservation: [docs/remote-access.md](../docs/remote-access.md).
Keep the dashboards bound to `127.0.0.1` (the default in `.env.example`), because they have no
authentication.

**5. Clone the repo and check the host**

```bash
sudo apt-get install -y git make
git clone https://github.com/Racoo203/super-auto-strat.git <repo-dir> && cd <repo-dir>
cp .env.example .env && nano .env    # RUN_NAME, UPDATES, N_ENVS (~ physical cores)
make check                           # driver, docker, GPU in containers: all OK?
make test                            # build the image, run pytest inside it
make gpu                             # should print the GPU, not "NO GPU"
make smoke                           # 2 updates on the GPU, resume, checkpoints
```

**6. Test a reboot once.** `make up`, wait for the first `EVAL` line in `make logs`, then
`sudo reboot`. After it comes back, `make ps` should show `train` running and `make logs` should
show "resuming from update N".

## Day to day

| Task | Command |
|---|---|
| Start (or resume) training and the dashboards | `make up` |
| Follow the training log | `make logs` (Ctrl+C only stops following) |
| Status | `make ps`, `nvidia-smi` |
| Stop gracefully (saves `latest.pt`) | `make stop` |
| Deploy new code | `git pull && make restart` (see below) |
| Start a new run | change `RUN_NAME` in `.env`, then `make restart` |
| Hyperparameter search / validation | `make stop`, then `make tune` / `make validate`: [docs/tuning.md](../docs/tuning.md) |
| Replay or watch the agent | `make shell`, then `python -m agents.ppo.replay runs/<run>` / `python -m agents.ppo.watch runs/<run>` |
| Back up runs | `make backup` (writes a tarball to `backups/`) |
| Everything `make` can do | `make help` |

**Deploy new code.** `git pull && make restart` rebuilds the image and **resumes the current
`RUN_NAME` with the new code**. That's right for fixes that don't change what is measured. If the
change affects evaluation, the objective or the config (e.g. a new metric, different
`eval_episodes`, a different network), start a new `RUN_NAME` instead. Otherwise one run mixes
two regimes and its curves can't be read. To avoid interrupting a run at all, wait until
`make logs` shows `done: <run> ...`.

**Dashboards from the laptop.** With the `Host sas` alias from
[docs/remote-access.md](../docs/remote-access.md), `ssh sas` (or `ssh -N sas`) opens the tunnels.
Without it:

```bash
ssh -N -L 6006:localhost:6006 -L 8080:localhost:8080 <user>@<pc-ip>
```

Then go to http://localhost:6006 (TensorBoard) and http://localhost:8080 (Optuna). Which metrics
to read: [docs/monitoring.md](../docs/monitoring.md).

**Copy a run to the laptop** to analyse it in `PPO.ipynb` (run these on the laptop):

```bash
scp -r sas:<repo-dir>/runs/<run> runs/                       # Windows: scp ships with OpenSSH, rsync doesn't
rsync -avz sas:<repo-dir>/runs/<run>/ runs/<run>/            # macOS/Linux: incremental, resumable
scp sas:<repo-dir>/runs/optuna.db runs/                      # all studies (copy it while nothing writes)
```

## What happens when something goes wrong

- **The process crashes** (OOM, an exception). Docker restarts it up to 5 times (`restart: on-failure:5`).
  The trainer always runs with `--resume`, so it picks up from `latest.pt`, which is saved at
  every evaluation and on exit.
- **The server reboots or loses power.** Docker starts on boot and brings the trainer back. With
  a power cut you lose up to `eval_every` updates of progress.
- **You stop it** (`make stop` / `docker compose stop`). Docker sends SIGTERM. The trainer
  finishes cleanly, saves `latest.pt` and is not restarted. `make up` resumes it.
- **The run finishes.** Exit code 0, so Docker does not restart it. Running `make up` again with the
  same `RUN_NAME` is a no-op ("nothing to do").
- **Training diverges** (NaN loss). The trainer raises and Docker retries it a few times. Check
  `make logs`, change the config and start a new `RUN_NAME`.
- **Optuna storage fails** (study deleted in the dashboard, locked `optuna.db`). Training logs a
  warning and keeps going. Only the Optuna record of that run is lost.

Resuming restores the model, the optimizer and the update counter. The self-play ghost pool is
rebuilt within a few updates. Every restart opens a new Optuna trial on the same run directory.

## Troubleshooting

**`KeyError: 'Record does not exist.'` in `make logs`.** The run's Optuna study or trial was
deleted in optuna-dashboard while the run was writing to it (smoke runs never appear there, so
anything in the dashboard is real). Older code crashed on this. Now it is a warning
(`Optuna report failed ... training continues`), and the run keeps going untracked in Optuna. `make
restart` gives it a fresh trial. Don't delete studies or trials while anything is running.

**`make gpu` prints "NO GPU".**
1. `nvidia-smi` on the host. If it fails, (re)install the driver (step 1) and reboot.
2. `docker run --rm --gpus all ubuntu nvidia-smi`. If it fails, the Container Toolkit isn't wired
   into Docker: `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
3. `make check` points at the failing piece.

**`make tune` says "train is running", or `make up` says "sas-tune is running".** Only one of
train / tune / validate may run at a time, because they share the GPU and the CPU env workers. Stop
the other one first: `make stop`, `make tune-stop` or `make validate-stop`.

**`ssh sas`: `Connection timed out` vs `Connection refused`.** Timed out: nothing answers at that
address (wrong or changed IP, server off or asleep, a different network). Refused: the server is
reachable but SSH isn't listening (`sudo systemctl status ssh`, `sudo ufw status`). More in
[docs/remote-access.md](../docs/remote-access.md#troubleshooting).

**Did it restart, and how often?**

```bash
docker inspect -f '{{.RestartCount}}' $(docker compose ps -q train)
make ps                                   # "Up 3 minutes" after a long run means a recent restart
docker compose logs train | grep -E "resuming from|Traceback|Error"   # not `make logs`: it follows forever
```

After 5 failed restarts Docker gives up. Read the traceback in `make logs`, fix the cause and
`make up`.

**`permission denied ... /var/run/docker.sock`.** Your user isn't in the `docker` group yet in this
session. Log out and back in after `usermod -aG docker` (step 2).

## Reproducibility

- **Dependencies.** `pyproject.toml` pins the direct dependencies to the versions they were tested
  with, and `uv.lock` pins everything else. The image installs exactly that set (`uv sync --frozen`).
  To upgrade, edit the pins, run `uv lock`, test and commit.
- **Code version.** The image is tagged `super-auto-strat:<commit>`, and every run records
  `git_commit` in Optuna (`-dirty` if the checkout had local edits). Deploy committed code only.
  Start containers through `make` (`make up`, `make tune`, …): it exports `GIT_COMMIT`, `UID` and
  `GID`. A bare `docker compose run` tags runs `git_commit=unknown` and uses the `:latest` image.
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
