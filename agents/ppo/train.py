"""The PPO training loop (formerly inline in PPO.ipynb), with tracking.

    from agents.ppo import PPOConfig, train
    result = train(PPOConfig(n_updates=300), "baseline")

Per update it logs (TensorBoard + metrics.jsonl, see experiments.py):
    train/*   episodes finished this update: ret, length, win_rate, wins, losses, lives, rounds
    loss/*    policy, value, entropy, approx_kl, clip_frac, explained_var
    perf/sps  environment steps per second
Every cfg.eval_every updates it plays cfg.eval_episodes games on fixed seeds (eval/*), saves them
as runs/<run>/replays/update_XXXX.jsonl, reports eval/win_rate to Optuna and saves checkpoints.
If training crashes or is interrupted, latest.pt is still written and the trial is marked FAIL.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .advantages import compute_gae
from .buffer import RolloutBuffer
from .checkpoints import save_checkpoint
from .config import PPOConfig, get_device
from .envs import make_vec_env
from .evaluate import evaluate, save_replays, summarize_records
from .experiments import Run
from .losses import entropy_bonus, ppo_clip_loss, value_loss
from .network import ActorCritic
from .rollout import collect_rollout
from .tracking import EpisodeTracker, summarize_episodes


@dataclass
class TrainResult:
    run_name: str
    run_dir: Path
    best_eval_win_rate: float | None
    model: ActorCritic


def ppo_update(model, optimizer, buffer: RolloutBuffer, cfg: PPOConfig) -> dict[str, float]:
    """K epochs of minibatch updates on the collected batch. Returns mean loss diagnostics."""
    stats: dict[str, list[float]] = {k: [] for k in ("policy", "value", "entropy", "approx_kl", "clip_frac")}
    for _ in range(cfg.n_epochs):
        for mb in buffer.minibatches(cfg.minibatch_size):
            dist, new_values = model.distribution(mb["obs"], mb["masks"])
            new_log_probs = dist.log_prob(mb["actions"])
            entropy = dist.entropy()

            pl = ppo_clip_loss(new_log_probs, mb["log_probs"], mb["advantages"], cfg.clip_eps)
            vl = value_loss(new_values, mb["returns"])
            ent = entropy_bonus(entropy)
            loss = pl + cfg.value_coef * vl - cfg.entropy_coef * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio = new_log_probs - mb["log_probs"]
                ratio = log_ratio.exp()
                stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())  # k3 estimator
                stats["clip_frac"].append(((ratio - 1).abs() > cfg.clip_eps).float().mean().item())
            stats["policy"].append(pl.item())
            stats["value"].append(vl.item())
            stats["entropy"].append(ent.item())

    out = {f"loss/{k}": float(np.mean(v)) for k, v in stats.items()}
    y, v = buffer.returns.flatten(), buffer.values.flatten()
    var_y = torch.var(y).item()
    out["loss/explained_var"] = float("nan") if var_y == 0 else 1 - torch.var(y - v).item() / var_y
    return out


def train(
    cfg: PPOConfig,
    run_name: str | None = None,
    *,
    storage: str | None = None,
    trial=None,
    prune: bool = False,
    env_kwargs: dict | None = None,
    log_every: int = 1,
    log_fn: Callable[[str], None] | None = print,
) -> TrainResult:
    """Train PPO and track the run. See the module docstring.

    storage: Optuna storage URL (default sqlite:///<runs_dir>/optuna.db).
    trial / prune: used by tune.py; a normal run creates its own trial.
    """
    device = get_device()
    torch.manual_seed(cfg.seed)
    env_kwargs = env_kwargs or {}

    with Run(cfg, run_name, storage=storage, trial=trial) as run:
        run.trial.set_user_attr("device", str(device))
        envs, obs, mask = make_vec_env(cfg.n_envs, async_envs=cfg.async_envs, seed=cfg.seed, **env_kwargs)
        n_states = envs.single_observation_space.shape[0]
        n_actions = int(envs.single_action_space.n)
        model = ActorCritic(n_states, n_actions, hidden=cfg.hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
        buffer = RolloutBuffer(cfg.n_steps, cfg.n_envs, n_states, n_actions, device)
        tracker = EpisodeTracker(cfg.n_envs)
        ckpt_dir = run.dir / "checkpoints"
        update = -1

        if log_fn:
            log_fn(f"run '{run.name}' -> {run.dir}  (trial #{run.trial.number}, {device})")
        try:
            for update in range(cfg.n_updates):
                t0 = time.time()
                obs, mask, last_value = collect_rollout(envs, model, buffer, obs, mask, device, tracker)
                buffer.advantages, buffer.returns = compute_gae(
                    buffer.rewards, buffer.values, buffer.dones, last_value, cfg.gamma, cfg.gae_lambda
                )
                metrics = ppo_update(model, optimizer, buffer, cfg)
                metrics["perf/sps"] = cfg.batch_size / (time.time() - t0)
                metrics.update(summarize_episodes(tracker.pop(), prefix="train/"))

                last = update == cfg.n_updates - 1
                if cfg.eval_every and ((update + 1) % cfg.eval_every == 0 or last):
                    records = evaluate(model, cfg.eval_episodes, seed=cfg.seed, greedy=cfg.eval_greedy, env_kwargs=env_kwargs)
                    save_replays(records, run.dir / "replays" / f"update_{update:04d}.jsonl")
                    ev = summarize_records(records)
                    metrics.update(ev)
                    save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, cfg, update, run=run.name)
                    if run.report(update, ev["eval/win_rate"], prune=prune):
                        save_checkpoint(ckpt_dir / "best.pt", model, optimizer, cfg, update, run=run.name, **ev)
                if cfg.checkpoint_every and (update + 1) % cfg.checkpoint_every == 0:
                    save_checkpoint(ckpt_dir / f"update_{update:04d}.pt", model, optimizer, cfg, update, run=run.name)

                run.log(update, metrics)
                if log_fn and (update % log_every == 0 or "eval/win_rate" in metrics):
                    log_fn(_progress_line(update, metrics))
        finally:
            envs.close()
            if update >= 0:  # whatever happened, keep the latest weights
                save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, cfg, update, run=run.name)

    return TrainResult(run.name, run.dir, run.best_value, model)


def _progress_line(update: int, m: dict) -> str:
    parts = [f"update {update:4d}"]
    if "train/episodes" in m:
        parts.append(
            f"{int(m['train/episodes'])} eps, ret {m['train/ret']:.2f}, win {m['train/win_rate']:.0%}, "
            f"wins {m['train/wins']:.1f}, len {m['train/length']:.0f}"
        )
    parts.append(f"ent {m['loss/entropy']:.2f}, kl {m['loss/approx_kl']:.1e}, {m['perf/sps']:.0f} sps")
    if "eval/win_rate" in m:
        parts.append(f"EVAL win {m['eval/win_rate']:.0%} wins {m['eval/wins']:.1f} rounds {m['eval/rounds']:.1f}")
    return " | ".join(parts)
