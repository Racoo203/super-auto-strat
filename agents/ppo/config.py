"""Hyperparameters and device selection, in one place so the notebook and a script share them."""

from dataclasses import asdict, dataclass

import torch


@dataclass
class PPOConfig:
    # --- parallel actors (the "for actor = 1..N" loop in the paper) ---
    n_envs: int = 8          # N: number of environments stepped in lockstep
    n_steps: int = 128       # T: steps collected per env per update -> batch = N * T
    async_envs: bool = True  # True: one process per env (real parallelism). False: in-process.

    # --- returns / advantages ---
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # --- optimisation ---
    lr: float = 2.5e-4
    n_epochs: int = 4        # K: passes over the collected batch
    minibatch_size: int = 256  # M <= N * T
    clip_eps: float = 0.2
    max_grad_norm: float = 0.5

    # --- loss weights (paper: L = L_clip - c1 * L_vf + c2 * S) ---
    value_coef: float = 0.5    # c1
    entropy_coef: float = 0.01  # c2
    normalize_advantage: bool = True  # per-minibatch (adv - mean) / std, before the clipped loss

    n_updates: int = 1000

    # --- network ---
    hidden: tuple[int, ...] = (2048, 1024, 512)  # trunk layer sizes of ActorCritic

    # --- opponents / self-play (agents/ppo/selfplay.py, sap/ghosts.py) ---
    opponent: str = "ghosts"     # "ghosts": fight teams the policies built (self-play); "random": random teams
    n_policies: int = 1          # 1: shared-policy self-play; P > 1: P agents, env i played by policy i % P
    opponents: str = "others"    # P > 1: "others" = only other policies' ghosts, "all" = anyone's
    ghost_recent: int = 32       # newest ghosts kept per round
    ghost_history: int = 64      # older ghosts kept per round (reservoir: a league of past policies)
    ghost_history_prob: float = 0.2    # chance an opponent comes from the older reservoir
    random_opponent_prob: float = 0.05  # chance of a random team anyway (a fixed reference point)

    # --- tracking / evaluation (agents/ppo/train.py) ---
    seed: int = 0
    eval_every: int = 25        # updates between evaluations (0 = never); also saves replays
    # games per evaluation (each kind: vs random teams, and with ghosts vs the pool). 50, not 20:
    # with 20 games eval/battle_win_rate (the Optuna objective) is too noisy to rank trials by.
    eval_episodes: int = 50
    eval_greedy: bool = False   # evaluate with argmax actions instead of sampling
    checkpoint_every: int = 50  # updates between numbered checkpoints (latest.pt: each eval + at the end)
    keep_checkpoints: int = 3  # numbered update_XXXX.pt kept per policy (0 = keep all)
    runs_dir: str = "runs"      # runs/<run name>/{checkpoints,replays,tb,metrics.jsonl}

    @property
    def batch_size(self) -> int:
        return self.n_envs * self.n_steps

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PPOConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "hidden" in known:
            known["hidden"] = tuple(known["hidden"])
        return cls(**known)


def get_device() -> torch.device:
    """CUDA/MPS/etc. if available (your PC), otherwise CPU (your laptop)."""
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    return torch.device("cpu")
