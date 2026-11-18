"""Save / load training checkpoints: model + optimizer + config + where training was."""

from pathlib import Path

import torch

from .config import PPOConfig, get_device
from .network import ActorCritic


def save_checkpoint(path: str | Path, model: ActorCritic, optimizer, cfg: PPOConfig, update: int, **extra) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_states = model.trunk[0].in_features
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "cfg": cfg.to_dict(),
            "update": update,
            "n_states": n_states,
            "n_actions": model.policy_head.out_features,
            **extra,
        },
        path.with_suffix(".tmp"),
    )
    path.with_suffix(".tmp").replace(path)  # atomic: a crash mid-save never leaves a broken file
    return path


def load_checkpoint(path: str | Path, device=None) -> tuple[ActorCritic, PPOConfig, dict]:
    """Rebuild the model from a checkpoint. Returns (model, cfg, raw checkpoint dict)."""
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = PPOConfig.from_dict(ckpt["cfg"])
    model = ActorCritic(ckpt["n_states"], ckpt["n_actions"], hidden=cfg.hidden).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt
