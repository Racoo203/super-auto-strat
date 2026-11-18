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


def prune_checkpoints(ckpt_dir: str | Path, keep: int = 3, stem_prefix: str = "update_") -> None:
    """Delete numbered checkpoints (e.g. update_0049.pt) beyond the newest `keep`, per policy
    suffix (update_0049.pt vs update_0049_p1.pt are pruned separately). latest.pt / best.pt use a
    different prefix and are never touched. keep <= 0 keeps everything (no-op)."""
    if keep <= 0:
        return
    ckpt_dir = Path(ckpt_dir)
    by_suffix: dict[str, list[Path]] = {}
    for p in sorted(ckpt_dir.glob(f"{stem_prefix}*.pt")):
        suffix = p.stem[len(stem_prefix) + 4:]  # "update_0049_p1" -> "_p1"; "update_0049" -> ""
        by_suffix.setdefault(suffix, []).append(p)
    for paths in by_suffix.values():
        for old in paths[:-keep]:
            old.unlink(missing_ok=True)
