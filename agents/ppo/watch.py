"""Watch a trained policy play, live, as text: state, its top actions with probabilities, V(s).

    python -m agents.ppo.watch runs/baseline/checkpoints/latest.pt --delay 0.5
    python -m agents.ppo.watch runs/baseline --greedy --episodes 3      # a run dir -> best.pt / latest.pt
    python -m agents.ppo.watch --random                                 # untrained policy, for comparison

In a notebook:
    from agents.ppo.watch import watch_in_notebook
    watch_in_notebook("runs/baseline", delay=0.3)      # or pass a model object
"""

import argparse
import sys
import time
from pathlib import Path

from sap.env import OBS_SIZE, ACTIONS, SapEnv

from .checkpoints import load_checkpoint
from .config import get_device
from .evaluate import EpisodeRecord, play_episode
from .network import ActorCritic
from .replay import summarize


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():  # a run dir or its checkpoints dir
        ck = path / "checkpoints" if (path / "checkpoints").is_dir() else path
        for name in ("best.pt", "latest.pt"):
            if (ck / name).exists():
                return ck / name
        raise FileNotFoundError(f"no best.pt/latest.pt in {ck}")
    return path


def load_policy(source) -> ActorCritic:
    """`source`: an ActorCritic, a checkpoint path, a run dir, or None (random untrained policy)."""
    if isinstance(source, ActorCritic):
        return source
    if source is None:
        return ActorCritic(OBS_SIZE, len(ACTIONS), hidden=(64,)).to(get_device())
    model, _, ckpt = load_checkpoint(resolve_checkpoint(source))
    print(f"loaded {resolve_checkpoint(source)} (update {ckpt['update']}, run {ckpt.get('run')})")
    return model


def watch(
    source=None,
    *,
    episodes: int = 1,
    seed: int | None = None,
    greedy: bool = False,
    delay: float = 0.5,
    battles_only: bool = False,
    clear: bool = True,
    env_kwargs: dict | None = None,
    printer=None,
) -> list[EpisodeRecord]:
    """Play `episodes` games, printing each frame as the policy decides. Returns the records."""
    model = load_policy(source)
    env = SapEnv(render_mode="ansi", **(env_kwargs or {}))
    printer = printer or _terminal_printer(clear)
    records = []
    for ep in range(episodes):
        # battles_only: show only the first decision of each round (its frame includes the battle
        # that just happened) and the final frame. env.shop_actions is 0 at the start of a round.
        def on_step(frame, step, ep=ep):
            if battles_only and step is not None and env.shop_actions != 0:
                return
            printer(f"[game {ep + 1}/{episodes}]\n{frame}")
            if delay:
                time.sleep(delay)

        game_seed = None if seed is None else seed + ep
        rec = play_episode(model, env, seed=game_seed, greedy=greedy, record_frames=False, on_step=on_step)
        records.append(rec)
        printer(summarize(rec))
    return records


def _terminal_printer(clear: bool):
    def show(text: str) -> None:
        if clear and sys.stdout.isatty():
            sys.stdout.write("\x1b[2J\x1b[H")  # clear screen, cursor home
        print(text, flush=True)

    return show


def watch_in_notebook(source=None, **kwargs) -> list[EpisodeRecord]:
    """Same as `watch`, redrawing one cell output in place."""
    from IPython.display import clear_output

    def show(text: str) -> None:
        clear_output(wait=True)
        print(text, flush=True)

    return watch(source, printer=show, **kwargs)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", help="checkpoint .pt or run directory")
    ap.add_argument("--random", action="store_true", help="use an untrained policy")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--greedy", action="store_true", help="argmax instead of sampling")
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between frames")
    ap.add_argument("--battles-only", action="store_true", help="one frame per round")
    ap.add_argument("--no-clear", action="store_true", help="scroll instead of redrawing")
    args = ap.parse_args(argv)
    if not args.checkpoint and not args.random:
        ap.error("give a checkpoint / run dir, or --random")
    watch(
        None if args.random else args.checkpoint, episodes=args.episodes, seed=args.seed,
        greedy=args.greedy, delay=args.delay, battles_only=args.battles_only, clear=not args.no_clear,
    )


if __name__ == "__main__":
    main()
