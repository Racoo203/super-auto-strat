"""Look back at recorded games (runs/<run>/replays/update_XXXX.jsonl, written during training).

    python -m agents.ppo.replay runs/baseline/replays/update_0099.jsonl              # summary of all games
    python -m agents.ppo.replay runs/baseline/replays/update_0099.jsonl -e 0 --play  # step through game 0
    python -m agents.ppo.replay runs/baseline                                         # latest replay file of a run

In a notebook:  `from agents.ppo.replay import load_replays, summarize, show_episode`.
"""

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

from .evaluate import EpisodeRecord, load_replays


def latest_replay(run_dir: str | Path) -> Path:
    files = sorted((Path(run_dir) / "replays").glob("update_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no replays in {run_dir}/replays")
    return files[-1]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return latest_replay(path) if path.is_dir() else path


def action_kind(label: str) -> str:
    """'buy shop pet 2 onto team pet 1' -> 'buy pet (stack)'; strips the slot numbers."""
    if label.startswith("buy shop pet"):
        return "buy pet (stack)" if "onto" in label else "buy pet"
    return re.sub(r"\s*\d+", "", label).replace(" to position", "").replace("shop ", "").replace("team ", "").strip()


def summarize(rec: EpisodeRecord) -> str:
    """Game result, round-by-round battle outcomes, how actions were spent, and the final team."""
    result = "WON" if rec.won else ("TRUNCATED" if rec.truncated else "LOST")
    lines = [
        f"seed {rec.seed}: {result} | {rec.wins}W {rec.losses}L {rec.draws}D | lives {rec.lives} | "
        f"{rec.rounds} rounds | {len(rec.steps)} actions | return {rec.ret:+.2f}"
    ]
    rounds = [(s.round, s.battle["outcome"][0].upper()) for s in rec.steps if s.battle]
    lines.append("  rounds: " + " ".join(f"{r}{o}" for r, o in rounds))
    per_round = Counter(s.round for s in rec.steps)
    lines.append("  actions/round: " + " ".join(str(per_round[r]) for r in sorted(per_round)))
    kinds = Counter(action_kind(s.label) for s in rec.steps)
    lines.append("  actions: " + ", ".join(f"{k} x{n}" for k, n in kinds.most_common()))
    invalid = sum(s.invalid for s in rec.steps)
    if invalid:
        lines.append(f"  invalid actions: {invalid}")
    if rec.steps and rec.steps[-1].frame:
        last_team = _team_block(rec.steps[-1].frame) or "(no pets)"
        lines.append("  team before last battle:\n" + "\n".join("    " + l for l in last_team.splitlines()))
    return "\n".join(lines)


def _team_block(frame: str) -> str:
    """The '[+] Agent [+]' block of a frame (up to the policy overlay)."""
    if "[+] Agent [+]" not in frame:
        return ""
    block = frame.split("[+] Agent [+]", 1)[1].split("\n\nPolicy", 1)[0]
    return "\n".join(l for l in block.strip("\n").splitlines() if "(empty)" not in l)


def summarize_file(records: list[EpisodeRecord]) -> str:
    n = len(records)
    won = sum(r.won for r in records)
    mean = lambda f: sum(f(r) for r in records) / max(1, n)  # noqa: E731
    head = (
        f"{n} games | win rate {won / max(1, n):.0%} | mean wins {mean(lambda r: r.wins):.1f} | "
        f"mean rounds {mean(lambda r: r.rounds):.1f} | mean actions/game {mean(lambda r: len(r.steps)):.0f}"
    )
    return head + "\n\n" + "\n\n".join(f"[{i}] {summarize(r)}" for i, r in enumerate(records))


def frames(rec: EpisodeRecord) -> list[str]:
    return [s.frame for s in rec.steps if s.frame] + ([rec.final_frame] if rec.final_frame else [])


def show_episode(rec: EpisodeRecord, delay: float = 0.0, battles_only: bool = False, out=None) -> None:
    """Print every frame of a recorded game (in a notebook, frames redraw in place).

    battles_only: only show the frame right after each battle (one per round).
    """
    clear = _notebook_clear()
    shown = frames(rec)
    if battles_only:
        # the frame after a battle (the next decision's frame) shows that battle's result and log
        after = [nxt.frame for s, nxt in zip(rec.steps, rec.steps[1:]) if s.battle]
        shown = [f for f in after + [rec.final_frame] if f]
    for i, frame in enumerate(shown):
        if clear and delay:
            clear(wait=True)
        print(f"[frame {i + 1}/{len(shown)}]\n{frame}", file=out or sys.stdout, flush=True)
        if delay:
            time.sleep(delay)


def _notebook_clear():
    try:
        from IPython import get_ipython
        from IPython.display import clear_output

        return clear_output if get_ipython() is not None and "IPKernelApp" in get_ipython().config else None
    except ImportError:
        return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="replay .jsonl file, or a run directory (uses its latest replay)")
    ap.add_argument("-e", "--episode", type=int, help="game index in the file")
    ap.add_argument("--play", action="store_true", help="print every frame of --episode")
    ap.add_argument("--battles-only", action="store_true", help="with --play: one frame per round")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between frames with --play")
    args = ap.parse_args(argv)

    path = resolve(args.path)
    records = load_replays(path)
    print(f"{path}\n")
    if args.episode is None:
        print(summarize_file(records))
        return
    rec = records[args.episode]
    if args.play:
        show_episode(rec, args.delay, args.battles_only)
    print(summarize(rec))


if __name__ == "__main__":
    main()
