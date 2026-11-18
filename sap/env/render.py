"""Plain-text frames of the environment, for `render_mode="ansi"/"human"`, live watching and replays.

A frame is just a string, so it can be printed, stored in a replay file or redrawn in a notebook.
"""

from sap import config
from sap.game import Game

from .actions import describe_action

RULE = "=" * 60


def render_policy(probs: list[tuple[int, float]], value: float | None = None, chosen: int | None = None) -> str:
    """The policy overlay: top-k (action, probability) pairs and V(s). `chosen` is marked with '>'."""
    lines = ["Policy" + (f"  V(s) = {value:+.2f}" if value is not None else "")]
    for action, p in probs:
        mark = ">" if action == chosen else " "
        bar = "#" * round(p * 20)
        lines.append(f" {mark} {p:6.1%} {bar:<20} {describe_action(action)}")
    return "\n".join(lines)


def render_frame(
    game: Game,
    *,
    shop_actions: int = 0,
    max_shop_actions: int | None = None,
    last_action: int | None = None,
    last_reward: float | None = None,
    invalid: bool = False,
    battle: dict | None = None,
    policy: str | None = None,
) -> str:
    """One text frame: status line, last action, last battle (if a turn just ended), store, team.

    `battle` is a dict with "outcome", "opponent" (str of the opponent team) and "log" (list of
    lines), as produced by SapEnv. `policy` is a pre-rendered overlay from `render_policy`.
    """
    g = game
    used = f"{shop_actions}/{max_shop_actions}" if max_shop_actions else str(shop_actions)
    lines = [
        RULE,
        f"Round {g.round} | Lives {g.lives} | Wins {g.wins}/{config.WIN_TARGET} | "
        f"Losses {g.losses} | Draws {g.draws} | Shop actions {used}",
        RULE,
    ]
    if last_action is not None:
        tag = "  (INVALID)" if invalid else ""
        reward = f"  reward {last_reward:+.2f}" if last_reward is not None else ""
        lines.append(f"Last action: {describe_action(last_action)}{tag}{reward}")
    if battle:
        lines.append(f"--- Battle: {battle['outcome'].upper()} ---")
        if battle.get("opponent"):
            lines.append(battle["opponent"])
        lines += [f"  {line}" for line in battle.get("log", [])]
        lines.append("-" * 30)
    if g.is_over:
        lines.append("GAME OVER: " + ("VICTORY" if g.won else "DEFEAT"))
    else:
        lines += [str(g.store), "", str(g.team)]
    if policy:
        lines += ["", policy]
    return "\n".join(lines)
