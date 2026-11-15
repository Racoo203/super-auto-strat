"""The action table: every shop action as (kind, a, b), its label, and how to apply it to a Game."""

from sap import config
from sap.game import Game

T = config.TEAM_SIZE
P = config.MAX_STORE_PET_SLOTS
I = config.MAX_STORE_ITEM_SLOTS


def _build_actions() -> list[tuple]:
    acts: list[tuple] = []
    acts += [("buy_pet", s, None) for s in range(P)]  # into a free team slot
    acts += [("buy_pet", s, t) for s in range(P) for t in range(T)]  # stack onto team[t]
    acts += [("buy_item", s, t) for s in range(I) for t in range(T)]  # untargeted items: t = 0
    acts += [("sell", t, None) for t in range(T)]
    acts += [("combine", a, b) for a in range(T) for b in range(T) if a != b]
    acts += [("move", a, b) for a in range(T) for b in range(T) if a != b]
    acts += [("freeze_pet", s, None) for s in range(P)]
    acts += [("freeze_item", s, None) for s in range(I)]
    acts += [("reroll", None, None), ("end_turn", None, None)]
    return acts


ACTIONS = _build_actions()


def describe_action(action: int) -> str:
    kind, a, b = ACTIONS[int(action)]

    def n(i):
        return i + 1 if i is not None else None  # 1-based, as on screen

    return {
        "buy_pet": f"buy shop pet {n(a)}" + (f" onto team pet {n(b)}" if b is not None else ""),
        "buy_item": f"buy shop item {n(a)} for team pet {n(b)}",
        "sell": f"sell team pet {n(a)}",
        "combine": f"stack team pet {n(a)} onto team pet {n(b)}",
        "move": f"move team pet {n(a)} to position {n(b)}",
        "freeze_pet": f"freeze/unfreeze shop pet {n(a)}",
        "freeze_item": f"freeze/unfreeze shop item {n(a)}",
    }.get(kind, kind.replace("_", " "))


def apply_action(game: Game, action: int) -> None:
    """Apply a shop action (anything but end_turn). Raises ActionError if the game refuses it."""
    kind, a, b = ACTIONS[action]
    if kind == "buy_pet":
        game.buy_pet(a, b)
    elif kind == "buy_item":
        item = game.store.item_slots[a].item
        game.buy_item(a, b if item.targeted else None)
    elif kind == "sell":
        game.sell(a)
    elif kind == "combine":
        game.combine(a, b)
    elif kind == "move":
        game.move(a, b)
    elif kind == "freeze_pet":
        game.freeze_pet(a)
    elif kind == "freeze_item":
        game.freeze_item(a)
    elif kind == "reroll":
        game.reroll()
