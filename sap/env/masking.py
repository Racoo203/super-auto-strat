"""Which actions are legal: a mirror of the store rules that never touches the game (kept in sync
by tests), minus actions that only undo the previous one."""

import numpy as np

from sap import config
from sap.game import Game

from .actions import ACTIONS


def undoes_last(action: int, last_action: int | None) -> bool:
    """True if `action` would just revert `last_action` (the previous shop action this round)."""
    if last_action is None:
        return False
    kind, a, b = ACTIONS[action]
    last_kind, last_a, last_b = ACTIONS[last_action]
    if kind == "move" and last_kind == "move":
        if (a, b) == (last_b, last_a):  # insert-move b->a is undone by a->b
            return True
        # between neighbours a move is a swap, so repeating it also undoes it
        return (a, b) == (last_a, last_b) and abs(a - b) == 1
    if kind in ("freeze_pet", "freeze_item"):
        return (kind, a) == (last_kind, last_a)  # second toggle unfreezes
    return False


def is_legal(game: Game | None, action: int, last_action: int | None = None) -> bool:
    if game is None or game.is_over:
        return False
    if undoes_last(action, last_action):
        return False
    kind, a, b = ACTIONS[action]
    store, team = game.store, game.team
    n = len(team)
    if kind == "end_turn":
        return True
    if kind == "reroll":
        return store.credits >= config.REROLL_COST
    if kind == "sell":
        return a < n
    if kind == "move":
        return a < n and b < n
    if kind == "combine":
        return a < n and b < n and team[a].name == team[b].name and not team[b].max_level
    if kind in ("buy_pet", "freeze_pet"):
        pet = store.pet_slots[a] if a < len(store.pet_slots) else None
        if pet is None:
            return False
        if kind == "freeze_pet":
            return True
        if store.credits < config.PET_COST:
            return False
        if b is None:
            return not team.is_full
        return b < n and team[b].name == pet.name and not team[b].max_level
    if kind in ("buy_item", "freeze_item"):
        slot = store.item_slots[a] if a < len(store.item_slots) else None
        if slot is None:
            return False
        if kind == "freeze_item":
            return True
        if store.credits < slot.price:
            return False
        return b < n if slot.item.targeted else b == 0
    return False


def action_mask(game: Game | None, last_action: int | None = None) -> np.ndarray:
    return np.array([is_legal(game, a, last_action) for a in range(len(ACTIONS))], dtype=bool)
