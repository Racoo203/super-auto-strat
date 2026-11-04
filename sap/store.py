import random

from sap import config
from sap.errors import ActionError
from sap.items import Item, random_item
from sap.pets import Pet, random_pet
from sap.team import Team


class Store:
    """Shop display and credits. Never does I/O: actions return values or raise ActionError."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.credits = 0
        self.pet_slots: list[Pet | None] = []
        self.item_slots: list[Item | None] = []

    def new_visit(self) -> None:
        """Start a store visit: credits reset (no carry-over) and a fresh display."""
        self.credits = config.CREDITS_PER_TURN
        self._roll()

    def _roll(self) -> None:
        self.pet_slots = [random_pet(self.rng) for _ in range(config.STORE_PET_SLOTS)]
        self.item_slots = [random_item(self.rng) for _ in range(config.STORE_ITEM_SLOTS)]

    def _spend(self, amount: int) -> None:
        if self.credits < amount:
            raise ActionError(f"Not enough credits (need {amount}, have {self.credits})")
        self.credits -= amount

    @staticmethod
    def _take_slot(slots: list, idx: int, kind: str):
        if not 0 <= idx < len(slots):
            raise ActionError(f"No {kind} slot {idx + 1}")
        if slots[idx] is None:
            raise ActionError(f"{kind.capitalize()} slot {idx + 1} is already sold")
        return slots[idx]

    def buy_pet(self, slot: int, team: Team) -> Pet:
        pet = self._take_slot(self.pet_slots, slot, "pet")
        if team.is_full:
            raise ActionError("Team is full")
        self._spend(config.PET_COST)
        team.add(pet)
        self.pet_slots[slot] = None
        return pet

    def buy_item(self, slot: int, team: Team, pet_idx: int) -> Item | None:
        """Give the item to team[pet_idx]. Returns the item it replaced, if any."""
        item = self._take_slot(self.item_slots, slot, "item")
        pet = team[pet_idx]  # validates the index before spending
        self._spend(config.ITEM_COST)
        replaced, pet.item = pet.item, item
        self.item_slots[slot] = None
        return replaced

    def reroll(self) -> None:
        self._spend(config.REROLL_COST)
        self._roll()

    def sell(self, team: Team, idx: int) -> Pet:
        pet = team.remove(idx)
        self.credits += config.SELL_PRICE
        return pet

    def __str__(self):
        lines = [f"[-] Store [-]  Credits: {self.credits}", "  Pets:"]
        for i, pet in enumerate(self.pet_slots):
            text = f"{pet}  ({config.PET_COST}c)" if pet else "[SOLD]"
            lines.append(f"    [{i + 1}] {text}")
        lines.append("  Items:")
        for i, item in enumerate(self.item_slots):
            text = f"{item}  ({config.ITEM_COST}c)" if item else "[SOLD]"
            lines.append(f"    [{i + 1}] {text}")
        return "\n".join(lines)
