import random
from dataclasses import dataclass
from typing import Callable

import sap.abilities  # noqa: F401  (registers pet abilities)
from sap import config
from sap.errors import ActionError
from sap.events import Context, Event, Trigger
from sap.items import ITEM_POOL, Item, feed, random_item
from sap.perks import Perk
from sap.pets import Pet, make_pet, pets_of_tier, random_pet
from sap.team import Team


@dataclass(eq=False)  # compared by identity, like pets (frozen-slot tracking)
class ShopItem:
    item: Item
    price: int

    def _ice(self, obj) -> str:
        return " *FROZEN*" if self.is_frozen(obj) else ""

    def __str__(self):
        return f"{self.item}  ({self.price}c)"


class Store:
    """Shop display and credits. Never does I/O: actions return values or raise ActionError.

    `round` and `max_tier` (set by the game each round) decide how many slots the shop shows and
    which tiers it rolls. Frozen slots survive rerolls and the next visit. `log` holds the ability
    messages produced by the most recent action.
    """

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.credits = 0
        self.round = 1
        self.max_tier = 1
        self.frozen: set = set()  # frozen shop pets / ShopItems (by identity)
        self.pet_slots: list[Pet | None] = []
        self.item_slots: list[ShopItem | None] = []
        self.pet_bonus = (0, 0)  # Canned Food: stats added to every shop pet, now and later
        self.lost_last_battle = False  # read by Snail
        self.log: list[str] = []

    def new_visit(self) -> None:
        """Start a store visit: credits reset (no carry-over) and a fresh display."""
        self.credits = config.CREDITS_PER_TURN
        self._roll()

    @property
    def pet_slot_count(self) -> int:
        return config.slots_for_round(config.STORE_PET_SLOTS, self.round)

    @property
    def item_slot_count(self) -> int:
        return config.slots_for_round(config.STORE_ITEM_SLOTS, self.round)

    def _roll(self) -> None:
        """Refill the display. Frozen slots stay (moved to the front); the rest are re-rolled."""
        kept_pets = [p for p in self.pet_slots if p is not None and p in self.frozen]
        kept_items = [i for i in self.item_slots if i is not None and i in self.frozen]
        self.frozen = set(kept_pets) | set(kept_items)
        self.pet_slots = kept_pets + [
            self._new_shop_pet() for _ in range(self.pet_slot_count - len(kept_pets))
        ]
        self.item_slots = kept_items + [
            ShopItem(item, item.cost)
            for item in (
                random_item(self.rng, self.max_tier)
                for _ in range(self.item_slot_count - len(kept_items))
            )
        ]

    def _new_shop_pet(self, tier: int | None = None) -> Pet:
        if tier is None:
            pet = random_pet(self.rng, self.max_tier)
        else:
            pet = make_pet(self.rng.choice(pets_of_tier(tier)))
        pet.buff(*self.pet_bonus)
        return pet

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

    # --- hooks used by abilities and food ---------------------------------------
    def stock(self, item_name: str, price: int) -> None:
        """Add an extra item to the shop (Worm, Pigeon, Cow). Ignored when the shop is full."""
        if len(self.item_slots) < config.MAX_STORE_ITEM_SLOTS:
            self.item_slots.append(ShopItem(ITEM_POOL[item_name], price))

    def add_level_up_pet(self) -> Pet | None:
        """A level-up adds one pet from the tier above the unlocked one (max tier 6) to the shop."""
        if len(self.pet_slots) >= config.MAX_STORE_PET_SLOTS:
            return None
        pet = self._new_shop_pet(tier=min(config.MAX_TIER, self.max_tier + 1))
        self.pet_slots.append(pet)
        return pet

    def discount_items(self, amount: int) -> None:
        for slot in self.item_slots:
            if slot:
                slot.price = max(0, slot.price - amount)

    def add_pet_bonus(self, attack: int, health: int) -> None:
        self.pet_bonus = (self.pet_bonus[0] + attack, self.pet_bonus[1] + health)
        for pet in self.pet_slots:
            if pet:
                pet.buff(attack, health)

    # --- abilities -----------------------------------------------------------
    def run(self, team: Team, action: Callable[[Context], None]) -> list[str]:
        """Run `action(ctx)` in a shop context for `team`, then resolve every chain it caused."""
        self.log = []
        ctx = Context([team], self.rng, self.log, store=self)
        action(ctx)
        ctx.resolve()
        return self.log

    def trigger(self, team: Team, *events: Event) -> list[str]:
        """Run shop-phase events (start/end of turn...) against `team`."""
        return self.run(team, lambda ctx: [ctx.emit(e) for e in events])

    @staticmethod
    def _check_stack(target: Pet, other: Pet) -> None:
        if other is target:
            raise ActionError("Pick two different pets to stack")
        if other.name != target.name:
            raise ActionError(f"Can't stack {other.name} onto {target.name}: different pets")
        if target.max_level:
            raise ActionError(f"{target.name} is already max level")

    # --- freezing ------------------------------------------------------------------
    def _toggle_freeze(self, slots: list, idx: int, kind: str) -> bool:
        obj = self._take_slot(slots, idx, kind)
        if obj in self.frozen:
            self.frozen.discard(obj)
            return False
        self.frozen.add(obj)
        return True

    def freeze_pet(self, slot: int) -> bool:
        """Freeze (or unfreeze) a shop pet. Free. Returns True if it is now frozen."""
        return self._toggle_freeze(self.pet_slots, slot, "pet")

    def freeze_item(self, slot: int) -> bool:
        """Freeze (or unfreeze) a shop item. Free. Returns True if it is now frozen."""
        return self._toggle_freeze(self.item_slots, slot, "item")

    def is_frozen(self, obj) -> bool:
        return obj is not None and obj in self.frozen

    # --- actions ---------------------------------------------------------------
    def buy_pet(self, slot: int, team: Team, target: int | None = None) -> Pet:
        """Buy into a free team slot, or stack onto team[target] (same pet). Returns the team pet."""
        pet = self._take_slot(self.pet_slots, slot, "pet")
        if target is None:
            if team.is_full:
                raise ActionError("Team is full")
        else:
            owned = team[target]
            self._check_stack(owned, pet)
        self._spend(config.PET_COST)
        self.pet_slots[slot] = None
        self.frozen.discard(pet)

        def action(ctx: Context) -> None:
            nonlocal pet
            if target is None:
                team.add(pet)
                ctx.emit(Event(Trigger.SUMMONED, pet, team))
                ctx.emit(Event(Trigger.FRIEND_SUMMONED, pet, team))
            else:
                bought, pet = pet, team[target]
                if pet.merge(bought):
                    ctx.level_up(pet, team)
            ctx.emit(Event(Trigger.BUY, pet, team))
            ctx.emit(Event(Trigger.FRIEND_BOUGHT, pet, team))

        self.run(team, action)
        return pet

    def combine(self, team: Team, src: int, dst: int) -> Pet:
        """Stack team[src] onto team[dst] (same pet). Free, and cannot be undone."""
        target, other = team[dst], team[src]
        self._check_stack(target, other)

        def action(ctx: Context) -> None:
            team.pets.remove(other)
            if target.merge(other):
                ctx.level_up(target, team)

        self.run(team, action)
        return target

    def buy_item(self, slot: int, team: Team, pet_idx: int | None = None) -> Perk | None:
        """Buy an item. Targeted items need team[pet_idx]. Returns the perk it replaced, if any."""
        shop_item: ShopItem = self._take_slot(self.item_slots, slot, "item")
        item = shop_item.item
        pet = None
        if item.targeted:
            if pet_idx is None:
                raise ActionError(f"{item.name} must be given to a pet")
            pet = team[pet_idx]  # validates the index before spending
        self._spend(shop_item.price)
        self.item_slots[slot] = None
        self.frozen.discard(shop_item)
        replaced = pet.perk if (pet and item.perk) else None
        self.run(team, lambda ctx: feed(ctx, team, item, pet))
        return replaced

    def reroll(self) -> None:
        self._spend(config.REROLL_COST)
        self._roll()

    @staticmethod
    def sell_value(pet: Pet) -> int:
        return config.SELL_PRICE_PER_LEVEL * pet.level + pet.sell_bonus

    def sell(self, team: Team, idx: int) -> Pet:
        pet = team[idx]

        def action(ctx: Context) -> None:
            ctx.emit(Event(Trigger.SELL, pet, team, idx))  # fires while it's still on the team
            ctx.resolve()
            if pet in team.pets:
                team.pets.remove(pet)

        self.run(team, action)
        self.credits += self.sell_value(pet)
        return pet

    def _ice(self, obj) -> str:
        return " *FROZEN*" if self.is_frozen(obj) else ""

    def __str__(self):
        lines = [f"[-] Store [-]  Credits: {self.credits}  (tier {self.max_tier})", "  Pets:"]
        for i, pet in enumerate(self.pet_slots):
            text = f"{pet}  ({config.PET_COST}c)" if pet else "[SOLD]"
            lines.append(f"    [{i + 1}]{self._ice(pet)} {text}")
        lines.append("  Items:")
        for i, slot in enumerate(self.item_slots):
            lines.append(f"    [{i + 1}]{self._ice(slot)} {slot if slot else '[SOLD]'}")
        return "\n".join(lines)
