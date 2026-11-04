"""Shop items (Super Auto Pets food).

An item is either eaten for an immediate effect (Apple, Pear, Salad Bowl...) or gives the pet a
held perk (Honey, Garlic, Melon... see sap/perks.py). A pet holds one perk; a new one replaces it.

Tier 0 items never roll in the shop: abilities stock them (Milk from Cow, Bread Crumbs from
Pigeon, Apples from Worm). Values follow the Super Auto Pets wiki; Better/Best Apple (+2/+2, +3/+3)
are assumed by analogy with Better/Best Milk.
"""

import random
from dataclasses import dataclass
from typing import Callable

from sap import config
from sap.events import Context, Event, Trigger, has_ability
from sap.pets import Pet
from sap.perks import PERKS
from sap.team import Team


@dataclass(frozen=True)
class Item:
    name: str
    tier: int  # 0 = only stocked by abilities, never rolled
    text: str
    cost: int = config.ITEM_COST
    perk: str | None = None  # perk this gives the pet that eats it
    targeted: bool = True  # bought onto one chosen pet (False: random pets / the shop)
    packs: tuple[str, ...] = ("Turtle Pack",)

    def __str__(self):
        return f"{self.name}: {self.text}"


def _perk(name: str, tier: int) -> Item:
    return Item(name, tier, f"Give one pet {name} ({PERKS[name].text})", perk=name)


ITEM_POOL: dict[str, Item] = {
    i.name: i
    for i in (
        # tier 1
        Item("Apple", 1, "Give one pet +1/+1."),
        _perk("Honey", 1),
        # tier 2
        Item("Cupcake", 2, "Give one pet +3/+3 until end of battle."),
        _perk("Meat Bone", 2),
        Item("Sleeping Pill", 2, "Make one pet faint.", cost=1),
        _perk("Lime", 2),
        # tier 3
        _perk("Cake", 3),
        _perk("Garlic", 3),
        Item("Salad Bowl", 3, "Give two random pets +1/+1.", targeted=False),
        # tier 4
        Item("Canned Food", 4, "Give all current and future shop pets +1/+1.", targeted=False),
        Item("Pear", 4, "Give one pet +2/+2."),
        # tier 5
        _perk("Chili", 5),
        Item("Chocolate", 5, "Give one pet +1 experience."),
        Item("Sushi", 5, "Give three random pets +1/+1.", targeted=False),
        # tier 6
        _perk("Melon", 6),
        _perk("Mushroom", 6),
        Item("Pizza", 6, "Give two random pets +2/+2.", targeted=False),
        _perk("Steak", 6),
        _perk("Pancakes", 6),
        # stocked by abilities only
        Item("Bread Crumbs", 0, "Give one pet +1 attack."),
        Item("Better Apple", 0, "Give one pet +2/+2."),
        Item("Best Apple", 0, "Give one pet +3/+3."),
        Item("Milk", 0, "Give one pet +1/+2."),
        Item("Better Milk", 0, "Give one pet +2/+4."),
        Item("Best Milk", 0, "Give one pet +3/+6."),
    )
}

# Eaten-item effects: (ctx, team, target or None) -> the pets that ate it.
FoodEffect = Callable[[Context, Team, Pet | None], list[Pet]]
FOOD_EFFECTS: dict[str, FoodEffect] = {}


def food(name: str):
    def register(effect: FoodEffect) -> FoodEffect:
        FOOD_EFFECTS[name] = effect
        return effect

    return register


def food_buff(ctx: Context, team: Team, pet: Pet, attack: int, health: int, temporary: bool = False) -> None:
    """Stats from food. A Cat on the team multiplies them (x2/x3/x4), twice per turn."""
    mult = 1
    for cat in sorted((p for p in team if has_ability(p, "Cat")), key=lambda p: -p.ability_level):
        if ctx.use(cat, "cat", 2):
            mult = cat.ability_level + 1
            ctx.say(f"{cat.name} multiplies the food x{mult}")
            break
    ctx.buff(pet, attack * mult, health * mult, temporary=temporary)


def _stat_food(name: str, attack: int, health: int, temporary: bool = False) -> None:
    @food(name)
    def effect(ctx, team, pet):
        food_buff(ctx, team, pet, attack, health, temporary)
        return [pet]


for _name, _a, _h in [
    ("Apple", 1, 1), ("Better Apple", 2, 2), ("Best Apple", 3, 3), ("Pear", 2, 2),
    ("Bread Crumbs", 1, 0), ("Milk", 1, 2), ("Better Milk", 2, 4), ("Best Milk", 3, 6),
]:
    _stat_food(_name, _a, _h)
_stat_food("Cupcake", 3, 3, temporary=True)


def _random_food(name: str, count: int, amount: int) -> None:
    @food(name)
    def effect(ctx, team, pet):
        eaters = ctx.pick([p for p in team if not p.fainted], count)
        for p in eaters:
            food_buff(ctx, team, p, amount, amount)
        return eaters


_random_food("Salad Bowl", 2, 1)
_random_food("Sushi", 3, 1)
_random_food("Pizza", 2, 2)


@food("Sleeping Pill")
def sleeping_pill(ctx, team, pet):
    pet.take_damage(pet.health)  # faints when the context resolves, triggering its Faint ability
    return []


@food("Chocolate")
def chocolate(ctx, team, pet):
    if pet.gain_xp(1):
        ctx.level_up(pet, team)
    return [pet]


@food("Canned Food")
def canned_food(ctx, team, pet):
    ctx.store.add_pet_bonus(1, 1)
    return []


def feed(ctx: Context, team: Team, item: Item, pet: Pet | None) -> None:
    """Apply an item to the team (and `pet` if targeted), then fire the eat triggers."""
    if item.perk:
        ctx.give_perk(pet, item.perk)
        eaters = [pet]
    else:
        eaters = FOOD_EFFECTS[item.name](ctx, team, pet)
    for p in eaters:
        ctx.emit(Event(Trigger.EAT_FOOD, p, team))
        ctx.emit(Event(Trigger.FRIEND_ATE_FOOD, p, team))


def items_up_to_tier(max_tier: int, pack: str = config.PACK) -> list[str]:
    return [n for n, i in ITEM_POOL.items() if 1 <= i.tier <= max_tier and pack in i.packs]


def random_item(rng: random.Random, max_tier: int = 1) -> Item:
    return ITEM_POOL[rng.choice(items_up_to_tier(max_tier))]
