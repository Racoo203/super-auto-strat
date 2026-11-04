import csv
import random
from dataclasses import dataclass, field, replace
from pathlib import Path

from sap import config
from sap.perks import Perk

DATA_FILE = Path(__file__).parent / "data" / "pets.csv"


@dataclass(frozen=True)
class PetSpec:
    """Static data for one pet type, as listed in the shop."""

    name: str
    tier: int
    attack: int
    health: int
    texts: tuple[str, ...] = ()  # ability description per level (display only)

    def text(self, level: int) -> str:
        return self.texts[level - 1] if self.texts else ""


def load_pool(path: Path = DATA_FILE, pack: str = config.PACK) -> dict[str, PetSpec]:
    """Read the scraped pet table: one row per (pet, level). Stats are the level-1 stats."""
    rows: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if pack in row["Packs"]:
                rows.setdefault(row["Name"], []).append(row)
    pool = {}
    for name, levels in rows.items():
        first = levels[0]
        pool[name] = PetSpec(
            name=name,
            tier=int(first["Tier"].split()[1]),  # "Tier 3 Pets" -> 3
            attack=int(first["Attack"]),
            health=int(first["Health"]),
            texts=tuple(r["Ability"] for r in sorted(levels, key=lambda r: int(r["Level"]))),
        )
    return pool


PET_POOL: dict[str, PetSpec] = load_pool()


def level_for(xp: int) -> int:
    return 1 + sum(xp >= need for need in config.LEVEL_XP)


def xp_for(level: int) -> int:
    return 0 if level <= 1 else config.LEVEL_XP[level - 2]


@dataclass(eq=False)  # pets compare by identity: two identical Sloths are still two pets
class Pet:
    name: str
    base_attack: int  # includes permanent buffs
    base_health: int
    perk: Perk | None = None
    damage: int = 0  # damage taken (battle copies: this battle; real team: shop damage, e.g. Hedgehog)
    xp: int = 0
    tier: int = 0
    temp_attack: int = 0  # "until end of battle" buffs (Cupcake, Horse...), cleared after each fight
    temp_health: int = 0
    sell_bonus: int = 0  # extra sell value (Cake)
    # Per-pet ability memory: usage counters ("uses:*"), the pet a Whale swallowed, the ability a
    # Parrot copied... Flat keys only, so a shallow copy is enough.
    state: dict = field(default_factory=dict)

    # --- stats ------------------------------------------------------------
    @property
    def attack(self) -> int:
        return max(0, min(config.MAX_STAT, self.base_attack + self.temp_attack))

    @property
    def max_health(self) -> int:
        return min(config.MAX_STAT, self.base_health + self.temp_health)

    @property
    def health(self) -> int:
        return self.max_health - self.damage

    @property
    def fainted(self) -> bool:
        return self.health <= 0

    def take_damage(self, amount: int) -> None:
        self.damage += amount

    def buff(self, attack: int = 0, health: int = 0, temporary: bool = False) -> None:
        if temporary:
            self.temp_attack += attack
            self.temp_health += health
        else:
            self.base_attack = min(config.MAX_STAT, self.base_attack + attack)
            self.base_health = min(config.MAX_STAT, self.base_health + health)

    def end_battle(self) -> None:
        """Drop everything that only lasts until the end of battle."""
        self.temp_attack = self.temp_health = 0
        self.state.pop("copied", None)

    def reset_uses(self) -> None:
        for key in [k for k in self.state if k.startswith("uses:")]:
            del self.state[key]

    # --- ability identity (a Parrot can borrow another pet's ability) -----------
    @property
    def ability_name(self) -> str:
        return self.state.get("copied", (self.name, 0))[0]

    @property
    def ability_level(self) -> int:
        copied = self.state.get("copied")
        return copied[1] if copied else self.level

    # --- leveling -----------------------------------------------------------
    @property
    def level(self) -> int:
        return level_for(self.xp)

    @property
    def max_level(self) -> bool:
        return self.level >= config.MAX_LEVEL

    def gain_xp(self, amount: int = 1) -> bool:
        """Returns True if this pet gained a level."""
        old_level = self.level
        self.xp = min(config.MAX_XP, self.xp + amount)
        return self.level > old_level

    def can_merge(self, other: "Pet") -> bool:
        return other is not self and other.name == self.name and not self.max_level

    def merge(self, other: "Pet") -> bool:
        """Stack `other` into this pet: best of each stat +1/+1, and `other`'s xp + 1.
        Stacking cannot be undone. Returns True if this pet gained a level."""
        if not self.can_merge(other):
            raise ValueError(f"Cannot stack {other.name} onto {self.name}")
        self.base_attack = min(config.MAX_STAT, max(self.base_attack, other.base_attack) + 1)
        self.base_health = min(config.MAX_STAT, max(self.base_health, other.base_health) + 1)
        self.temp_attack += other.temp_attack
        self.temp_health += other.temp_health
        self.sell_bonus += other.sell_bonus
        self.perk = self.perk or other.perk
        return self.gain_xp(other.xp + 1)

    # --- misc ---------------------------------------------------------------
    @property
    def ability_text(self) -> str:
        spec = PET_POOL.get(self.name)
        return spec.text(self.level) if spec else ""

    def copy(self) -> "Pet":
        return replace(self, state=dict(self.state))  # Perk is immutable

    def __str__(self):
        held = f" [{self.perk.name}]" if self.perk else ""
        progress = "" if self.max_level else f" ({self.xp}/{config.LEVEL_XP[self.level - 1]})"
        return f"{self.name} {self.attack}/{self.health} Lv{self.level}{progress}{held}"


def make_pet(name: str, level: int = 1) -> Pet:
    spec = PET_POOL[name]
    return Pet(name, spec.attack, spec.health, xp=xp_for(level), tier=spec.tier)


def pets_up_to_tier(max_tier: int) -> list[str]:
    return [n for n, spec in PET_POOL.items() if spec.tier <= max_tier]


def pets_of_tier(tier: int) -> list[str]:
    return [n for n, spec in PET_POOL.items() if spec.tier == tier]


def random_pet(rng: random.Random, max_tier: int = 1) -> Pet:
    return make_pet(rng.choice(pets_up_to_tier(max_tier)))
