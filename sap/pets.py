import random
from dataclasses import dataclass, replace

from sap.items import Item

# name -> (attack, health). Add a row to add a pet.
PET_POOL = {
    "Fish": (2, 3),
    "Dog": (3, 3),
    "Cat": (4, 2),
    "Snake": (5, 1),
    "Fox": (3, 4),
    "Duck": (1, 5),
}


@dataclass
class Pet:
    name: str
    base_attack: int
    base_health: int
    item: Item | None = None
    damage: int = 0  # damage taken during the current battle

    @property
    def attack(self) -> int:
        return self.base_attack + (self.item.attack_bonus if self.item else 0)

    @property
    def max_health(self) -> int:
        return self.base_health + (self.item.health_bonus if self.item else 0)

    @property
    def health(self) -> int:
        return self.max_health - self.damage

    @property
    def fainted(self) -> bool:
        return self.health <= 0

    def take_damage(self, amount: int) -> None:
        self.damage += amount

    def copy(self) -> "Pet":
        return replace(self)  # Item is immutable, so a shallow copy is safe

    def __str__(self):
        held = f" [{self.item.name}]" if self.item else ""
        return f"{self.name} {self.attack}/{self.health}{held}"


def make_pet(name: str) -> Pet:
    attack, health = PET_POOL[name]
    return Pet(name, attack, health)


def random_pet(rng: random.Random) -> Pet:
    return make_pet(rng.choice(list(PET_POOL)))
