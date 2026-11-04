import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    """A held power-up. Its bonus applies only while a pet holds it."""

    name: str
    attack_bonus: int = 0
    health_bonus: int = 0

    def __str__(self):
        return f"{self.name} (+{self.attack_bonus}/+{self.health_bonus})"


ITEM_POOL = {
    "Apple": Item("Apple", attack_bonus=1, health_bonus=1),
    "Bone": Item("Bone", attack_bonus=3),
    "Garlic": Item("Garlic", health_bonus=3),
}


def random_item(rng: random.Random) -> Item:
    return rng.choice(list(ITEM_POOL.values()))
