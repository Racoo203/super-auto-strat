"""Perks: the one thing a pet can hold. Most come from food (Honey, Garlic, Melon...); a few are
granted by abilities (Coconut from Gorilla, Peanut from Scorpion, Chili on Deer's Bus).

Damage-shaping perks are plain data read by Context.damage / Context.attack_damage. Perks that
react to a moment (Honey and Mushroom on faint, Cake at end of turn, Pancakes at start of battle)
register a handler in sap/abilities.py with @perk_ability.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Perk:
    name: str
    text: str
    attack_bonus: int = 0  # extra damage when this pet attacks (Meat Bone, Steak)
    damage_reduction: int = 0  # less damage taken from any source (Garlic, Melon, Coconut)
    min_damage: int = 0  # a reduced hit still deals at least this much (Garlic: 1)
    splash: int = 0  # when attacking, also hit the enemy behind the target (Chili)
    lethal: bool = False  # any pet this attacks and hurts is knocked out (Peanut)
    single_use: bool = False  # lost after it takes effect once (Melon, Steak, Coconut)

    def __str__(self):
        return self.name


IGNORE_ALL = 10**9

PERKS: dict[str, Perk] = {
    p.name: p
    for p in (
        Perk("Honey", "Faint -> summon one 1/1 Bee."),
        Perk("Meat Bone", "Attack with +3 damage.", attack_bonus=3),
        Perk("Lime", "Take 1 less damage.", damage_reduction=1, min_damage=1),
        Perk("Garlic", "Take 2 less damage.", damage_reduction=2, min_damage=1),
        Perk("Cake", "End turn -> increase sell value by 1 gold."),
        Perk("Chili", "Attack second enemy for 5 damage.", splash=5),
        Perk("Melon", "Take 20 less damage, once.", damage_reduction=20, single_use=True),
        Perk("Mushroom", "Faint -> come back as a 1/1."),
        Perk("Steak", "Attack with +20 damage, once.", attack_bonus=20, single_use=True),
        Perk("Pancakes", "Start of battle -> give friends +2/+2."),
        Perk("Coconut", "Ignore damage once.", damage_reduction=IGNORE_ALL, single_use=True),
        Perk("Peanut", "Knock out any pet attacked and hurt by this.", lethal=True),
    )
}
