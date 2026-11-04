"""Triggers, the ability registry and the event dispatcher.

How a pet ability works:
  1. The engine (battle or store) emits an Event at the right moment, e.g. Event(Trigger.SELL, pig).
  2. The dispatcher finds the pets that hear it (see Trigger.scope) and whose ability (or perk)
     uses that trigger.
  3. Each handler runs, highest attack first (ties broken by the seeded rng), through a Context
     that offers safe verbs: damage, buff, summon, gain_gold, stock...
  4. Handlers may emit further events (damage -> HURT, a faint -> FAINT, a summon -> SUMMONED).
     They are queued and resolved in order until the queue is empty, so chains just work.

Adding a pet ability = one decorated function in sap/abilities.py. Adding a new kind of trigger =
one Trigger member plus one `ctx.emit(...)` where the engine reaches that moment.
"""

import random
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from sap.pets import Pet
from sap.perks import PERKS
from sap.team import Team


class Scope(Enum):
    SELF = "self"  # only the subject pet hears it ("Faint", "Sell")
    FRIENDS = "friends"  # the subject's teammates, not the subject ("Friend summoned")
    BEHIND = "behind"  # the pet directly behind the subject ("Friend ahead attacks")
    ALL = "all"  # every pet in the context; the event has no subject ("Start of battle")
    NONE = "none"  # never dispatched: passive abilities checked directly by the engine


class Trigger(Enum):
    """Moments a pet can react to. The value is (label, scope)."""

    # battle
    START_OF_BATTLE = ("Start of battle", Scope.ALL)
    BEFORE_ATTACK = ("Before attack", Scope.SELF)
    AFTER_ATTACK = ("After attack", Scope.SELF)
    FRIEND_AHEAD_ATTACKS = ("Friend ahead attacks", Scope.BEHIND)
    HURT = ("Hurt", Scope.SELF)
    FRIEND_HURT = ("Friend hurt", Scope.FRIENDS)
    FAINT = ("Faint", Scope.SELF)
    FRIEND_FAINTS = ("Friend faints", Scope.FRIENDS)
    FRIEND_AHEAD_FAINTS = ("Friend ahead faints", Scope.BEHIND)
    KNOCK_OUT = ("Knock out", Scope.SELF)
    SUMMONED = ("Summoned", Scope.SELF)
    FRIEND_SUMMONED = ("Friend summoned", Scope.FRIENDS)
    # store
    BUY = ("Buy", Scope.SELF)
    FRIEND_BOUGHT = ("Friend bought", Scope.FRIENDS)
    SELL = ("Sell", Scope.SELF)
    LEVEL_UP = ("Level up", Scope.SELF)
    EAT_FOOD = ("Eats food", Scope.SELF)
    FRIEND_ATE_FOOD = ("Friendly ate food", Scope.FRIENDS)
    START_TURN = ("Start of turn", Scope.ALL)
    END_TURN = ("End turn", Scope.ALL)
    # always-on rule read by the engine (Cat, Tiger)
    PASSIVE = ("Passive", Scope.NONE)

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def scope(self) -> Scope:
        return self.value[1]


@dataclass
class Event:
    trigger: Trigger
    pet: Pet | None = None  # the subject: who fainted, was bought, was summoned...
    team: Team | None = None  # the subject's team (kept because a fainted pet has left it)
    index: int | None = None  # the subject's position (for a faint: the slot it left)


# effect(ctx, pet, event, level). `level` is usually pet.level, but differs when a Parrot copies
# an ability or a Tiger repeats one.
Effect = Callable[["Context", Pet, Event, int], None]
PerkEffect = Callable[["Context", Pet, Event], None]


@dataclass(frozen=True)
class Ability:
    trigger: Trigger
    effect: Effect


ABILITIES: dict[str, list[Ability]] = {}
PERK_ABILITIES: dict[str, dict[Trigger, PerkEffect]] = {}


def ability(pet_name: str, trigger: Trigger):
    """Register the decorated function as one of `pet_name`'s abilities."""

    def register(effect: Effect) -> Effect:
        ABILITIES.setdefault(pet_name, []).append(Ability(trigger, effect))
        return effect

    return register


def perk_ability(perk_name: str, trigger: Trigger):
    """Register what a perk does at a given moment. Signature: (ctx, pet, event)."""
    if perk_name not in PERKS:
        raise ValueError(f"Unknown perk {perk_name}")

    def register(effect: PerkEffect) -> PerkEffect:
        PERK_ABILITIES.setdefault(perk_name, {})[trigger] = effect
        return effect

    return register


def has_ability(pet: Pet, name: str) -> bool:
    return pet.ability_name == name


class Context:
    """What an effect can see and do. One per battle, or per store action.

    `teams` holds one team (store) or two (battle). Battle changes land on the battle copies, so
    they last until the end of the battle; store changes land on the real team and are permanent.
    """

    def __init__(self, teams: list[Team], rng: random.Random, log: list[str] | None = None, store=None):
        self.teams = list(teams)
        self.rng = rng
        self.log = log if log is not None else []
        self.store = store
        self.queue: deque[Event] = deque()
        self.last_hit: dict[int, Pet] = {}  # id(target) -> pet that last damaged it (knock outs)

    @property
    def in_battle(self) -> bool:
        return len(self.teams) == 2

    # --- lookups -----------------------------------------------------------
    def team_of(self, pet: Pet) -> Team | None:
        return next((t for t in self.teams if any(p is pet for p in t)), None)

    def alive(self, pet: Pet) -> bool:
        return not pet.fainted and self.team_of(pet) is not None

    def enemy_team(self, team: Team | None) -> Team | None:
        others = [t for t in self.teams if t is not team]
        return others[0] if self.in_battle and others else None

    def position(self, pet: Pet, event: Event | None = None) -> int | None:
        """Index of `pet` in its team, or the slot it left if it just fainted."""
        team = self.team_of(pet)
        if team is not None:
            return next(i for i, p in enumerate(team.pets) if p is pet)
        if event is not None and event.pet is pet:
            return event.index
        return None

    def friends(self, pet: Pet, team: Team | None = None) -> list[Pet]:
        """Living teammates. Pass `team` for a pet that already left it (e.g. on faint)."""
        team = team or self.team_of(pet)
        return [p for p in team if p is not pet and not p.fainted] if team else []

    def enemies(self, pet: Pet, team: Team | None = None) -> list[Pet]:
        enemy = self.enemy_team(team or self.team_of(pet))
        return [p for p in enemy if not p.fainted] if enemy else []

    def ahead(self, pet: Pet, n: int, event: Event | None = None, team: Team | None = None) -> list[Pet]:
        """Up to `n` living friends ahead of `pet`, nearest first."""
        team = team or self.team_of(pet) or (event.team if event else None)
        pos = self.position(pet, event)
        if team is None or pos is None:
            return []
        return [p for p in reversed(team.pets[:pos]) if not p.fainted][:n]

    def behind(self, pet: Pet, n: int, event: Event | None = None, team: Team | None = None) -> list[Pet]:
        """Up to `n` living friends behind `pet`, nearest first."""
        team = team or self.team_of(pet) or (event.team if event else None)
        pos = self.position(pet, event)
        if team is None or pos is None:
            return []
        start = pos + 1 if any(p is pet for p in team) else pos
        return [p for p in team.pets[start:] if not p.fainted and p is not pet][:n]

    def pick(self, pets: list[Pet], n: int) -> list[Pet]:
        return self.rng.sample(pets, min(n, len(pets)))

    def random_friends(self, pet: Pet, n: int, team: Team | None = None) -> list[Pet]:
        return self.pick(self.friends(pet, team), n)

    def random_enemies(self, pet: Pet, n: int, team: Team | None = None) -> list[Pet]:
        return self.pick(self.enemies(pet, team), n)

    def use(self, pet: Pet, key: str, limit: int) -> bool:
        """Count one use of a limited ability ("works N times per turn/battle"). False if spent."""
        counter = f"uses:{key}"
        if pet.state.get(counter, 0) >= limit:
            return False
        pet.state[counter] = pet.state.get(counter, 0) + 1
        return True

    # --- verbs ---------------------------------------------------------------
    def say(self, msg: str) -> None:
        self.log.append(f"    {msg}")

    def attack_damage(self, attacker: Pet) -> int:
        """Damage of a normal attack, including (and consuming) attack perks."""
        amount, perk = attacker.attack, attacker.perk
        if perk and perk.attack_bonus:
            amount += perk.attack_bonus
            if perk.single_use:
                attacker.perk = None
                self.say(f"{attacker.name} uses up {perk.name}")
        return amount

    def damage(self, target: Pet, amount: int, source: Pet | None = None, attack: bool = False) -> int:
        """Deal damage through the target's perk. Returns the damage actually dealt."""
        if amount <= 0 or target.fainted:
            return 0
        perk = target.perk
        if perk and perk.damage_reduction:
            reduced = max(perk.min_damage, amount - perk.damage_reduction)
            if reduced < amount:
                self.say(f"{target.name}'s {perk.name} blocks {amount - reduced}")
                if perk.single_use:
                    target.perk = None
            amount = reduced
        if amount <= 0:
            return 0
        if attack and source and source.perk and source.perk.lethal:
            amount = max(amount, target.health)
        target.take_damage(amount)
        if source:
            self.last_hit[id(target)] = source
            if not attack:
                self.say(f"{source.name} deals {amount} to {target.name}")
        team = self.team_of(target)
        self.emit(Event(Trigger.HURT, target, team))
        self.emit(Event(Trigger.FRIEND_HURT, target, team))
        return amount

    def reduce_health(self, target: Pet, amount: int, source: Pet | None = None) -> None:
        """Remove health directly: not damage, so perks don't block it and Hurt doesn't trigger."""
        if amount <= 0:
            return
        target.take_damage(amount)
        if source:
            self.say(f"{source.name} removes {amount} health from {target.name}")

    def set_health(self, target: Pet, value: int) -> None:
        target.base_health += max(1, value) - target.health

    def buff(self, target: Pet, attack: int = 0, health: int = 0, source: Pet | None = None,
             temporary: bool = False) -> None:
        target.buff(attack, health, temporary)
        if source:
            until = " until end of battle" if temporary and not self.in_battle else ""
            self.say(f"{source.name} gives {target.name} +{attack}/+{health}{until}")

    def give_perk(self, target: Pet, perk_name: str, source: Pet | None = None) -> None:
        target.perk = PERKS[perk_name]
        if source:
            self.say(f"{source.name} gives {target.name} {perk_name}")

    def summon(self, team: Team | None, pet: Pet, index: int, source: Pet | None = None) -> bool:
        if team is None:
            return False
        if team.is_full:
            self.say(f"No room to summon {pet.name}")
            return False
        index = max(0, min(index, len(team)))
        team.pets.insert(index, pet)
        self.say(f"{source.name if source else 'Ability'} summons {pet}")
        self.emit(Event(Trigger.SUMMONED, pet, team, index))
        self.emit(Event(Trigger.FRIEND_SUMMONED, pet, team, index))
        return True

    def level_up(self, pet: Pet, team: Team | None = None) -> None:
        self.say(f"{pet.name} reached level {pet.level}!")
        if self.store is not None:
            bonus = self.store.add_level_up_pet()
            if bonus:
                self.say(f"Level-up bonus: {bonus.name} (tier {bonus.tier}) added to the shop")
        self.emit(Event(Trigger.LEVEL_UP, pet, team or self.team_of(pet)))

    def gain_gold(self, amount: int, source: Pet | None = None) -> None:
        if self.store is None:
            return
        self.store.credits += amount
        if source:
            self.say(f"{source.name} gives +{amount} credits")

    def stock(self, item_name: str, price: int, source: Pet | None = None) -> None:
        if self.store is None:
            return
        self.store.stock(item_name, price)
        if source:
            self.say(f"{source.name} stocks {item_name} ({price}c)")

    # --- dispatch --------------------------------------------------------------
    def emit(self, event: Event) -> None:
        self.queue.append(event)

    def resolve(self) -> None:
        """Process queued events (and the faints they cause) until nothing is left."""
        while True:
            self._remove_fainted()
            if not self.queue:
                return
            self._dispatch(self.queue.popleft())

    def _remove_fainted(self) -> None:
        for team in self.teams:
            survivors = []
            for pet in team.pets:
                if not pet.fainted:
                    survivors.append(pet)
                    continue
                self.say(f"{pet.name} ({team.name}) fainted")
                index = len(survivors)
                for trigger in (Trigger.FAINT, Trigger.FRIEND_FAINTS, Trigger.FRIEND_AHEAD_FAINTS):
                    self.emit(Event(trigger, pet, team, index))
                killer = self.last_hit.pop(id(pet), None)
                if killer is not None and killer is not pet and self.alive(killer):
                    self.emit(Event(Trigger.KNOCK_OUT, killer, self.team_of(killer)))
            team.pets = survivors

    def _candidates(self, event: Event) -> list[Pet]:
        scope = event.trigger.scope
        if scope is Scope.SELF:
            return [event.pet]
        if scope is Scope.FRIENDS:
            return [p for p in event.team if p is not event.pet]
        if scope is Scope.BEHIND:
            return self.behind(event.pet, 1, event, event.team)
        if scope is Scope.ALL:
            return [p for t in self.teams for p in t]
        return []

    def _dispatch(self, event: Event) -> None:
        handlers = []  # (sort key, pet, run)
        for pet in self._candidates(event):
            key = (-pet.attack, self.rng.random())
            for ab in ABILITIES.get(pet.ability_name, []):
                if ab.trigger is event.trigger:
                    handlers.append((key + (0,), pet, ab))
            perk_effect = PERK_ABILITIES.get(pet.perk.name, {}).get(event.trigger) if pet.perk else None
            if perk_effect:
                handlers.append((key + (1,), pet, perk_effect))
        handlers.sort(key=lambda h: h[0])

        for _, pet, handler in handlers:
            # A pet knocked out earlier in this same event no longer acts (except to its own faint).
            if event.trigger is not Trigger.FAINT and not self.alive(pet):
                continue
            if isinstance(handler, Ability):
                handler.effect(self, pet, event, pet.ability_level)
                self._tiger_repeat(pet, event, handler)
            else:
                handler(self, pet, event)

    def _tiger_repeat(self, pet: Pet, event: Event, ab: Ability) -> None:
        """Tiger: in battle, the friend ahead of a Tiger repeats its ability at the Tiger's level."""
        if not self.in_battle:
            return
        team = self.team_of(pet) or (event.team if event.pet is pet else None)
        for tiger in self.behind(pet, 1, event, team):
            if has_ability(tiger, "Tiger"):
                self.say(f"{tiger.name} makes {pet.name} repeat its ability")
                ab.effect(self, pet, event, tiger.ability_level)
