import random
from dataclasses import dataclass, field
from enum import Enum

import sap.abilities  # noqa: F401  (registers pet abilities)
from sap.config import MAX_BATTLE_TURNS
from sap.events import Context, Event, Trigger
from sap.pets import Pet
from sap.team import Team


class Outcome(Enum):
    """Outcome from the point of view of the first team passed to `fight`."""

    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


@dataclass
class BattleResult:
    outcome: Outcome
    turns: int
    log: list[str] = field(default_factory=list)


def _strike(ctx: Context, attacker: Pet, target: Pet, amount: int) -> None:
    ctx.damage(target, amount, source=attacker, attack=True)
    perk = attacker.perk
    if perk and perk.splash:  # Chili: also hit the enemy behind the target
        for second in ctx.behind(target, 1):
            ctx.damage(second, perk.splash, source=attacker, attack=True)
            ctx.say(f"{attacker.name}'s {perk.name} hits {second.name} for {perk.splash}")


def fight(
    team_a: Team,
    team_b: Team,
    max_turns: int = MAX_BATTLE_TURNS,
    rng: random.Random | None = None,
) -> BattleResult:
    """Battle two teams with simultaneous strikes. The original teams are not modified.

    Phases: start of battle -> repeat (before attack -> strike -> attack/hurt/faint chains) -> result.
    """
    a, b = team_a.copy(), team_b.copy()
    for pet in [*a, *b]:
        pet.reset_uses()  # "N times per battle" limits start fresh
    log = [f"{a.name} vs {b.name}"]
    ctx = Context([a, b], rng or random.Random(), log)

    ctx.emit(Event(Trigger.START_OF_BATTLE))
    ctx.resolve()

    turn = 0
    while a and b and turn < max_turns:
        turn += 1
        for team in (a, b):
            ctx.emit(Event(Trigger.BEFORE_ATTACK, team.front, team, 0))
        ctx.resolve()
        if not (a and b):
            break

        pa, pb = a.front, b.front
        dmg_to_b, dmg_to_a = ctx.attack_damage(pa), ctx.attack_damage(pb)
        log.append(
            f"Turn {turn}: {pa.name} ({a.name}) hits {pb.name} for {dmg_to_b} | "
            f"{pb.name} ({b.name}) hits {pa.name} for {dmg_to_a}"
        )
        # Both strike at the same time; faints and triggers resolve after both hits land.
        _strike(ctx, pa, pb, dmg_to_b)
        _strike(ctx, pb, pa, dmg_to_a)
        for pet, team in ((pa, a), (pb, b)):
            ctx.emit(Event(Trigger.FRIEND_AHEAD_ATTACKS, pet, team, 0))
            ctx.emit(Event(Trigger.AFTER_ATTACK, pet, team, 0))
        ctx.resolve()

    if a and not b:
        outcome = Outcome.WIN
    elif b and not a:
        outcome = Outcome.LOSS
    else:  # both wiped at once, or the turn cap was hit
        outcome = Outcome.DRAW
    log.append(f"Result for {a.name}: {outcome.value.upper()}")
    return BattleResult(outcome, turn, log)
