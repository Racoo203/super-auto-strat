from dataclasses import dataclass, field
from enum import Enum

from sap.config import MAX_BATTLE_TURNS
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


def fight(team_a: Team, team_b: Team, max_turns: int = MAX_BATTLE_TURNS) -> BattleResult:
    """Battle two teams with simultaneous strikes. The original teams are not modified."""
    a, b = team_a.copy(), team_b.copy()
    log = [f"{a.name} vs {b.name}"]
    turn = 0

    while a and b and turn < max_turns:
        turn += 1
        pa, pb = a.front, b.front
        dmg_to_b, dmg_to_a = pa.attack, pb.attack
        # Both strike at the same time, then both take the hits, then faint checks.
        pa.take_damage(dmg_to_a)
        pb.take_damage(dmg_to_b)
        log.append(
            f"Turn {turn}: {pa.name} ({a.name}) hits {pb.name} for {dmg_to_b} | "
            f"{pb.name} ({b.name}) hits {pa.name} for {dmg_to_a}"
        )
        for team in (a, b):
            for pet in team.remove_fainted():
                log.append(f"    {pet.name} ({team.name}) fainted")

    if a and not b:
        outcome = Outcome.WIN
    elif b and not a:
        outcome = Outcome.LOSS
    else:  # both wiped at once, or the turn cap was hit
        outcome = Outcome.DRAW
    log.append(f"Result for {a.name}: {outcome.value.upper()}")
    return BattleResult(outcome, turn, log)
