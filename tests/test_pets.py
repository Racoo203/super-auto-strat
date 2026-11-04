import random

import pytest

from sap import config
from sap.battle import Outcome, fight
from sap.errors import ActionError
from sap.events import ABILITIES, Context, Event, Trigger, ability
from sap.pets import PET_POOL, Pet, make_pet
from sap.store import Store
from sap.team import Team


def new_store(seed=0):
    store = Store(random.Random(seed))
    store.new_visit()
    return store


def stack(name: str, copies: int) -> Pet:
    pet = make_pet(name)
    for _ in range(copies - 1):
        pet.merge(make_pet(name))
    return pet


# --- data -----------------------------------------------------------------
def test_pool_loads_turtle_pack_with_tiers():
    assert len(PET_POOL) >= 60
    assert (PET_POOL["Mosquito"].attack, PET_POOL["Mosquito"].health) == (2, 2)
    assert PET_POOL["Mosquito"].tier == 1 and PET_POOL["Boar"].tier == 6
    assert make_pet("Pig").ability_text.startswith("Sell")


def test_every_ability_belongs_to_a_known_pet():
    assert set(ABILITIES) <= set(PET_POOL)


# --- leveling ---------------------------------------------------------------
def test_three_copies_reach_level_2_and_six_reach_level_3():
    assert stack("Sloth", 2).level == 1
    assert stack("Sloth", 3).level == 2
    assert stack("Sloth", 5).level == 2
    assert stack("Sloth", 6).level == 3


def test_merge_keeps_best_stats_plus_one():
    a, b = Pet("Sloth", 1, 5), Pet("Sloth", 4, 2)
    assert a.merge(b) is False  # 2 copies: still level 1
    assert (a.attack, a.health, a.xp) == (5, 6, 1)
    assert a.merge(Pet("Sloth", 1, 1)) is True  # 3rd copy: level up


def test_cannot_merge_different_pets_or_past_max_level():
    with pytest.raises(ValueError):
        make_pet("Pig").merge(make_pet("Ant"))
    maxed = stack("Sloth", 6)
    assert maxed.max_level and not maxed.can_merge(make_pet("Sloth"))


def test_merging_a_stack_carries_its_xp():
    a, b = stack("Sloth", 2), stack("Sloth", 2)  # 1 xp each
    a.merge(b)
    assert a.xp == 3 and a.level == 2  # 4 copies


# --- store: stacking, combining, selling ---------------------------------
def test_buy_onto_same_pet_stacks_it():
    store, team = new_store(), Team("T", [make_pet("Sloth")])
    store.pet_slots[0] = make_pet("Sloth")
    pet = store.buy_pet(0, team, target=0)
    assert len(team) == 1 and pet is team[0] and pet.xp == 1
    assert store.credits == config.CREDITS_PER_TURN - config.PET_COST


def test_buy_onto_a_different_pet_fails_without_spending():
    store, team = new_store(), Team("T", [make_pet("Sloth")])
    store.pet_slots[0] = make_pet("Pig")
    with pytest.raises(ActionError):
        store.buy_pet(0, team, target=0)
    assert store.credits == config.CREDITS_PER_TURN and store.pet_slots[0] is not None


def test_stacking_works_with_a_full_team():
    store = new_store()
    team = Team("T", [make_pet("Sloth") for _ in range(config.TEAM_SIZE)])
    store.pet_slots[0] = make_pet("Sloth")
    store.buy_pet(0, team, target=2)
    assert team[2].xp == 1


def test_combine_two_team_pets():
    store, team = new_store(), Team("T", [make_pet("Sloth"), make_pet("Pig"), make_pet("Sloth")])
    store.combine(team, 0, 2)
    assert [p.name for p in team] == ["Pig", "Sloth"] and team[1].xp == 1
    with pytest.raises(ActionError):
        store.combine(team, 0, 1)  # Pig onto Sloth
    with pytest.raises(ActionError):
        store.combine(team, 1, 1)  # onto itself


def test_sell_value_is_level():
    store, team = new_store(), Team("T", [stack("Sloth", 3)])
    store.sell(team, 0)
    assert store.credits == config.CREDITS_PER_TURN + 2 * config.SELL_PRICE_PER_LEVEL


# --- shop abilities -----------------------------------------------------------
@pytest.mark.parametrize("copies, bonus", [(1, 1), (3, 2), (6, 3)])
def test_pig_gives_extra_credits_per_level(copies, bonus):
    store, team = new_store(), Team("T", [stack("Pig", copies)])
    level = team[0].level
    store.sell(team, 0)
    assert store.credits == config.CREDITS_PER_TURN + level * config.SELL_PRICE_PER_LEVEL + bonus


def test_beaver_sell_buffs_two_friends_attack():
    store = new_store()
    team = Team("T", [make_pet("Beaver"), Pet("A", 1, 1), Pet("B", 1, 1), Pet("C", 1, 1)])
    store.sell(team, 0)
    assert sorted(p.attack for p in team) == [1, 2, 2]


def test_duck_sell_buffs_shop_pets_health():
    store, team = new_store(), Team("T", [make_pet("Duck")])
    before = [p.health for p in store.pet_slots]
    store.sell(team, 0)
    assert [p.health for p in store.pet_slots] == [h + 1 for h in before]


def test_otter_buy_buffs_one_friend_health():
    store, team = new_store(), Team("T", [Pet("A", 1, 1)])
    store.pet_slots[0] = make_pet("Otter")
    store.buy_pet(0, team)
    assert team[0].health == 2


def test_fish_level_up_buffs_friends():
    store = new_store()
    team = Team("T", [stack("Fish", 2), Pet("A", 1, 1), Pet("B", 1, 1)])
    store.pet_slots[0] = make_pet("Fish")
    store.buy_pet(0, team, target=0)
    assert team[0].level == 2
    assert [(p.attack, p.health) for p in team.pets[1:]] ==[(2, 2), (2, 2)]


def test_shop_buffs_are_permanent():
    store, team = new_store(), Team("T", [Pet("A", 1, 1)])
    store.pet_slots[0] = make_pet("Otter")
    store.buy_pet(0, team)
    fight(team, Team("E", [Pet("E", 1, 1)]))
    assert team[0].max_health == 2


# --- battle abilities ---------------------------------------------------------------
@pytest.mark.parametrize("copies, hits", [(1, 1), (3, 2), (6, 3)])
def test_mosquito_hits_one_enemy_per_level(copies, hits):
    a = Team("A", [stack("Mosquito", copies)])
    b = Team("B", [Pet(f"E{i}", 0, 1) for i in range(5)])
    result = fight(a, b, max_turns=0)  # start of battle only
    assert sum("Mosquito deals 1" in line for line in result.log) == hits
    assert sum("fainted" in line for line in result.log) == hits


def test_mosquito_can_win_before_any_attack():
    result = fight(Team("A", [make_pet("Mosquito")]), Team("B", [Pet("E", 9, 1)]))
    assert result.outcome is Outcome.WIN and result.turns == 0


def test_ant_faint_buffs_a_friend():
    a = Team("A", [Pet("Ant", 2, 1), Pet("Back", 1, 50)])
    b = Team("B", [Pet("E", 1, 50)])
    result = fight(a, b, max_turns=1)
    assert any("Ant gives Back +1/+1" in line for line in result.log)


def test_cricket_summons_zombie_and_horse_buffs_it():
    a = Team("A", [Pet("Cricket", 1, 1), make_pet("Horse")])
    b = Team("B", [Pet("E", 1, 50)])
    result = fight(a, b, max_turns=1)
    log = "\n".join(result.log)
    assert "Cricket summons Zombie Cricket 1/1" in log
    assert "Horse gives Zombie Cricket +1/+0" in log


def test_battle_buffs_do_not_leak_into_the_real_team():
    a = Team("A", [Pet("Ant", 2, 1), Pet("Back", 1, 50)])
    fight(a, Team("B", [Pet("E", 1, 50)]), max_turns=1)
    assert (a[1].attack, a[1].max_health) == (1, 50)


def test_simultaneous_triggers_run_highest_attack_first():
    order = []

    @ability("TestLow", Trigger.START_OF_BATTLE)
    def low(ctx, pet, event, lv):
        order.append("low")

    @ability("TestHigh", Trigger.START_OF_BATTLE)
    def high(ctx, pet, event, lv):
        order.append("high")

    try:
        teams = [Team("A", [Pet("TestLow", 1, 1)]), Team("B", [Pet("TestHigh", 9, 1)])]
        ctx = Context(teams, random.Random(0))
        ctx.emit(Event(Trigger.START_OF_BATTLE))
        ctx.resolve()
        assert order == ["high", "low"]
    finally:
        del ABILITIES["TestLow"], ABILITIES["TestHigh"]
