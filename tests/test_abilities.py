"""Tier 2-6 abilities, perks, food and tier unlocking."""

import random

import pytest

from sap import config
from sap.battle import Outcome, fight
from sap.events import ABILITIES, Trigger, Event
from sap.game import Game
from sap.items import ITEM_POOL, items_up_to_tier
from sap.perks import PERKS
from sap.pets import PET_POOL, Pet, make_pet
from sap.store import Store
from sap.team import Team


def P(name, a, h, level=1, perk=None):
    """A pet with the given ability name and explicit stats."""
    pet = make_pet(name, level) if name in PET_POOL else Pet(name, a, h)
    pet.base_attack, pet.base_health = a, h
    pet.perk = PERKS[perk] if perk else None
    return pet


def dummy(a=1, h=50, perk=None):
    return P("Dummy", a, h, perk=perk)


def battle(mine, theirs, turns=100, seed=0):
    return fight(Team("A", mine), Team("B", theirs), max_turns=turns, rng=random.Random(seed))


def logtext(result):
    return "\n".join(result.log)


def shop(team_pets, seed=0):
    store = Store(random.Random(seed))
    store.new_visit()
    return store, Team("T", team_pets)


# --- completeness & data ------------------------------------------------------
def test_every_pack_pet_except_sloth_has_an_ability():
    missing = set(PET_POOL) - set(ABILITIES) - {"Sloth"}
    assert not missing, missing


def test_pack_is_configurable_and_defaults_to_turtle():
    assert config.PACK == "Turtle Pack"
    assert len(PET_POOL) == 61


# --- tier unlocks --------------------------------------------------------------
@pytest.mark.parametrize("rnd, tier", [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (10, 5), (11, 6), (30, 6)])
def test_tier_unlocks_every_two_rounds(rnd, tier):
    assert config.max_tier_for_round(rnd) == tier


def test_shop_only_rolls_unlocked_tiers():
    game = Game(seed=0)
    game.start("T")
    for _ in range(20):
        game.reroll() if game.store.credits else game.store.new_visit()
        assert all(p.tier <= 1 for p in game.store.pet_slots if p)
        assert all(s.item.tier <= 1 for s in game.store.item_slots if s)
    game.round = 4
    game._begin_round()  # round 5 -> tier 3
    assert game.store.max_tier == 3
    seen = set()
    for _ in range(40):
        game.store._roll()
        seen |= {p.tier for p in game.store.pet_slots}
    assert max(seen) == 3


def test_items_and_perks_unlock_by_tier():
    assert set(items_up_to_tier(1)) == {"Apple", "Honey"}
    assert "Melon" not in items_up_to_tier(5) and "Melon" in items_up_to_tier(6)
    assert not any(ITEM_POOL[n].tier == 0 for n in items_up_to_tier(6))  # Milk etc. never roll


# --- perks ---------------------------------------------------------------------
def test_garlic_reduces_damage_to_minimum_one():
    r = battle([P("Atk", 5, 50)], [dummy(a=0, h=10, perk="Garlic")], turns=1)
    assert "blocks 2" in logtext(r)
    r = battle([P("Atk", 1, 50)], [dummy(a=0, h=10, perk="Garlic")], turns=1)
    assert "blocks" not in logtext(r)  # 1 damage stays 1


def test_melon_blocks_twenty_once():
    r = battle([P("Atk", 25, 50)], [dummy(a=0, h=10, perk="Melon")], turns=1)
    assert r.outcome is Outcome.DRAW  # 25 - 20 = 5 < 10: survives the hit
    r = battle([P("Atk", 25, 50)], [dummy(a=0, h=10, perk="Melon")], turns=2)
    assert r.outcome is Outcome.WIN  # melon used up, second hit kills


def test_meat_bone_and_steak_add_attack_damage():
    r = battle([dummy(a=2, perk="Meat Bone")], [dummy(a=0, h=50)], turns=1)
    assert "hits Dummy for 5" in logtext(r)
    r = battle([dummy(a=2, perk="Steak")], [dummy(a=0, h=50)], turns=2)
    assert "for 22" in logtext(r) and "uses up Steak" in logtext(r)


def test_chili_splashes_second_enemy():
    back = dummy(a=0, h=5)
    r = battle([dummy(a=1, perk="Chili")], [dummy(a=0, h=50), back], turns=1)
    assert "Chili hits Dummy for 5" in logtext(r)
    assert sum("fainted" in line for line in r.log) == 1


def test_peanut_knocks_out_on_attack():
    r = battle([dummy(a=1, perk="Peanut")], [dummy(a=0, h=50)], turns=1)
    assert r.outcome is Outcome.WIN


def test_coconut_ignores_one_hit():
    r = battle([P("Atk", 50, 50)], [dummy(a=0, h=1, perk="Coconut")], turns=1)
    assert r.outcome is Outcome.DRAW


def test_honey_summons_bee_and_mushroom_revives():
    r = battle([dummy(a=0, h=1, perk="Honey")], [dummy(a=5)], turns=1)
    assert "summons Bee 1/1" in logtext(r)
    r = battle([P("Pig", 4, 1, perk="Mushroom")], [dummy(a=5)], turns=1)
    assert "summons Pig 1/1" in logtext(r)


def test_pancakes_buff_friends_at_start_of_battle():
    r = battle([dummy(perk="Pancakes"), dummy()], [dummy()], turns=0)
    assert "gives Dummy +2/+2" in logtext(r)


def test_cake_raises_sell_value_each_end_turn():
    store, team = shop([dummy(perk="Cake")])
    store.trigger(team, Event(Trigger.END_TURN))
    store.trigger(team, Event(Trigger.END_TURN))
    assert store.sell_value(team[0]) == 1 + 2


# --- food ---------------------------------------------------------------------------
def buy(store, team, name, pet_idx=0, price=3):
    store.stock(name, price)
    return store.buy_item(len(store.item_slots) - 1, team, pet_idx)


def test_cupcake_lasts_until_end_of_battle():
    game = Game(seed=0)
    game.start("T")
    game.team.add(dummy(a=1, h=1))
    buy(game.store, game.team, "Cupcake")
    assert (game.team[0].attack, game.team[0].health) == (4, 4)
    game.end_turn(Team("E", [dummy(a=0, h=1)]))
    assert (game.team[0].attack, game.team[0].health) == (1, 1)


def test_sleeping_pill_faints_and_triggers_faint_ability():
    store, team = shop([P("Cricket", 1, 3)])
    buy(store, team, "Sleeping Pill", price=1)
    assert [p.name for p in team] == ["Zombie Cricket"]


def test_salad_bowl_needs_no_target():
    store, team = shop([dummy(a=1, h=1), dummy(a=1, h=1)])
    buy(store, team, "Salad Bowl", pet_idx=None)
    assert [(p.attack, p.health) for p in team] == [(2, 2), (2, 2)]


def test_canned_food_buffs_current_and_future_shop_pets():
    store, team = shop([dummy()])
    before = store.pet_slots[0].attack
    buy(store, team, "Canned Food", pet_idx=None)
    assert store.pet_slots[0].attack == before + 1
    store._roll()
    p = store.pet_slots[0]
    assert p.attack == PET_POOL[p.name].attack + 1


def test_chocolate_gives_xp_and_can_level_up():
    store, team = shop([P("Fish", 2, 3, level=1), dummy(a=1, h=1)])
    team[0].xp = 1
    buy(store, team, "Chocolate")
    assert team[0].level == 2 and team[1].attack == 2  # Fish level-up fired


# --- tier 2 ---------------------------------------------------------------------
def test_crab_copies_half_health_of_healthiest_friend():
    r = battle([P("Crab", 4, 1), dummy(h=20)], [dummy()], turns=0)
    assert "-> 10" in logtext(r)


def test_flamingo_buffs_two_behind():
    r = battle([P("Flamingo", 3, 1), dummy(), dummy(), dummy()], [dummy(a=5)], turns=1)
    assert logtext(r).count("Flamingo gives Dummy +1/+1") == 2


def test_hedgehog_damages_everyone():
    r = battle([P("Hedgehog", 4, 1), dummy(h=2)], [dummy(a=5, h=2)], turns=1)
    assert r.outcome is Outcome.DRAW


def test_kangaroo_grows_when_friend_ahead_attacks():
    r = battle([dummy(), P("Kangaroo", 2, 3)], [dummy(a=0)], turns=2)
    assert logtext(r).count("Kangaroo gives Kangaroo +1/+1") == 2


def test_peacock_gains_attack_when_hurt():
    r = battle([P("Peacock", 2, 5)], [dummy(a=1)], turns=1)
    assert "Peacock gives Peacock +4/+0" in logtext(r)


def test_rat_gives_opponent_dirty_rats():
    r = battle([P("Rat", 3, 1, level=3)], [dummy(a=5)], turns=1)
    assert logtext(r).count("summons Dirty Rat") == 3


def test_snail_buffs_after_a_loss():
    store, team = shop([P("Snail", 2, 2), dummy(a=1, h=1)])
    store.trigger(team, Event(Trigger.END_TURN))
    assert team[1].health == 1
    store.lost_last_battle = True
    store.trigger(team, Event(Trigger.END_TURN))
    assert team[1].health == 2


def test_spider_summons_a_tier_3_pet():
    r = battle([P("Spider", 2, 1)], [dummy(a=5)], turns=1)
    summoned = [l for l in r.log if "Spider summons" in l]
    assert summoned and summoned[0].split()[2] in {n for n, s in PET_POOL.items() if s.tier == 3}


def test_swan_and_worm_at_start_of_turn():
    store, team = shop([P("Swan", 1, 2, level=2), P("Worm", 1, 3)])
    credits, slots = store.credits, len(store.item_slots)
    store.trigger(team, Event(Trigger.START_TURN))
    assert store.credits == credits + 2
    assert len(store.item_slots) == slots + 1 and store.item_slots[-1].price == 2


# --- tier 3 ---------------------------------------------------------------------
def test_badger_hits_adjacent_including_enemy_front():
    r = battle([P("Badger", 6, 1), dummy()], [dummy(a=5)], turns=1)
    assert logtext(r).count("Badger deals 3") == 2


def test_camel_buffs_friend_behind_when_hurt():
    r = battle([P("Camel", 3, 10), dummy()], [dummy(a=1)], turns=1)
    assert "Camel gives Dummy +1/+2" in logtext(r)


def test_dodo_gives_attack_ahead():
    r = battle([dummy(), P("Dodo", 4, 2)], [dummy()], turns=0)
    assert "Dodo gives Dummy +2/+0" in logtext(r)


def test_dog_and_horse_react_to_summons():
    r = battle([P("Cricket", 1, 1), P("Dog", 3, 2), make_pet("Horse")], [dummy(a=5)], turns=1)
    assert "Dog gives Dog +2/+1" in logtext(r) and "Horse gives Zombie Cricket +1/+0" in logtext(r)


def test_dolphin_snipes_lowest_health_enemy():
    r = battle([P("Dolphin", 4, 3, level=2)], [dummy(h=50), dummy(h=4), dummy(h=6)], turns=0)
    assert logtext(r).count("Dolphin deals 4") == 2


def test_elephant_hurts_friend_behind():
    r = battle([P("Elephant", 3, 7), dummy()], [dummy(a=0)], turns=1)
    assert "Elephant deals 1 to Dummy" in logtext(r)


def test_giraffe_buffs_friend_ahead_at_end_turn():
    store, team = shop([dummy(a=1, h=1), P("Giraffe", 1, 2)])
    store.trigger(team, Event(Trigger.END_TURN))
    assert (team[0].attack, team[0].health) == (2, 2)


def test_ox_gains_melon_when_friend_ahead_faints():
    r = battle([dummy(a=0, h=1), P("Ox", 1, 3)], [dummy(a=5)], turns=1)
    assert "Ox gives Ox Melon" in logtext(r)


def test_rabbit_buffs_pet_that_ate():
    store, team = shop([dummy(a=1, h=1), P("Rabbit", 1, 2)])
    buy(store, team, "Apple")
    assert team[0].health == 3


def test_sheep_summons_two_rams():
    r = battle([P("Sheep", 2, 1)], [dummy(a=5)], turns=1)
    assert logtext(r).count("summons Ram 2/2") == 2


# --- tier 4 ---------------------------------------------------------------------
def test_bison_needs_level_3_friend():
    store, team = shop([P("Bison", 4, 4), P("Sloth", 1, 1, level=3)])
    store.trigger(team, Event(Trigger.END_TURN))
    assert (team[0].attack, team[0].health) == (5, 6)


def test_blowfish_hits_back_when_hurt():
    r = battle([P("Blowfish", 3, 6)], [dummy(a=1)], turns=1)
    assert "Blowfish deals 3" in logtext(r)


def test_deer_summons_bus_with_chili():
    r = battle([P("Deer", 2, 1)], [dummy(a=5)], turns=1)
    assert "summons Bus 5/3 Lv1 (0/2) [Chili]" in logtext(r)


def test_hippo_grows_on_knock_out():
    r = battle([P("Hippo", 4, 50)], [dummy(a=0, h=1)], turns=1)
    assert "Hippo gives Hippo +3/+3" in logtext(r)


def test_parrot_copies_friend_ahead_until_end_of_battle():
    game = Game(seed=0)
    game.start("T")
    game.team.pets = [P("Mosquito", 2, 2), P("Parrot", 4, 2)]
    result = game.end_turn(Team("E", [dummy(a=0, h=50)]))
    assert logtext(result).count("Mosquito deals 1") == 1  # the real Mosquito
    assert "Parrot deals 1" in logtext(result)
    assert game.team[1].ability_name == "Parrot"  # copy wore off


def test_penguin_buffs_leveled_friends():
    store, team = shop([P("Penguin", 1, 3), P("Sloth", 1, 1, level=2), dummy(a=1, h=1)])
    store.trigger(team, Event(Trigger.END_TURN))
    assert (team[1].attack, team[2].attack) == (2, 1)


def test_skunk_cuts_highest_health_enemy():
    r = battle([P("Skunk", 3, 5)], [dummy(h=30), dummy(h=9)], turns=0)
    assert "removes 9 health from Dummy" in logtext(r)


def test_squirrel_discounts_food():
    store, team = shop([P("Squirrel", 2, 5)])
    store.trigger(team, Event(Trigger.START_TURN))
    assert all(s.price == config.ITEM_COST - 1 for s in store.item_slots)


def test_turtle_gives_melon_behind():
    r = battle([P("Turtle", 2, 1), dummy()], [dummy(a=5)], turns=1)
    assert "Turtle gives Dummy Melon" in logtext(r)


def test_whale_swallows_and_releases():
    r = battle([P("Pig", 4, 1), P("Whale", 3, 1)], [dummy(a=5)], turns=1)
    log = logtext(r)
    assert "Whale swallows Pig" in log and "Whale summons Pig 4/1" in log


# --- tier 5 ---------------------------------------------------------------------
def test_armadillo_buffs_everyone():
    a, b = dummy(h=1), dummy(h=1)
    r = battle([P("Armadillo", 2, 6), a], [b], turns=0)
    assert "gives all pets +8 health" in logtext(r)


def test_cow_replaces_food_with_free_milk():
    store, team = shop([])
    store.pet_slots[0] = make_pet("Cow")
    store.buy_pet(0, team)
    assert [(s.item.name, s.price) for s in store.item_slots] == [("Milk", 0), ("Milk", 0)]


def test_crocodile_hits_last_enemy():
    r = battle([P("Crocodile", 8, 4)], [dummy(), dummy(h=8)], turns=0)
    assert "Crocodile deals 8" in logtext(r) and "fainted" in logtext(r)


def test_monkey_buffs_front_friend():
    store, team = shop([dummy(a=1, h=1), dummy(a=1, h=1), P("Monkey", 1, 2)])
    store.trigger(team, Event(Trigger.END_TURN))
    assert (team[0].attack, team[1].attack) == (3, 1)


def test_rhino_chains_knock_outs_and_doubles_on_tier_1():
    r = battle([P("Rhino", 6, 50)], [dummy(a=0, h=1), make_pet("Ant"), make_pet("Pig")], turns=1)
    assert "Rhino deals 8 to Ant" in logtext(r)


def test_rooster_summons_chick_with_half_attack():
    r = battle([P("Rooster", 6, 1)], [dummy(a=5)], turns=1)
    assert "summons Chick 3/1" in logtext(r)


def test_scorpion_gets_peanut_when_bought():
    store, team = shop([])
    store.pet_slots[0] = make_pet("Scorpion")
    store.buy_pet(0, team)
    assert team[0].perk.name == "Peanut"


def test_seal_buffs_friends_when_eating():
    store, team = shop([P("Seal", 3, 8), dummy(a=1, h=1)])
    buy(store, team, "Apple")
    assert team[1].attack == 2


def test_shark_grows_when_friend_faints():
    r = battle([dummy(a=0, h=1), P("Shark", 2, 2)], [dummy(a=5)], turns=1)
    assert "Shark gives Shark +2/+2" in logtext(r)


def test_turkey_buffs_summoned_friend():
    r = battle([P("Cricket", 1, 1), P("Turkey", 3, 4)], [dummy(a=5)], turns=1)
    assert "Turkey gives Zombie Cricket +3/+1" in logtext(r)


# --- tier 6 ---------------------------------------------------------------------
def test_boar_buffs_before_attack():
    r = battle([P("Boar", 10, 6)], [dummy(a=0)], turns=1)
    assert "Boar gives Boar +4/+2" in logtext(r)


def test_cat_doubles_food_twice_per_turn():
    store, team = shop([dummy(a=1, h=1), P("Cat", 4, 5)])
    for _ in range(3):
        buy(store, team, "Apple")
    assert team[0].attack == 1 + 2 + 2 + 1


def test_dragon_buffs_when_tier_1_bought():
    store, team = shop([P("Dragon", 6, 8), dummy(a=1, h=1)])
    store.pet_slots[0] = make_pet("Sloth")
    store.buy_pet(0, team)
    assert team[1].attack == 2


def test_fly_summons_zombie_flies_but_not_for_zombie_flies():
    r = battle([dummy(a=0, h=1), P("Fly", 5, 50)], [dummy(a=50, h=50)], turns=3)
    assert logtext(r).count("summons Zombie Fly") == 1


def test_gorilla_gains_coconut_when_hurt():
    r = battle([P("Gorilla", 7, 10)], [dummy(a=1)], turns=2)
    assert "Gorilla gives Gorilla Coconut" in logtext(r) and "Coconut blocks" in logtext(r)


def test_leopard_hits_for_half_attack():
    r = battle([P("Leopard", 10, 4)], [dummy()], turns=0)
    assert "Leopard deals 5" in logtext(r)


def test_mammoth_buffs_all_friends_on_faint():
    r = battle([P("Mammoth", 4, 1), dummy(), dummy()], [dummy(a=5)], turns=1)
    assert logtext(r).count("Mammoth gives Dummy +2/+2") == 2


def test_snake_snipes_when_friend_ahead_attacks():
    r = battle([dummy(), P("Snake", 6, 6)], [dummy()], turns=1)
    assert "Snake deals 5" in logtext(r)


def test_tiger_repeats_friend_ahead_in_battle():
    r = battle([P("Mosquito", 2, 2), P("Tiger", 6, 4)], [dummy(), dummy()], turns=0)
    assert logtext(r).count("Mosquito deals 1") == 2


def test_wolverine_triggers_every_four_friend_hurts():
    team = [dummy(a=0), dummy(a=0), dummy(a=0), dummy(a=0), P("Wolverine", 5, 4)]
    r = battle([P("Hedgehog", 0, 1)] + team[:3] + [team[4]], [dummy(a=5, h=50)], turns=1)
    assert "Wolverine removes 2 health" in logtext(r)


def test_whale_can_swallow_and_release_a_token():
    """Regression: a swallowed Bee (not in PET_POOL) used to crash with KeyError on release."""
    result = battle([P("Bee", 1, 1), P("Whale", 1, 1)], [dummy(a=5)])
    assert "Whale swallows Bee" in logtext(result)
    assert "Whale summons Bee" in logtext(result)
