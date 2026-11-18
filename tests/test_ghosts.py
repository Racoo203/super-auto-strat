"""Ghost opponents (sap/ghosts.py) and the env / game hooks they use."""

import pickle
import random

import numpy as np
import pytest

from sap.battle import Outcome
from sap.game import Game
from sap.ghosts import RANDOM_SOURCE, Ghost, GhostOpponent, GhostPool
from sap.pets import Pet, make_pet
from sap.team import Team


def team(*names, name="T"):
    return Team(name, [make_pet(n) for n in names])


def ghost(rnd, source=0, *names):
    return Ghost(team(*(names or ("Fish",))), rnd, source)


# --- pool -----------------------------------------------------------------------
def test_sample_prefers_the_exact_round_then_the_nearest_earlier_one():
    pool = GhostPool()
    pool.add([ghost(2, 0, "Ant"), ghost(5, 0, "Dog")], random.Random(0))
    rng = random.Random(0)
    assert pool.sample(5, rng).team.pets[0].name == "Dog"
    assert pool.sample(4, rng).team.pets[0].name == "Ant"  # no round 3/4 ghosts: nearest earlier
    assert pool.sample(1, rng) is None                       # nothing at or before round 1


def test_empty_pool_and_source_filter():
    pool = GhostPool()
    assert pool.sample(3, random.Random(0)) is None
    pool.add([ghost(3, 0, "Ant"), ghost(3, 1, "Dog")], random.Random(0))
    rng = random.Random(0)
    assert {pool.sample(3, rng, sources={1}).source for _ in range(20)} == {1}
    assert pool.sample(3, rng, sources={7}) is None


def test_empty_teams_are_not_ghosts():
    pool = GhostPool()
    assert pool.add([Ghost(Team("T"), 1, 0)], random.Random(0)) == 0 and len(pool) == 0


def test_pool_stores_copies_under_a_ghost_name():
    t = team("Ant")
    pool = GhostPool()
    pool.add([Ghost(t, 1, 3)], random.Random(0))
    g = pool.sample(1, random.Random(0))
    t.pets[0].base_attack = 40
    assert g.team.pets[0].attack != 40 and g.team.name == "Ghost p3 r1"


def test_recent_is_fifo_and_older_ghosts_go_to_the_history_reservoir():
    pool = GhostPool(recent_per_round=2, history_per_round=3)
    rng = random.Random(0)
    pool.add([Ghost(Team("T", [Pet(f"P{i}", 1, 1)]), 1, 0) for i in range(10)], rng)
    bucket = pool.rounds[1]
    assert [g.team.pets[0].name for g in bucket.recent] == ["P8", "P9"]
    assert len(bucket.history) == 3 and len(pool) == 5
    assert all(g.team.pets[0].name not in ("P8", "P9") for g in bucket.history)
    # history_prob=1 draws from the reservoir
    assert pool.sample(1, random.Random(1), history_prob=1.0).team.pets[0].name not in ("P8", "P9")


def test_pool_and_opponent_pickle():
    pool = GhostPool()
    pool.add([ghost(1, 0, "Ant", "Dog")], random.Random(0))
    back = pickle.loads(pickle.dumps(GhostOpponent(pool, sources={0})))
    assert len(back.pool) == 1 and back.sources == {0}


# --- opponent_fn ---------------------------------------------------------------
def started_game(seed=0):
    g = Game(seed=seed)
    g.start("Agent")
    return g


def test_ghost_opponent_returns_a_ghost_and_records_its_source():
    pool = GhostPool()
    pool.add([ghost(1, 4, "Dog")], random.Random(0))
    fn = GhostOpponent(pool)
    opp = fn(started_game())
    assert opp.pets[0].name == "Dog" and fn.last_source == 4


def test_ghost_opponent_falls_back_to_a_random_team():
    fn = GhostOpponent(GhostPool())
    opp = fn(started_game())
    assert len(opp) >= 1 and fn.last_source == RANDOM_SOURCE
    pool = GhostPool()
    pool.add([ghost(1)], random.Random(0))
    always_random = GhostOpponent(pool, random_prob=1.0)
    always_random(started_game())
    assert always_random.last_source == RANDOM_SOURCE


def test_ghost_opponent_is_deterministic_for_a_seeded_game():
    pool = GhostPool()
    pool.add([Ghost(Team("T", [Pet(f"P{i}", 1, 1)]), 1, 0) for i in range(10)], random.Random(0))
    picks = [GhostOpponent(pool, history_prob=0.5)(started_game(seed=7)).pets[0].name for _ in range(3)]
    assert len(set(picks)) == 1


# --- game: end_turn = finish_shop + play_battle + record_result -----------------
def test_end_turn_is_the_composition_of_the_atomic_steps():
    a, b = started_game(seed=3), started_game(seed=3)
    for g in (a, b):
        g.team.add(make_pet("Fish"))
    opponent = team("Ant", name="E")
    ra = a.end_turn(opponent)
    b.finish_shop()
    rb = b.play_battle(opponent)
    b.record_result(rb)
    assert (ra.outcome, ra.log) == (rb.outcome, rb.log)
    assert (a.round, a.wins, a.losses, a.lives) == (b.round, b.wins, b.losses, b.lives)


def test_finish_shop_returns_the_team_after_end_of_turn_abilities():
    g = started_game()
    g.team.pets = [Pet("Front", 1, 1), make_pet("Giraffe")]  # Giraffe: end turn -> friend ahead +1/+1
    fought = g.finish_shop()
    assert (fought.pets[0].attack, fought.pets[0].health) == (2, 2)


def test_play_battle_empty_team_loses():
    g = started_game()
    assert g.play_battle(team("Ant")).outcome is Outcome.LOSS


# --- env ------------------------------------------------------------------------
gym = pytest.importorskip("gymnasium")

from sap.env import ACTIONS, SapEnv  # noqa: E402

END_TURN = ACTIONS.index(("end_turn", None, None))
BUY_0 = ACTIONS.index(("buy_pet", 0, None))


def test_env_fights_the_ghost_and_reports_its_source():
    pool = GhostPool()
    pool.add([Ghost(Team("T", [Pet("Wall", 50, 50)]), 1, 2)], random.Random(0))
    env = SapEnv(opponent_fn=GhostOpponent(pool))
    env.reset(seed=0)
    env.step(BUY_0)
    *_, info = env.step(END_TURN)
    assert info["opponent_source"] == 2 and info["battle"] == "loss"
    assert "Wall" in info["battle_record"]["opponent"] and info["battle_record"]["opponent_source"] == 2


def test_env_keeps_one_snapshot_per_battle_across_resets():
    env = SapEnv()
    env.reset(seed=0)
    env.step(BUY_0)
    env.step(END_TURN)
    env.step(END_TURN)
    env.reset(seed=1)
    env.step(END_TURN)
    snaps = env.pop_snapshots()
    assert [r for r, _ in snaps] == [1, 2, 1]
    assert len(snaps[0][1]) == 1 and env.pop_snapshots() == []


def test_vector_env_harvest_and_push_opponents():
    from agents.ppo.config import PPOConfig
    from agents.ppo.envs import make_vec_env
    from agents.ppo.selfplay import harvest, make_pool, opponent_sources, push_opponents

    envs, obs, mask = make_vec_env(2, async_envs=False, seed=0)
    seats = np.array([0, 1])
    envs.step(np.array([BUY_0, BUY_0]))
    envs.step(np.array([END_TURN, END_TURN]))
    pool = make_pool(PPOConfig())
    assert harvest(envs, seats, pool, random.Random(0)) == 2
    assert {g.source for g in pool.rounds[1].recent} == {0, 1}

    push_opponents(envs, seats, pool, PPOConfig(n_policies=2, opponents="others", random_opponent_prob=0.0))
    fns = [e.unwrapped.opponent_fn for e in envs.envs]  # get_attr would call them
    assert [f.sources for f in fns] == [{1}, {0}]
    _, _, _, _, info = envs.step(np.array([END_TURN, END_TURN]))
    assert list(info["opponent_source"]) == [1, 0]  # each seat fought the other policy's ghost
    envs.close()
    assert opponent_sources(0, 1, "others") is None and opponent_sources(1, 3, "all") is None
