"""Pet and perk abilities. One decorated function per (pet, trigger).

Each pet effect gets (ctx, pet, ev, lv): `lv` is the level to act at (usually pet.level; a Parrot
or Tiger can change it). Effects only act through `ctx` verbs, so faints, knock outs, hurt chains
and summons all flow through the event queue.

Scaling: the scraped text gives levels 1 and 2 (level 3's text repeats level 2's), so every
ability scales linearly: level 3 = 3x level 1.

Sloth is the only pet with no ability.
"""

from sap.events import Context, Event, Trigger, ability, has_ability, perk_ability
from sap.perks import PERKS
from sap.pets import Pet, make_pet, pets_of_tier, xp_for


def token(name: str, attack: int, health: int, level: int = 1, perk: str | None = None) -> Pet:
    """A summoned-only pet (Zombie Cricket, Ram, Bee...)."""
    return Pet(name, attack, health, xp=xp_for(level), perk=PERKS[perk] if perk else None)


# =============================================================================
# Tier 1
# =============================================================================
@ability("Ant", Trigger.FAINT)
def ant(ctx: Context, pet: Pet, ev: Event, lv: int) -> None:
    """Faint -> give one random friend +L/+L."""
    for friend in ctx.random_friends(pet, 1, ev.team):
        ctx.buff(friend, lv, lv, source=pet)


@ability("Beaver", Trigger.SELL)
def beaver(ctx, pet, ev, lv):
    """Sell -> give two random friends +L attack."""
    for friend in ctx.random_friends(pet, 2):
        ctx.buff(friend, attack=lv, source=pet)


@ability("Cricket", Trigger.FAINT)
def cricket(ctx, pet, ev, lv):
    """Faint -> summon one L/L Zombie Cricket in its place."""
    ctx.summon(ev.team, token("Zombie Cricket", lv, lv, lv), ev.index, source=pet)


@ability("Duck", Trigger.SELL)
def duck(ctx, pet, ev, lv):
    """Sell -> give shop pets +L health."""
    for shop_pet in ctx.store.pet_slots if ctx.store else []:
        if shop_pet:
            ctx.buff(shop_pet, health=lv, source=pet)


@ability("Fish", Trigger.LEVEL_UP)
def fish(ctx, pet, ev, lv):
    """Level up -> give two random friends +X/+X, X = the level it just left (1 at Lv2, 2 at Lv3)."""
    bonus = lv - 1
    for friend in ctx.random_friends(pet, 2):
        ctx.buff(friend, bonus, bonus, source=pet)


@ability("Horse", Trigger.FRIEND_SUMMONED)
def horse(ctx, pet, ev, lv):
    """Friend summoned (or bought) -> give it +L attack until end of battle."""
    ctx.buff(ev.pet, attack=lv, source=pet, temporary=True)


@ability("Mosquito", Trigger.START_OF_BATTLE)
def mosquito(ctx, pet, ev, lv):
    """Start of battle -> deal 1 damage to L random enemies."""
    for enemy in ctx.random_enemies(pet, lv):
        ctx.damage(enemy, 1, source=pet)


@ability("Otter", Trigger.BUY)
def otter(ctx, pet, ev, lv):
    """Buy -> give L random friends +1 health."""
    for friend in ctx.random_friends(pet, lv):
        ctx.buff(friend, health=1, source=pet)


@ability("Pig", Trigger.SELL)
def pig(ctx, pet, ev, lv):
    """Sell -> gain +L extra credits."""
    ctx.gain_gold(lv, source=pet)


@ability("Pigeon", Trigger.SELL)
def pigeon(ctx, pet, ev, lv):
    """Sell -> stock L free Bread Crumbs."""
    for _ in range(lv):
        ctx.stock("Bread Crumbs", 0, source=pet)


# =============================================================================
# Tier 2
# =============================================================================
@ability("Crab", Trigger.START_OF_BATTLE)
def crab(ctx, pet, ev, lv):
    """Start of battle -> copy 50%xL of the most healthy friend's health."""
    friends = ctx.friends(pet)
    if friends:
        best = max(friends, key=lambda p: p.health)
        ctx.set_health(pet, best.health * lv // 2)
        ctx.say(f"{pet.name} copies health from {best.name} -> {pet.health}")


@ability("Flamingo", Trigger.FAINT)
def flamingo(ctx, pet, ev, lv):
    """Faint -> give the two nearest friends behind +L/+L."""
    for friend in ctx.behind(pet, 2, ev):
        ctx.buff(friend, lv, lv, source=pet)


@ability("Hedgehog", Trigger.FAINT)
def hedgehog(ctx, pet, ev, lv):
    """Faint -> deal 2xL damage to all pets."""
    for team in ctx.teams:
        for other in list(team):
            ctx.damage(other, 2 * lv, source=pet)


@ability("Kangaroo", Trigger.FRIEND_AHEAD_ATTACKS)
def kangaroo(ctx, pet, ev, lv):
    """Friend ahead attacks -> gain +L/+L."""
    ctx.buff(pet, lv, lv, source=pet)


@ability("Peacock", Trigger.HURT)
def peacock(ctx, pet, ev, lv):
    """Hurt -> gain +4xL attack."""
    ctx.buff(pet, attack=4 * lv, source=pet)


@ability("Rat", Trigger.FAINT)
def rat(ctx, pet, ev, lv):
    """Faint -> summon L 1/1 Dirty Rats up front for the opponent."""
    enemy = ctx.enemy_team(ev.team)
    for _ in range(lv):
        ctx.summon(enemy, token("Dirty Rat", 1, 1), 0, source=pet)


@ability("Snail", Trigger.END_TURN)
def snail(ctx, pet, ev, lv):
    """End turn -> if you lost last battle, give friends +L health."""
    if ctx.store and ctx.store.lost_last_battle:
        for friend in ctx.friends(pet):
            ctx.buff(friend, health=lv, source=pet)


@ability("Spider", Trigger.FAINT)
def spider(ctx, pet, ev, lv):
    """Faint -> summon one random level-L tier 3 pet as a 2L/2L."""
    names = pets_of_tier(3)
    if names:
        summoned = make_pet(ctx.rng.choice(names), level=lv)
        summoned.base_attack = summoned.base_health = 2 * lv
        ctx.summon(ev.team, summoned, ev.index, source=pet)


@ability("Swan", Trigger.START_TURN)
def swan(ctx, pet, ev, lv):
    """Start of turn -> gain +L gold."""
    ctx.gain_gold(lv, source=pet)


@ability("Worm", Trigger.START_TURN)
def worm(ctx, pet, ev, lv):
    """Start of turn -> stock one 2-gold Apple (Better Apple at Lv2, Best Apple at Lv3)."""
    ctx.stock(("Apple", "Better Apple", "Best Apple")[lv - 1], 2, source=pet)


# =============================================================================
# Tier 3
# =============================================================================
@ability("Badger", Trigger.FAINT)
def badger(ctx, pet, ev, lv):
    """Faint -> deal 50%xL of its attack to adjacent pets (the enemy front pet, if it was in front)."""
    amount = pet.attack * lv // 2
    targets = ctx.ahead(pet, 1, ev, ev.team) + ctx.behind(pet, 1, ev, ev.team)
    if ev.index == 0:
        targets += ctx.enemies(pet, ev.team)[:1]
    for target in targets:
        ctx.damage(target, amount, source=pet)


@ability("Camel", Trigger.HURT)
def camel(ctx, pet, ev, lv):
    """Hurt -> give the nearest friend behind +L/+2L."""
    for friend in ctx.behind(pet, 1):
        ctx.buff(friend, lv, 2 * lv, source=pet)


@ability("Dodo", Trigger.START_OF_BATTLE)
def dodo(ctx, pet, ev, lv):
    """Start of battle -> give 50%xL of its attack to the nearest friend ahead."""
    for friend in ctx.ahead(pet, 1):
        ctx.buff(friend, attack=pet.attack * lv // 2, source=pet)


@ability("Dog", Trigger.FRIEND_SUMMONED)
def dog(ctx, pet, ev, lv):
    """Friend summoned -> gain +2L/+L until end of battle."""
    ctx.buff(pet, 2 * lv, lv, source=pet, temporary=True)


@ability("Dolphin", Trigger.START_OF_BATTLE)
def dolphin(ctx, pet, ev, lv):
    """Start of battle -> deal 4 damage to the lowest health enemy. Triggers L times."""
    for _ in range(lv):
        enemies = ctx.enemies(pet)
        if enemies:
            ctx.damage(min(enemies, key=lambda p: p.health), 4, source=pet)


@ability("Elephant", Trigger.AFTER_ATTACK)
def elephant(ctx, pet, ev, lv):
    """After attack -> deal 1 damage to the nearest friend behind. Triggers L times."""
    for _ in range(lv):
        for friend in ctx.behind(pet, 1):
            ctx.damage(friend, 1, source=pet)


@ability("Giraffe", Trigger.END_TURN)
def giraffe(ctx, pet, ev, lv):
    """End turn -> give the nearest L friends ahead +1/+1."""
    for friend in ctx.ahead(pet, lv):
        ctx.buff(friend, 1, 1, source=pet)


@ability("Ox", Trigger.FRIEND_AHEAD_FAINTS)
def ox(ctx, pet, ev, lv):
    """Friend ahead faints -> gain Melon and +L attack. Works L times per turn."""
    if ctx.use(pet, "ox", lv):
        ctx.give_perk(pet, "Melon", source=pet)
        ctx.buff(pet, attack=lv, source=pet)


@ability("Rabbit", Trigger.FRIEND_ATE_FOOD)
def rabbit(ctx, pet, ev, lv):
    """Friendly ate food -> give them +L health. Works 4 times per turn."""
    if ctx.use(pet, "rabbit", 4):
        ctx.buff(ev.pet, health=lv, source=pet)


@ability("Sheep", Trigger.FAINT)
def sheep(ctx, pet, ev, lv):
    """Faint -> summon two 2L/2L Rams."""
    for _ in range(2):
        ctx.summon(ev.team, token("Ram", 2 * lv, 2 * lv, lv), ev.index, source=pet)


# =============================================================================
# Tier 4
# =============================================================================
@ability("Bison", Trigger.END_TURN)
def bison(ctx, pet, ev, lv):
    """End turn -> if this has a level 3 friend, gain +L/+2L."""
    if any(f.level >= 3 for f in ctx.friends(pet)):
        ctx.buff(pet, lv, 2 * lv, source=pet)


@ability("Blowfish", Trigger.HURT)
def blowfish(ctx, pet, ev, lv):
    """Hurt -> deal 3xL damage to one random enemy."""
    for enemy in ctx.random_enemies(pet, 1):
        ctx.damage(enemy, 3 * lv, source=pet)


@ability("Deer", Trigger.FAINT)
def deer(ctx, pet, ev, lv):
    """Faint -> summon one 5L/3L Bus with Chili."""
    ctx.summon(ev.team, token("Bus", 5 * lv, 3 * lv, lv, perk="Chili"), ev.index, source=pet)


@ability("Hippo", Trigger.KNOCK_OUT)
def hippo(ctx, pet, ev, lv):
    """Knock out -> gain +3L/+3L."""
    ctx.buff(pet, 3 * lv, 3 * lv, source=pet)


@ability("Parrot", Trigger.END_TURN)
def parrot(ctx, pet, ev, lv):
    """End turn -> copy the ability of the nearest pet ahead, at level L, until end of battle."""
    for friend in ctx.ahead(pet, 1):
        name = friend.ability_name
        if name != "Parrot":
            pet.state["copied"] = (name, lv)
            ctx.say(f"{pet.name} copies {name}'s ability (Lv{lv})")


@ability("Penguin", Trigger.END_TURN)
def penguin(ctx, pet, ev, lv):
    """End turn -> give two level 2+ friends +L/+L."""
    for friend in ctx.pick([f for f in ctx.friends(pet) if f.level >= 2], 2):
        ctx.buff(friend, lv, lv, source=pet)


@ability("Skunk", Trigger.START_OF_BATTLE)
def skunk(ctx, pet, ev, lv):
    """Start of battle -> reduce the highest health enemy's health by 33%xL."""
    enemies = ctx.enemies(pet)
    if enemies:
        target = max(enemies, key=lambda p: p.health)
        ctx.reduce_health(target, target.health * 33 * lv // 100, source=pet)


@ability("Squirrel", Trigger.START_TURN)
def squirrel(ctx, pet, ev, lv):
    """Start of turn -> discount all shop food by L gold."""
    if ctx.store:
        ctx.store.discount_items(lv)
        ctx.say(f"{pet.name} discounts shop food by {lv}")


@ability("Turtle", Trigger.FAINT)
def turtle(ctx, pet, ev, lv):
    """Faint -> give Melon to the nearest L friends behind."""
    for friend in ctx.behind(pet, lv, ev):
        ctx.give_perk(friend, "Melon", source=pet)


@ability("Whale", Trigger.START_OF_BATTLE)
def whale_swallow(ctx, pet, ev, lv):
    """Start of battle -> swallow the nearest friend ahead..."""
    for friend in ctx.ahead(pet, 1):
        ctx.team_of(pet).pets.remove(friend)  # leaves the team without fainting
        pet.state["swallowed"] = friend.name
        ctx.say(f"{pet.name} swallows {friend.name}")


@ability("Whale", Trigger.FAINT)
def whale_release(ctx, pet, ev, lv):
    """...and release it at level L when the Whale faints."""
    name = pet.state.pop("swallowed", None)
    if name:
        ctx.summon(ev.team, make_pet(name, level=lv), ev.index, source=pet)


# =============================================================================
# Tier 5
# =============================================================================
@ability("Armadillo", Trigger.START_OF_BATTLE)
def armadillo(ctx, pet, ev, lv):
    """Start of battle -> give ALL other pets (both teams) +8L health."""
    for team in ctx.teams:
        for other in team:
            if other is not pet:
                ctx.buff(other, health=8 * lv)
    ctx.say(f"{pet.name} gives all pets +{8 * lv} health")


@ability("Cow", Trigger.BUY)
def cow(ctx, pet, ev, lv):
    """Buy -> replace the food shop with two free Milk (Better Milk at Lv2, Best Milk at Lv3)."""
    if ctx.store:
        ctx.store.item_slots = []
        for _ in range(2):
            ctx.stock(("Milk", "Better Milk", "Best Milk")[lv - 1], 0, source=pet)


@ability("Crocodile", Trigger.START_OF_BATTLE)
def crocodile(ctx, pet, ev, lv):
    """Start of battle -> deal 8 damage to the last enemy. Triggers L times."""
    for _ in range(lv):
        enemies = ctx.enemies(pet)
        if enemies:
            ctx.damage(enemies[-1], 8, source=pet)


@ability("Monkey", Trigger.END_TURN)
def monkey(ctx, pet, ev, lv):
    """End turn -> give the front-most friend +2L/+2L."""
    friends = ctx.friends(pet)
    if friends:
        ctx.buff(friends[0], 2 * lv, 2 * lv, source=pet)


@ability("Rhino", Trigger.KNOCK_OUT)
def rhino(ctx, pet, ev, lv):
    """Knock out -> deal 4xL damage to the first enemy (double against tier 1 pets)."""
    enemies = ctx.enemies(pet)
    if enemies:
        first = enemies[0]
        ctx.damage(first, 4 * lv * (2 if first.tier == 1 else 1), source=pet)


@ability("Rooster", Trigger.FAINT)
def rooster(ctx, pet, ev, lv):
    """Faint -> summon L Chicks with 1 health and 50% of this pet's attack."""
    for _ in range(lv):
        ctx.summon(ev.team, token("Chick", max(1, pet.attack // 2), 1, lv), ev.index, source=pet)


@ability("Scorpion", Trigger.SUMMONED)
def scorpion(ctx, pet, ev, lv):
    """Summoned (or bought) -> gain Peanut."""
    ctx.give_perk(pet, "Peanut", source=pet)


@ability("Seal", Trigger.EAT_FOOD)
def seal(ctx, pet, ev, lv):
    """Eats food -> give three random friends +L attack."""
    for friend in ctx.random_friends(pet, 3):
        ctx.buff(friend, attack=lv, source=pet)


@ability("Shark", Trigger.FRIEND_FAINTS)
def shark(ctx, pet, ev, lv):
    """Friend faints -> gain +2L/+2L."""
    ctx.buff(pet, 2 * lv, 2 * lv, source=pet)


@ability("Turkey", Trigger.FRIEND_SUMMONED)
def turkey(ctx, pet, ev, lv):
    """Friend summoned -> give it +3L/+L."""
    ctx.buff(ev.pet, 3 * lv, lv, source=pet)


# =============================================================================
# Tier 6
# =============================================================================
@ability("Boar", Trigger.BEFORE_ATTACK)
def boar(ctx, pet, ev, lv):
    """Before attack -> gain +4L/+2L."""
    ctx.buff(pet, 4 * lv, 2 * lv, source=pet)


@ability("Cat", Trigger.PASSIVE)
def cat(ctx, pet, ev, lv):
    """Food gives (L+1)x stats, twice per turn. Applied in sap.items.food_buff."""


@ability("Dragon", Trigger.FRIEND_BOUGHT)
def dragon(ctx, pet, ev, lv):
    """Tier 1 friend bought -> give friends +L/+L."""
    if ev.pet.tier == 1:
        for friend in ctx.friends(pet):
            ctx.buff(friend, lv, lv, source=pet)


@ability("Fly", Trigger.FRIEND_FAINTS)
def fly(ctx, pet, ev, lv):
    """Friend faints -> summon one 4L/4L Zombie Fly in its place. Works 3 times per turn."""
    if ev.pet.name != "Zombie Fly" and ctx.use(pet, "fly", 3):
        ctx.summon(ev.team, token("Zombie Fly", 4 * lv, 4 * lv, lv), ev.index, source=pet)


@ability("Gorilla", Trigger.HURT)
def gorilla(ctx, pet, ev, lv):
    """Hurt -> gain Coconut. Works L times per battle."""
    if ctx.use(pet, "gorilla", lv):
        ctx.give_perk(pet, "Coconut", source=pet)


@ability("Leopard", Trigger.START_OF_BATTLE)
def leopard(ctx, pet, ev, lv):
    """Start of battle -> deal 50% of its attack to L random enemies."""
    for enemy in ctx.random_enemies(pet, lv):
        ctx.damage(enemy, pet.attack // 2, source=pet)


@ability("Mammoth", Trigger.FAINT)
def mammoth(ctx, pet, ev, lv):
    """Faint -> give all friends +2L/+2L."""
    for friend in ctx.friends(pet, ev.team):
        ctx.buff(friend, 2 * lv, 2 * lv, source=pet)


@ability("Snake", Trigger.FRIEND_AHEAD_ATTACKS)
def snake(ctx, pet, ev, lv):
    """Friend ahead attacks -> deal 5xL damage to one random enemy."""
    for enemy in ctx.random_enemies(pet, 1):
        ctx.damage(enemy, 5 * lv, source=pet)


@ability("Tiger", Trigger.PASSIVE)
def tiger(ctx, pet, ev, lv):
    """In battle, the friend ahead repeats its ability at level L. Applied in Context._dispatch."""


@ability("Wolverine", Trigger.FRIEND_HURT)
def wolverine(ctx, pet, ev, lv):
    """Every four friends hurt -> remove 2xL health from all enemies."""
    pet.state["uses:wolverine"] = count = pet.state.get("uses:wolverine", 0) + 1
    if count % 4 == 0:
        for enemy in ctx.enemies(pet):
            ctx.reduce_health(enemy, 2 * lv, source=pet)


# =============================================================================
# Perks that react to moments (damage perks are data in sap/perks.py)
# =============================================================================
@perk_ability("Honey", Trigger.FAINT)
def honey(ctx, pet, ev):
    ctx.summon(ev.team, token("Bee", 1, 1), ev.index, source=pet)


@perk_ability("Mushroom", Trigger.FAINT)
def mushroom(ctx, pet, ev):
    revived = Pet(pet.name, 1, 1, xp=pet.xp, tier=pet.tier)
    ctx.summon(ev.team, revived, ev.index, source=pet)


@perk_ability("Cake", Trigger.END_TURN)
def cake(ctx, pet, ev):
    pet.sell_bonus += 1
    ctx.say(f"{pet.name}'s Cake: sell value +1")


@perk_ability("Pancakes", Trigger.START_OF_BATTLE)
def pancakes(ctx, pet, ev):
    for friend in ctx.friends(pet):
        ctx.buff(friend, 2, 2, source=pet)
