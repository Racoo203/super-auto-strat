class Animal:
    def __init__(self, name, tier, attack, health):
        self.name = name
        self.tier = tier
        self.attack = attack
        self.health = health

    def __str__(self):
        return f"{self.name} (ATK: {self.attack}, HP: {self.health}, TIER: {self.tier})"

class Dog(Animal):
    def __init__(self):
        super().__init__(
            name="Dog",
            tier=1,
            attack=2,
            health=5
        )