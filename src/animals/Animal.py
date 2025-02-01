class Animal:
    def __init__(
        self,
        animal_tag : str,
        tier : int,
        health : int,
        attack : int
    ):
        self.animal_tag = animal_tag
        self.tier = tier
        self.health = health
        self.attack = attack
    