from Animal import Animal

class Ant(Animal):
    def __init__(
        self,
        animal_tag : str = 'Ant',
        tier : int = 1,
        health : int = 2,
        attack : int = 2
    ):
        self.animal_tag = animal_tag
        self.tier = tier
        self.health = health
        self.attack = attack
    