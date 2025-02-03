class Food:
    def __init__(self, name, effect):
        self.name = name
        self.effect = effect

    def __str__(self):
        return f"{self.name} (Effect: {self.effect})"

class Meat(Food):
    def __init__(self):
        super().__init__("Meat", "Boost Attack")
