from typing import List, Tuple
from structures.generator.generator import Generator


class Person(Generator):
    name: str
    age: int
    gender: str


class People(Generator):
    person_set: List[Person]


class MysticalMonster(Generator):
    name: str
    age: int
    powers: List[str]
    powers_description: List[str]
    spirit: str
    back_story: str


class Star(Generator):
    position: Tuple[float, float]
    luminosity: float
    special_characteristics: Tuple[str, str, str]
    near_stars: List[str]

    def __init__(self, description, **kwargs):
        super().__init__(**kwargs)
        self.description = description


class Topics(Generator):
    topics_list: List[str]
    topics_descriptions: List[str]
