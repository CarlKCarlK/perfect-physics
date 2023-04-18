# use sprepr to load from a file
import copy
import uuid
from dataclasses import dataclass

from sympy import simplify, sqrt


@dataclass
class Circle:
    x: float
    y: float
    r: float
    vx: float
    vy: float
    m: float = 1
    id: str = None

    def __post_init__(self):
        if self.id is None:
            self.id = uuid.uuid4()

    def tick(self, t):
        if isinstance(t, float):
            self.x = float(self.x + self.vx * t)
            self.y = float(self.y + self.vy * t)
        else:
            self.x = simplify(self.x + self.vx * t)
            self.y = simplify(self.y + self.vy * t)

    def tick_clone(self, t):
        circle = self.clone()
        if isinstance(t, float):
            circle.x = float(self.x + self.vx * t)
            circle.y = float(self.y + self.vy * t)
        else:
            circle.x = simplify(self.x + self.vx * t)
            circle.y = simplify(self.y + self.vy * t)
        return circle

    def energy(self):
        return simplify(self.m * (self.vx**2 + self.vy**2) / 2)

    def __eq__(self, other):
        return self.id == other.id

    def find_circle(self, circle_list):
        found = [c for c in circle_list if c == self]
        if len(found) == 1:
            return found[0]
        else:
            raise ValueError(f"Found {len(found)} circles with id {self.id}")

    def float_clone(self):
        return Circle(
            float(self.x),
            float(self.y),
            float(self.r),
            float(self.vx),
            float(self.vy),
            float(self.m),
            self.id,
        )

    def clone(self):
        return copy.copy(self)

    def distance(self, other):
        if not isinstance(other, Circle):
            raise TypeError("Argument 'other' must be an instance of Circle.")
        return simplify(sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2))
