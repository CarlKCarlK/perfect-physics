import uuid
from dataclasses import dataclass
from sympy import S


@dataclass
class Wall:
    x0: float
    y0: float
    x1: float
    y1: float
    id: str = None

    def __post_init__(self):
        if self.id is None:
            self.id = uuid.uuid4()

    def __eq__(self, other):
        return self.id == other.id

    def find_wall(self, wall_list):
        found = [w for w in wall_list if w == self]
        if len(found) == 1:
            return found[0]
        else:
            raise ValueError(f"Found {len(found)} walls with id {self.id}")

    def float_clone(self):
        return Wall(
            float(self.x0),
            float(self.y0),
            float(self.x1),
            float(self.y1),
            self.id,
        )

    @property
    def vx(self):
        return S.Zero

    @property
    def vy(self):
        return S.Zero
