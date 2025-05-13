import numpy as np
from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    # return as list []
    def to_list(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])


class BoundingBoxes(BaseModel):
    boxes: list[BoundingBox]
