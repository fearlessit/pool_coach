import math
import numpy as np

class BilliardBall:
    def __init__(self, xmin, ymin, xmax, ymax, detection_confidence):
        self.diameter:float = math.dist((xmin, ymin), (xmax, ymax))
        self.xmin:int = xmin
        self.ymin:int = ymin
        self.xmax:int = xmax
        self.ymax:int = ymax
        self.detection_confidence:float = detection_confidence
        self.x:int = (xmin+xmax) / 2.0
        self.y:int = (ymin+ymax) / 2.0

    def has_neighbour_ball(self, balls):
        max_diameter = max(b.diameter for b in balls) if balls else 0.0
        for ball in balls:
            distance = np.sqrt((self.x - ball.x)**2 + (self.y - ball.y)**2)
            if 2.5 * max_diameter > distance > 0.0:
                return True
        return False

    def get_diameter(self):
        return self.diameter
