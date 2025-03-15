import math
import numpy as np

from conf.constants import INIT_FPS
from utils.last_values import LastValues


class BilliardBall:
    def __init__(self, ball_id, xmin, ymin, xmax, ymax, detection_confidence,target_temp=None):
        self.detection_confidence = detection_confidence
        self.diameter:float = math.dist((xmin, ymin), (xmax, ymax))
        self.xmin:int = xmin
        self.ymin:int = ymin
        self.xmax:int = xmax
        self.ymax:int = ymax
        self.x:int = (xmin+xmax) / 2.0
        self.y:int = (ymin+ymax) / 2.0
        self.id = ball_id
        self.previous_balls = target_temp
        #self.prev_x = LastValues(int(INIT_FPS/2))
        #self.prev_y = LastValues(int(INIT_FPS/2))
        self.prev_x = None
        self.prev_y = None

    def has_neighbour_ball(self, balls):
        max_diameter = max(b.diameter for b in balls) if balls else 0.0
        for ball in balls:
            distance = np.sqrt((self.x - ball.x)**2 + (self.y - ball.y)**2)
            if 2.5 * max_diameter > distance > 0.0:
                return True
        return False

    def get_diameter(self):
        return self.diameter

    def get_radius(self):
        return (abs(self.xmax-self.xmin)+abs(self.ymax-self.ymin))/4.0

    def is_moving(self, previous_ball):
        distance = np.sqrt((self.x - previous_ball.x) ** 2 + (self.y - previous_ball.y) ** 2)
        return distance > 0.1  # Threshold for detecting movement

