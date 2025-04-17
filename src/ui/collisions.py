import cv2
import math
from random import random

class Collisions:
    def __init__(self):
        self.collisionMarkingPoints = []

    def is_any_colliding_balls(self, billiard_inference): ## TODO: optimointi
        balls = billiard_inference.balls
        is_moving = billiard_inference.is_moving
        for i, ball1 in enumerate(balls):
            for j, ball2 in enumerate(balls):
                if i != j:
                    distance_between_balls = math.dist((ball1.x, ball1.y), (ball2.x, ball2.y))
                    # TODO:
                    c = 2.00 if is_moving else 1.0 # this is just a magic number,
                    # collision should be detected also including the velocity (speed and direction) of the balls
                    c = 1.0
                    collision_distance = (ball1.get_radius() + ball2.get_radius()) * c
                    if distance_between_balls <= collision_distance:
                        self.collisionMarkingPoints.append(( (ball1.x+ball2.x)/2, (ball1.y+ball2.y)/2,  30))
        return len(self.collisionMarkingPoints) > 0

    def draw_collision_markings(self, frame):
        for i in range(len(self.collisionMarkingPoints)):
            col = self.collisionMarkingPoints.pop(0)
            k = col[2]
            r = 255 - k * 2
            g = 30
            b = k
            x = int(col[0])
            y = int(col[1])
            if random() < 0.5:
                cv2.circle(frame, (x, y), k, (r, g, b), k)
            else:
                cv2.circle(frame, (x, y), k, (b, r, g), k)

            if col[2] > 0 and len(self.collisionMarkingPoints) < 100:
                self.collisionMarkingPoints.append((col[0], col[1], col[2] - 1))


