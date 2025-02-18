import math
from typing import List
from billiard.billiard_ball import BilliardBall
from billiard.last_average import LastAverage
from conf import constants


class BilliardBallInference:

    def __init__(self, yolo_model, detection_threshold=constants.DETECTION_THRESHOLD, min_ball_count_in_rack=constants.MIN_BALL_COUNT_IN_RACK, diameter_average_window_size=constants.FPS * 2, ball_count_average_window_size=constants.FPS * 1.25):
        self.yolo_model = yolo_model
        self.min_ball_count_in_rack = min_ball_count_in_rack
        self.detection_threshold = detection_threshold

        self.rolling_max_ball_diameter = 0.0
        self.ball_count_average = 0
        self.balls: List[BilliardBall] = []
        self.detections = []
        self.rolling_diameter_average = LastAverage(window_size=int(diameter_average_window_size))
        self.rolling_ball_count_average = LastAverage(window_size=int(ball_count_average_window_size))

    def detect_balls_in_frame(self, frame):
        results = self.yolo_model(frame, verbose=False) # , conf=self.detection_threshold
        self.detections = results[0].boxes
        self.ball_count_average = self.rolling_ball_count_average.update(len(self.balls))
        self.balls.clear()
        for i in range(len(self.detections)):
            if self.detections[i].conf >= self.detection_threshold:
                xyxy_tensor = self.detections[i].xyxy.cpu()  # Detections in Tensor format in CPU memory
                xyxy = xyxy_tensor.numpy().squeeze()  # Convert tensors to Numpy array
                xmin, ymin, xmax, ymax = xyxy.astype(float)  # Extract individual coordinates and convert to int
                detection_confidence = self.detections[i].conf.item()  # Get detection confidence
                self.balls.append(BilliardBall(xmin, ymin, xmax, ymax, detection_confidence))

        frame_max_ball_diameter = max([ball.diameter for ball in self.balls]) if self.balls else 0.0
        self.rolling_max_ball_diameter = self.rolling_diameter_average.update(frame_max_ball_diameter)
        return len(self.balls)

    def get_unracked_ball_count(self):
        unracked_balls = 0
        for ball in self.balls:
            distant_ball_count = 0
            for ball2 in self.balls:
                distance = math.dist((ball.x, ball.y), (ball2.x, ball2.y))
                if distance > 5*self.rolling_max_ball_diameter:
                    distant_ball_count += 1
                if distant_ball_count >= 2:
                    unracked_balls += 1
                    break
        return unracked_balls

    def is_rack_ready(self, unracked_ball_count):
        return unracked_ball_count <= 1.5 and self.ball_count_average >= self.min_ball_count_in_rack

    def get_rolling_ball_count_average(self):
        return self.rolling_ball_count_average.get_average()

