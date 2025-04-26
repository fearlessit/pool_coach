import math
from typing import List

from scipy import constants
from billiard.ball import BilliardBall
from conf.constants import BALL_DIAMETER, INIT_FPS, BALL_MOVEMENT_THRESHOLD
from utils.last_values import LastValues
from conf import constants
from utils.stable_change import StableChange



def balls_distance(ball1, ball2):
    return math.dist((ball1.x, ball1.y), (ball2.x, ball2.y))

class BilliardInference:

    def __init__(self, yolo_model, detection_threshold=constants.DETECTION_THRESHOLD,
                 min_ball_count_in_rack=constants.MIN_BALL_COUNT_IN_RACK):
        self.yolo_model = yolo_model
        self.detections = []
        self.min_ball_count_in_rack = min_ball_count_in_rack
        self.detection_threshold = detection_threshold
        self.recent_balls_momentum = LastValues(window_size=int(constants.INIT_FPS * 1.0))
        self.balls_moving = StableChange(frame_count_to_wait=self.recent_balls_momentum.window_size, last_stable_value=False)
        self.last_ball_counts = LastValues(window_size=int(constants.INIT_FPS * 1.25))
        self.ball_count_average = 0.0
        self.balls:List[BilliardBall] = []
        self.prev_ball_means = []
        self.balls_momentum = 0.0
        self.next_ball_id = 0


    def detect_balls_in_frame(self, frame):
        #results = self.yolo_model.predict(source=frame, iou=constants.IOU, verbose=constants.VERBOSE)

        # TRACKER
        results = self.yolo_model.track(source=frame, persist=True, tracker="./bytetrack.yaml", conf=constants.DETECTION_THRESHOLD, iou=constants.IOU, verbose=constants.VERBOSE)

        self.detections = results[0].boxes
        previous_frame_balls = self.balls.copy()
        self.balls.clear()
        for i in range(len(self.detections)):
            if self.detections[i].conf >= self.detection_threshold:

                track_id = self.detections[i].id
                if track_id is not None and len(track_id) > 0:
                    track_id = int(track_id[0])
                else:
                    track_id = 0

                xyxy_tensor = self.detections[i].xyxy.cpu()  # Detections in Tensor format in CPU memory
                xyxy = xyxy_tensor.numpy().squeeze()  # Convert tensors to Numpy array
                xmin, ymin, xmax, ymax = xyxy.astype(float)  # Extract individual coordinates and convert to int
                detection_confidence = self.detections[i].conf.item()  # Get detection confidence
                billiard_ball = BilliardBall(track_id, xmin, ymin, xmax, ymax, detection_confidence)

                for previous_frame_ball in previous_frame_balls:
                    if previous_frame_ball.id == track_id:
                        billiard_ball.velocity_x = abs(billiard_ball.x -previous_frame_ball.x) / INIT_FPS
                        billiard_ball.velocity_y = abs(billiard_ball.y -previous_frame_ball.y) / INIT_FPS
                        billiard_ball.speed = math.sqrt(billiard_ball.velocity_x**2 + billiard_ball.velocity_y**2)

                self.balls.append(billiard_ball)

        self.ball_count_average = self.last_ball_counts.update(len(self.balls))


    def is_balls_momentum_moving(self):
        balls_momentum = sum(ball.x+ball.y for ball in self.balls)
        momentum_avg = self.recent_balls_momentum.update(balls_momentum)
        momentum_avg_diff = abs(momentum_avg -balls_momentum) / (len(self.balls)+1)
        is_moving = momentum_avg_diff > BALL_MOVEMENT_THRESHOLD
        return self.balls_moving.stable_value(is_moving)


    def get_unracked_ball_count(self):
        unracked_balls = 0
        for ball1 in self.balls:
            distant_ball_count = 0
            for ball2 in self.balls:
                distance = balls_distance(ball1, ball2)
                if distance > 7.5 * BALL_DIAMETER:# self.rolling_average_ball_diameter:
                    distant_ball_count += 1
                if distant_ball_count >= 2:
                    unracked_balls += 1
                    break
        return unracked_balls

    def is_rack_ready(self):
        unracked_ball_count = self.get_unracked_ball_count()
        return unracked_ball_count <= 1.5 and self.ball_count_average >= self.min_ball_count_in_rack

    def get_recent_ball_count_average(self):
        return self.last_ball_counts.get_average()
