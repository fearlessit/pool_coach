import cv2
import math
from time import perf_counter
from typing import List
from scipy import constants
from billiard.ball import BilliardBall
from conf.constants import MOVEMENT_THRESHOLD, BALL_DIAMETER, INIT_FPS
from utils.last_values import LastValues
from conf import constants
from utils.stable_change import StableChange
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

deepsort = None
def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
            nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
            use_cuda=True)

init_tracker()

def balls_distance(ball1, ball2):
    return math.dist((ball1.x, ball1.y), (ball2.x, ball2.y))

class BilliardInference:

    def __init__(self, yolo_model, detection_threshold=constants.DETECTION_THRESHOLD,
                 min_ball_count_in_rack=constants.MIN_BALL_COUNT_IN_RACK):
        self.is_moving = False
        self.yolo_model = yolo_model
        self.detections = []
        self.min_ball_count_in_rack = min_ball_count_in_rack
        self.detection_threshold = detection_threshold
        self.recent_balls_momentum = LastValues(window_size=int(constants.INIT_FPS * 1.0))
        self.ball_movement_stable_change = StableChange(frame_count_to_wait=self.recent_balls_momentum.window_size, last_stable_value=False)
        self.last_ball_counts = LastValues(window_size=int(constants.INIT_FPS * 1.25))
        self.ball_count_average = 0.0
        self.balls:List[BilliardBall] = []
        self.prev_ball_means = []
        self.balls_momentum = 0.0
        self.next_ball_id = 0
        self.tracker_time = {}

    def detect_balls_in_frame(self, frame):
        results = self.yolo_model.predict(source=frame, iou=constants.IOU, verbose=constants.VERBOSE, classes=[0])
        for r in results:
            if len(r.boxes) > 0:
                bbox_xywh = r.boxes.xywh.cpu().numpy().astype('float64')
                confs = r.boxes.conf.cpu().numpy().astype('float64')
                clss = r.boxes.cls.cpu().numpy().astype('float64')
                outputs = deepsort.update(bbox_xywh, confs, clss, frame)
                if len(outputs) > 0:
                    for output in outputs:
                        x1, y1, x2, y2, track_id = map(int, output[:5])

                        # time
                        if track_id not in self.tracker_time:
                            self.tracker_time[track_id] = [perf_counter(), perf_counter()]
                        self.tracker_time[track_id][1] = perf_counter()

                        time_in_store = self.tracker_time[track_id][1] - self.tracker_time[track_id][0]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                        cv2.putText(frame, f"ID:{track_id} Time:{time_in_store:.2f}s", (x1-20, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 4)

        #cv2.imshow('frame', frame)
        self.detections = results[0].boxes
        new_balls = []
        for i in range(len(self.detections)):
            if self.detections[i].conf >= self.detection_threshold:
                xyxy_tensor = self.detections[i].xyxy.cpu()  # Detections in Tensor format in CPU memory
                xyxy = xyxy_tensor.numpy().squeeze()  # Convert tensors to Numpy array
                xmin, ymin, xmax, ymax = xyxy.astype(float)  # Extract individual coordinates and convert to int
                detection_confidence = self.detections[i].conf.item()  # Get detection confidence
                y = (ymin+ymax)/2.0
                x = (xmin+xmax)/2.0
                new_balls.append((x,y,xmin,ymin,xmax,ymax,detection_confidence))

        for old_ball in self.balls:
            # Find the closest new ball to the old ball
            closest_new_ball = None
            closest_distance = 1000000
            for new_ball in new_balls:
                distance = math.dist((old_ball.x +old_ball.velocity[0], old_ball.y +old_ball.velocity[1]), (new_ball[0], new_ball[1]))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_new_ball = new_ball

            if closest_new_ball:
                if closest_distance < 5 * BALL_DIAMETER:
                    new_balls.remove(closest_new_ball)
                    x, y, xmin, ymin, xmax, ymax, detection_confidence = closest_new_ball
                    old_ball.prev_x = old_ball.x
                    old_ball.prev_y = old_ball.y
                    old_ball.x = x
                    old_ball.y = y
                    old_ball.velocity = (old_ball.x - old_ball.prev_x, old_ball.y - old_ball.prev_y)
                    old_ball.xmin = xmin
                    old_ball.ymin = ymin
                    old_ball.xmax = xmax
                    old_ball.ymax = ymax
                    old_ball.detection_confidence = detection_confidence
                else:
                    self.balls.remove(old_ball)
            else:
                self.balls.remove(old_ball)

        for new_ball in new_balls:
            self.next_ball_id += 1
            x, y, xmin, ymin, xmax, ymax, detection_confidence = new_ball
            self.balls.append(BilliardBall(self.next_ball_id, xmin, ymin, xmax, ymax, detection_confidence))
        self.ball_count_average = self.last_ball_counts.update(len(self.balls))

    def is_any_ball_moving(self):
        balls_momentum = sum(ball.x+ball.y for ball in self.balls)
        momentum_avg = self.recent_balls_momentum.update(balls_momentum)
        momentum_avg_diff = abs(momentum_avg -balls_momentum) / (len(self.balls)+1)
        self.is_moving = momentum_avg_diff > MOVEMENT_THRESHOLD
        return self.ball_movement_stable_change.stable_value(self.is_moving)

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
