from ultralytics import YOLO
from billiard.inference import BilliardInference
from conf import constants
from games.speed_pool import SpeedPool
from conf.constants import *
from ui.billiard_ui import BilliardUI
from ui.collisions import Collisions
import os

# Set OpenCV FFMPEG read attempts before creating video capture
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '100000'


video_capture = cv2.VideoCapture(constants.input_video_capture_parameter)
model = YOLO(MODEL_DIR, task='detect', verbose=False)
billiard_inference = BilliardInference(model)
ui = BilliardUI(billiard_inference, video_capture)
speed_pool = SpeedPool()
frame = video_capture.read()


collisions = Collisions()
avg_frame_rate=0.0
game_info_text = ''
#with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
is_flag = False
flag_collided = 0
while True:
    frame = ui.read_frame()
    billiard_inference.detect_balls_in_frame(frame)

    game_info_text = speed_pool.run_frame(billiard_inference)

    is_ball_moving = billiard_inference.is_any_ball_moving()

    ui.highlight_detected_balls()
    if collisions.predict_collisions(billiard_inference):
        flag_collided = 10

    if not billiard_inference.is_rack_ready():
        collisions.draw_collision_markings(frame)

    flag_collided -= 1
    ui.draw_info_texts(game_info_text, is_ball_moving, flag_collided > 0)

    ui.show_frame()
    ui.wait_and_react_keyboard_input()
