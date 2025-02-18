import cv2
from ultralytics import YOLO
from games.speed_pool import SpeedPool
from ui.billiard_ui import BilliardUI
from billiard.ball_inference import BilliardBallInference
from conf.constants import *
import os

# Set OpenCV FFMPEG read attempts before creating video capture
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '100000'

model = YOLO(MODEL_DIR, task='detect')
vision_inference = BilliardBallInference(model)

# Initialize video capturing
# video_capture = cv2.VideoCapture(0) # Realtime webcam capture
video_capture = cv2.VideoCapture(f"/{WORKING_DIR}/assets/speed_ball.mp4") # Recorded offline capture

ui = BilliardUI(vision_inference, video_capture)
speed_pool = SpeedPool()

avg_frame_rate=0.0
#with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
while True:
    frame = ui.read_frame()
    frame_ball_count = vision_inference.detect_balls_in_frame(frame)
    ui.highlight_detected_balls()

    game_info_text = speed_pool.run_frame(vision_inference)

    ui.draw_info_texts(game_info_text)
    ui.show_frame()
    ui.wait_and_react_keyboard_input()

