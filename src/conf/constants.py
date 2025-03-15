import os

import cv2

# Input video capture parameter

input_video_capture_parameter = 2 # Realtime webcam capture

#input_video_capture_parameter = os.path.expanduser("~/videos/almost2.mp4")
#input_video_capture_parameter = os.path.expanduser("~/videos/seiskaa.mkv")
input_video_capture_parameter = "./assets/ykxkax.mp4"
#input_video_capture_parameter = os.path.expanduser("~/videos/lineup.mkv")
#input_video_capture_parameter = "./assets/seven.mkv"
#input_video_capture_parameter = os.path.expanduser("~/videos/lineup.mkv")
#input_video_capture_parameter = os.path.expanduser("~/videos/liiketunnistus.mkv")
#input_video_capture_parameter = "./assets/speed_pool.mp4"
#input_video_capture_parameter = os.path.expanduser("~/videos/eight.mp4")

# Define configuration parameters
DETECTION_THRESHOLD = 0.5
IOU = 0.1
MOVEMENT_THRESHOLD = 1.0
MIN_BALL_COUNT_IN_RACK = 9
VERBOSE = False
BALL_DIAMETER = 50
# Define hard coded constants

INIT_FPS = 30
BOUNDING_BOX_COLORS = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106), (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

MODEL_DIR = "model/best_weights.pt"

BRIGHT_COLOR = (256, 128, 256)
RED_COLOR = (0, 0, 256)


def beep():
    os.system("assets/beep.wav")

