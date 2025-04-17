import os

# Input video capture parameter
input_video_capture_parameter = 2 # Realtime webcam capture

input_video_capture_parameter = "./assets/speed_pool.mp4"
#input_video_capture_parameter = "./assets/videos/sampo_ida_nine.mp4")
#input_video_capture_parameter = "./assets/videos/almost2.mp4")
#input_video_capture_parameter = "./assets/videos/seiskaa.mkv")
#input_video_capture_parameter = "./assets/videos/ykxkax.mp4"
#input_video_capture_parameter = "./assets/videos/lineup.mkv")
#input_video_capture_parameter = "./assets/videos/liiketunnistus.mkv")
#input_video_capture_parameter = "./assets/videos/speed_pool.mp4"
#input_video_capture_parameter = "./assets/videos/eight.mp4")
#input_video_capture_parameter = "./assets/videos/seven.mkv"


# Define configuration parameters
DETECTION_THRESHOLD = 0.5
IOU = 0.1
MOVEMENT_THRESHOLD = 1.0
MIN_BALL_COUNT_IN_RACK = 9
VERBOSE = True
BALL_DIAMETER = 50
# Define hard coded constants

INIT_FPS = 30

MODEL_PATH = "model/best_weights.pt"
#MODEL_PATH = "yolo11n.pt"

BRIGHT_COLOR = (256, 128, 256)
RED_COLOR = (0, 0, 256)


def beep():
    os.system("assets/beep.wav")
