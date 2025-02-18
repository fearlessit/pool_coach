import os

# Define configuration parameters
DETECTION_THRESHOLD = 0.10
MIN_BALL_COUNT_IN_RACK = 7


# Define hard coded constants
FPS = 30
BOUNDING_BOX_COLORS = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106), (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]
WORKING_DIR = os.getcwd()

# MODEL_DIR = f"{WORKING_DIR}/models/only_balls.v3i.yolov11/runs/detect/train/weights/best.pt"
MODEL_DIR = f"{WORKING_DIR}/model/best_weights.pt"

BRIGHT_COLOR = (256, 128, 256)


def beep():
    os.system(f"aplay {WORKING_DIR}/assets/beep.wav")

