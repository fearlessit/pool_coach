import cv2
import sys
import time
from billiard.last_average import LastAverage
from conf.constants import *

class BilliardUI:

    def __init__(self, vision_inference, video_capture):
        self.vision_inference = vision_inference
        self.video_capture = video_capture
        self.frame_rate_buffer = []
        self.fps_avg = LastAverage(window_size=200)
        self.t_start = 0
        self.fps = video_capture.get(cv2.CAP_PROP_FPS)
        self.frame_duration = 1.0 / self.fps
        self.frame_start = 0.0
        self.frame_sleep_time = 1
        self.exit_text = ''
        self.frame = None

    def read_frame(self):
        self.t_start = time.perf_counter()
        self.frame_start = time.time()
        ret, frame = self.video_capture.read()
        if not ret:
            print('No more frames to read, exit now.')
            self.force_exit()
        self.frame = frame
        return frame


    def show_frame(self):
        cv2.imshow('Billiard Coach', self.frame)  # Display image

        # Calculate FPS for this frame and update the FPS average
        t_stop = time.perf_counter()
        frame_rate_calc = float(1 / (t_stop - self.t_start))
        self.fps_avg.update(frame_rate_calc)

        # Calculate sleep time for the frame to match the desired FPS
        elapsed_time = time.time() - self.frame_start
        self.frame_sleep_time = int(max(1, (self.frame_duration - elapsed_time) * 1000.0))

    def wait_and_react_keyboard_input(self):
        key = cv2.waitKey(self.frame_sleep_time)

        if key == ord('q') or key == ord('Q'):
            self.force_exit()

    def highlight_detected_balls(self):
        balls = self.vision_inference.balls
        for ball in balls:
            cv2.rectangle(self.frame, (int(ball.xmin), int(ball.ymin)), (int(ball.xmax), int(ball.ymax)), BRIGHT_COLOR, 5)
            detection_confidence_text = f'{int(ball.detection_confidence * 100)}%'
            label_size, base_line = cv2.getTextSize(detection_confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ball.ymin, label_size[1] + 10)
            cv2.rectangle(self.frame, (int(ball.xmin), int(label_ymin - label_size[1] - 10)), (int(ball.xmin + label_size[0]), int(label_ymin + base_line - 10)), BRIGHT_COLOR, cv2.FILLED)
            cv2.putText(self.frame, detection_confidence_text, (int(ball.xmin), int(label_ymin - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


    def draw_info_texts(self, game_text):
        avg_ball_count = self.vision_inference.rolling_ball_count_average.get_average()
        general_info_text = f'Balls detected={avg_ball_count:0.1f}, FPS={FPS} (MAX FPS {int(self.fps_avg.get_average())})'
        general_text_color = (0, 255, 0) if self.fps_avg.get_average() > FPS else (0, 0, 255)
        cv2.putText(self.frame, general_info_text, (40,25), cv2.FONT_HERSHEY_SIMPLEX, .9, general_text_color,2)
        cv2.putText(self.frame, game_text, (40, 55), cv2.FONT_HERSHEY_SIMPLEX, .9, BRIGHT_COLOR, 2)
        self.exit_text = game_text

    def force_exit(self):
        print(f'Game info text: "{self.exit_text}"')
        self.video_capture.release()
        cv2.destroyAllWindows()
        sys.exit(0)
