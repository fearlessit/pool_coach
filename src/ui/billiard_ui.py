import sys
import time
import cv2

from utils.last_values import LastValues
from conf.constants import *


class BilliardUI:

    def __init__(self, vision_inference, video_capture):
        self.vision_inference = vision_inference
        self.video_capture = video_capture
        self.frame_rate_buffer = []
        self.fps_avg = LastValues(window_size=200)
        self.t_start = 0
        self.fps = int(video_capture.get(cv2.CAP_PROP_FPS) if video_capture.get(cv2.CAP_PROP_FPS) > 0 else INIT_FPS)
        self.frame_duration = 1.0 / self.fps
        self.frame_start = 0.0
        self.frame_sleep_time = 1
        self.exit_text = ""
        self.frame = None
        self.i = 0

    def read_frame(self):
        self.t_start = time.perf_counter()
        self.frame_start = time.time()
        ret, self.frame = self.video_capture.read()
        if not ret:
            print('No more frames to read, exit now.')
            self.force_exit()
        return self.frame

    def show_frame(self):

        cv2.imshow('Billiard Coach', self.frame)  # Display image

        # Calculate FPS for this frame and update the FPS average
        t_stop = time.perf_counter()
        frame_rate_calc = float(1 / (t_stop - self.t_start))
        self.fps_avg.update(frame_rate_calc)

        # Calculate sleep time for the frame to match the desired FPS
        elapsed_time = time.time() - self.frame_start
        self.frame_sleep_time = int(max(1.0, (self.frame_duration - elapsed_time) * 1000.0))

    def wait_and_react_keyboard_input(self):

        key = cv2.waitKey(self.frame_sleep_time)

        if key == ord('q') or key == ord('Q'):
            self.force_exit()

        if key == ord('+'):
            self.fps += 1
            self.frame_duration = 1.0 / self.fps
            print(f'FPS increased to {self.fps}')

        if key == ord('-'):
            self.fps -= 1
            self.frame_duration = 1.0 / self.fps
            print(f'FPS decreased to {self.fps}')

    def highlight_detected_balls(self):
        cv2.annotate = lambda frame, detections, names: labels
        balls = self.vision_inference.balls
        for ball in balls:
            cv2.rectangle(self.frame, (int(ball.xmin), int(ball.ymin)), (int(ball.xmax), int(ball.ymax)), (0,0,0), 2)
            #ball_header_text = f"id={ball.id}, r={(ball.get_radius()):0.0f}, conf={ball.detection_confidence:0.2f} ({ball.y:0.0f},{ball.x:0.0f})"
            ball_header_text = f"{ball.id}"
            label_size, base_line = cv2.getTextSize(ball_header_text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 1)
            label_ymin = max(ball.ymin, label_size[1] + 10)
            #cv2.rectangle(self.frame, (int(ball.xmax -5), int(label_ymin - label_size[1] - 17)), (int(ball.xmax + label_size[0]+10), int(label_ymin + base_line-10)), (150,130,130), cv2.FILLED)
            cv2.putText(self.frame, ball_header_text, (int(ball.x -(ball.get_radius()/2 if ball.id < 10 else ball.get_radius())), int(ball.y +ball.get_radius()/2)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, (0, 256, 256), 3)

    def draw_info_texts(self, game_text, draw_flag1=False, draw_flag2=False):
        avg_ball_count = self.vision_inference.last_ball_counts.get_average()
        general_info_text = f'Balls detected={avg_ball_count:0.1f}, FPS={self.fps} (MAX FPS {int(self.fps_avg.get_average())})'
        general_text_color = BRIGHT_COLOR if self.fps_avg.get_average() > self.fps else (0,0,256)
        cv2.putText(self.frame, general_info_text, (40,30), cv2.FONT_HERSHEY_SIMPLEX, .9, general_text_color,2)
        cv2.putText(self.frame, game_text, (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (50, 256, 50), 3)

        if draw_flag1:
            cv2.circle(self.frame, (100, 200), 10, RED_COLOR, 40)
        if draw_flag2:
            cv2.circle(self.frame, (100, 250), 10, (256,128,0), 20)
        self.exit_text = game_text

    def force_exit(self):
        print(f'Quitting the app\nGame info text: "{self.exit_text}"')
        self.video_capture.release()
        cv2.destroyAllWindows()
        sys.exit(0)
