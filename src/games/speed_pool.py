import time
import math
from conf.constants import beep

class SpeedPool:

    def __init__(self):
        self.rack_start_time = time.perf_counter()
        self.rack_time = 0.0
        self.is_game_on = False
        self.rack_has_been_ready = False
        self.rack_start_time = time.perf_counter()

    def run_frame(self, vision_inference):
        unracked_ball_count = vision_inference.get_unracked_ball_count()

        if vision_inference.is_rack_ready(unracked_ball_count):
            self.rack_has_been_ready = True

        if self.rack_has_been_ready and unracked_ball_count > 2 and not self.is_game_on:
            self.is_game_on = True
            self.rack_start_time = time.perf_counter()
            beep()
        else:
            if self.is_game_on:
                self.rack_time = time.perf_counter() - self.rack_start_time
                if vision_inference.ball_count_average <= 1.05:
                    self.is_game_on = False
                    self.rack_has_been_ready = False
                    beep()
                if vision_inference.is_rack_ready(unracked_ball_count):
                    self.is_game_on = False
                    self.rack_time = 0.0

        game_off_text = ''
        if self.rack_has_been_ready:
            if (vision_inference.is_rack_ready(unracked_ball_count)
                    and vision_inference.ball_count_average > vision_inference.min_ball_count_in_rack and not self.is_game_on):
                game_off_text = f'RACK OF {math.ceil(vision_inference.ball_count_average - 1.0)} BALLS IS READY! '

        game_time_text = str(round(self.rack_time, 2)) + " seconds"
        return game_off_text+game_time_text


