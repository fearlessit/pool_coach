from conf import constants


class StableChange:

    def __init__(self, last_stable_value = None, frame_count_to_wait = 1.0 * constants.INIT_FPS):
        self.frame_count = 0
        self.last_value = None
        self.frame_count_to_wait:int = int(frame_count_to_wait)
        self.last_stable_value = last_stable_value

    def is_stable_change(self, value):
        if self.last_value != value:
            self.last_value = value
            self.frame_count = 0
            return False

        self.frame_count += 1
        stable_change = self.frame_count % self.frame_count_to_wait == 0
        if stable_change:
            self.last_stable_value = value
        return None

    def stable_value(self, value):
        self.is_stable_change(value)
        return self.last_stable_value

    def reset(self):
        self.last_value = None
        self.frame_count = 0
