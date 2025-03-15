import pandas as pd

class LastValues:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.values = []

    def update(self, v):
        self.values.append(v)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return self.get_average()

    def get_average(self):
        if len(self.values) == 0:
            return 0.0
        return sum(self.values) / len(self.values)

    def get_sum(self):
        return sum(self.values)

    def get_diff_sum(self):
        if len(self.values) < 2:
            return 0.0
        return sum(abs(self.values[i] - self.values[i - 1]) for i in range(1, len(self.values)))

    def get_diff_avg(self):
        return self.get_diff_sum() / len(self.values) if len(self.values) > 1 else 0.0
