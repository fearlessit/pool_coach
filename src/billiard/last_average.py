import pandas as pd

class LastAverage:
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
        df = pd.Series(self.values)
        return df.rolling(window=self.window_size, min_periods=1).mean().iloc[-1]
