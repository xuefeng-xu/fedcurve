import numpy as np
from scipy.interpolate import PchipInterpolator


class _piecewise:
    def __init__(self, x, y, interp, y_order):
        self.y_order = y_order

        if interp == "pchip":
            self.base_interp = PchipInterpolator

        self.xmin, self.xmax = min(x), max(x)
        self._segment_fit(x, y)

    def _segment_fit(self, x, y):
        # Skip duplicate x values
        self.edge = []
        self.interp = []

        i = 0
        while i < len(x) - 1:
            # Search for the start of the segment
            while i < len(x) - 1 and x[i] == x[i + 1]:
                i += 1
            start = i

            if start == len(x) - 1:
                break

            # Search for the end of the segment
            i += 1
            while i < len(x) - 1 and x[i] != x[i + 1]:
                i += 1
            end = i + 1

            self.edge.append(x[end - 1])
            self.interp.append(self.base_interp(x[start:end], y[start:end]))
            i = start = end

    def _evaluate(self, x):
        if len(self.interp) == 1:
            y = self.interp[0](x)

        else:
            y = []
            for xi in x:
                j = np.searchsorted(self.edge, xi, side="right")
                y.append(self.interp[j](xi))

        return np.clip(y, 0.0, 1.0)

    def __call__(self, x):
        x = np.asarray(x)
        idx_left = x < self.xmin
        idx_right = x >= self.xmax

        y = np.zeros_like(x)
        if self.y_order == "desc":
            y[idx_left] = 1.0
        else:  # "aes"
            y[idx_right] = 1.0

        idx = np.logical_and(~idx_left, ~idx_right)
        if np.any(idx) > 0:
            y[idx] = self._evaluate(x[idx])

        return y
