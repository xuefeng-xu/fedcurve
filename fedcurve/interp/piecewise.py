import numpy as np
from scipy.interpolate import PchipInterpolator


class _piecewise:
    # Assumptions:
    # x is in the range [0, 1] and monotonic increasing
    # y is in the range [0, 1] and monotonic decreasing

    def __init__(self, x, y, interp):
        if interp == "pchip":
            self.base_interp = PchipInterpolator
        else:
            raise ValueError(f"Unknown interp: {interp}")

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
        y = np.zeros_like(x)

        idx_1 = x < self.xmin
        idx_0 = x >= self.xmax
        y[idx_1] = 1.0
        # y[idx_0] = 0.0

        idx = np.logical_and(~idx_1, ~idx_0)
        if np.any(idx) > 0:
            y[idx] = self._evaluate(x[idx])

        return y
