import numpy as np
from scipy.interpolate import PchipInterpolator


def eliminate_duplicates(x, y):
    x_unique, inv_idx = np.unique(x, return_inverse=True)

    y_avg = np.zeros_like(x_unique)
    for i in range(len(x_unique)):
        # Compute averaged y values for each unique x
        y_avg[i] = np.mean(y[inv_idx == i])
        # y_avg[i] = np.max(y[inv_idx == i])

    return x_unique, y_avg


def phchip_interp(x, y):
    # x cannot include duplicate values for pchip
    x, y = eliminate_duplicates(x, y)

    x_min, x_max = min(x), max(x)
    pchip = PchipInterpolator(x, y)

    def phchip_func(t):
        t = np.asarray(t)
        z = np.zeros_like(t)

        idx_1 = t < x_min
        idx_0 = t > x_max
        z[idx_1] = 1.0
        # z[idx_0] = 0.0

        idx_ = np.logical_and(~idx_1, ~idx_0)
        z[idx_] = np.clip(pchip(t[idx_]), 0.0, 1.0)
        return z

    return phchip_func


def phchip_interp1(x, y):
    # Since x cannot include duplicate values
    # Use piecewise pchip interpolation
    pass
