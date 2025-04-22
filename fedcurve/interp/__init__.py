import numpy as np
from scipy.interpolate import interp1d
from .piecewise import _piecewise


def _check(x, y, y_order):
    # Assumptions:
    # x and y are both in the range [0, 1]
    # x is ascending (0 to 1)
    # y is ascending (0 to 1) or descending (1 to 0)

    if len(x) != len(y):
        raise ValueError(f"Length of x and y must be equal")

    if max(x) > 1.0 or min(x) < 0.0:
        raise ValueError(f"x must be in the range [0, 1]")
    if max(y) > 1.0 or min(y) < 0.0:
        raise ValueError(f"y must be in the range [0, 1]")

    if not np.all(np.diff(x) >= 0):
        raise ValueError(f"x must be ascending")

    if y_order not in ["asc", "desc"]:
        raise ValueError(f"Unknown y_order: {y_order}")

    if y_order == "asc":
        if not np.all(np.diff(y) >= 0):
            raise ValueError(f"y must be ascending")
    else:  # "desc"
        if not np.all(np.diff(y) <= 0):
            raise ValueError(f"y must be descending")


def _midpoint(x, y, y_order):
    if y_order == "desc":
        fill_value = (1, 0)
    else:  # "asc"
        fill_value = (0, 1)

    prev_func = interp1d(
        x, y, kind="previous", bounds_error=False, fill_value=fill_value
    )
    next_func = interp1d(x, y, kind="next", bounds_error=False, fill_value=fill_value)

    def midpoint_func(t):
        return (prev_func(t) + next_func(t)) / 2

    return midpoint_func


def _linear(x, y, y_order):
    if y_order == "desc":
        fill_value = (1, 0)
    else:  # "asc"
        fill_value = (0, 1)

    return interp1d(x, y, kind="linear", bounds_error=False, fill_value=fill_value)


def piecewise_interp(x, y, interp, y_order="desc"):
    _check(x, y, y_order)

    if interp == "midpoint":
        return _midpoint(x, y, y_order)
    elif interp == "linear":
        return _linear(x, y, y_order)
    elif interp == "pchip":
        return _piecewise(x, y, interp="pchip", y_order=y_order)
    else:
        raise ValueError(f"Unknown interp: {interp}")
