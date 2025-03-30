from scipy.interpolate import interp1d
from .piecewise import _piecewise


def _midpoint(x, y):
    prev_func = interp1d(x, y, kind="previous", bounds_error=False, fill_value=(1, 0))
    next_func = interp1d(x, y, kind="next", bounds_error=False, fill_value=(1, 0))

    def midpoint_func(t):
        return (prev_func(t) + next_func(t)) / 2

    return midpoint_func


def piecewise_interp(q, q_frac, interp):
    if interp == "midpoint":
        return _midpoint(q, 1 - q_frac)
    elif interp == "linear":
        return interp1d(
            q, 1 - q_frac, kind="linear", bounds_error=False, fill_value=(1, 0)
        )
    elif interp == "pchip":
        return _piecewise(q, 1 - q_frac, interp="pchip")
    else:
        raise ValueError(f"Unknown interp: {interp}")
