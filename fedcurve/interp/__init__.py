from scipy.interpolate import interp1d
from .pchip import phchip_interp


def midpoint_interp(x, y):
    prev_func = interp1d(x, y, kind="previous", bounds_error=False, fill_value=(1, 0))
    next_func = interp1d(x, y, kind="next", bounds_error=False, fill_value=(1, 0))

    def midpoint_func(t):
        return (prev_func(t) + next_func(t)) / 2

    return midpoint_func


def interp_function(q, q_frac, interp):
    if interp == "midpoint":
        return midpoint_interp(q, 1 - q_frac)
    elif interp == "linear":
        return interp1d(
            q, 1 - q_frac, kind="linear", bounds_error=False, fill_value=(1, 0)
        )
    elif interp == "pchip":
        return phchip_interp(q, 1 - q_frac)
    else:
        raise ValueError(f"Unknown interp: {interp}")
