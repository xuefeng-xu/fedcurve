import numpy as np
from cvxopt import solvers
from cvxopt import matrix, spmatrix


def post_process(x, z, order):
    # Clip the values to be in [0, 1]
    x = np.clip(x + z, 0, 1)
    # Ensure the values are in ascending or descending order
    if order == "asc":
        x = np.maximum.accumulate(x)
    else:  # "desc"
        x = np.minimum.accumulate(x)
    return x


def smooth_norm_1(x, order):
    # Reference: https://gitlab.inria.fr/abarczew/ab_technical/-/blob/b957b869ab8024e003cd0fc69a356309429dd7be/medical-statistics-and-privacy/dp-cum/code/2024/utils/smooth.py#L91-174
    n = len(x)
    # Linear solver: min sum |z[i]|
    # z[i] = v[i] - u[i]
    # v[i] >= 0, u[i] >= 0
    # min sum v[i] + u[i]
    c = matrix(1.0, (2 * n, 1), "d")

    # 2 * n constraints for v[i] and u[i]
    # n - 1 constraints for monotoneity
    # 2 constraints for min and max
    num_cons = 2 * n + (n - 1) + 2
    G = spmatrix([], [], [], (num_cons, 2 * n), "d")
    h = matrix(0.0, (num_cons, 1), "d")

    # 2 * n constraints for v[i] and u[i]
    # -v[i] <= 0, -u[i] <= 0
    for i in range(2 * n):
        G[i, i] = -1.0

    # n - 1 constraints for monotoneity
    for i in range(n - 1):
        if order == "asc":
            # v[i] - u[i] - v[i+1] + u[i+1] <= x[i+1] - x[i]
            G[i + 2 * n, i] = 1
            G[i + 2 * n, i + 1] = -1
            G[i + 2 * n, i + n] = -1
            G[i + 2 * n, i + 1 + n] = 1
            h[i + 2 * n] = x[i + 1] - x[i]
        else:  # "desc"
            # -v[i] + u[i] + v[i+1] - u[i+1] <= x[i] - x[i+1]
            G[i + 2 * n, i] = -1
            G[i + 2 * n, i + 1] = 1
            G[i + 2 * n, i + n] = 1
            G[i + 2 * n, i + 1 + n] = -1
            h[i + 2 * n] = x[i] - x[i + 1]

    # constraints for min and max
    if order == "asc":
        # min: -v[0] + u[0] <= x[0]
        G[-2, 0] = -1
        G[-2, n] = 1
        h[-2] = x[0]
        # max: v[n-1] - u[n-1] <= 1 - x[n-1]
        G[-1, n - 1] = 1
        G[-1, n - 1 + n] = -1
        h[-1] = 1 - x[-1]
    else:  # "desc"
        # max: v[0] - u[0] <= 1 - x[0]
        G[-2, 0] = 1
        G[-2, n] = -1
        h[-2] = 1 - x[0]
        # min: -v[n-1] + u[n-1] <= x[n-1]
        G[-1, n - 1] = -1
        G[-1, n - 1 + n] = 1
        h[-1] = x[-1]

    solvers.options["show_progress"] = False
    res = solvers.lp(c, G, h)
    v_u = np.array(res["x"]).reshape(2, -1)
    z = v_u[0] - v_u[1]
    return post_process(x, z, order)


def smooth_norm_2(x, order):
    # Reference: https://gitlab.inria.fr/abarczew/ab_technical/-/blob/b957b869ab8024e003cd0fc69a356309429dd7be/medical-statistics-and-privacy/dp-cum/code/2024/utils/smooth.py#L177-227
    n = len(x)
    # Quadratic solver: min sum z[i]^2
    P = spmatrix([], [], [], (n, n), "d")
    for i in range(n):
        P[i, i] = 1.0
    q = matrix(0.0, (n, 1), "d")

    # n - 1 constraints for monotoneity
    # 2 constraints for min and max
    num_cons = (n - 1) + 2
    G = spmatrix([], [], [], (num_cons, n), "d")
    h = matrix(0.0, (num_cons, 1), "d")

    # n - 1 constraints for monotoneity
    for i in range(n - 1):
        if order == "asc":
            # z[i] - z[i+1] <= x[i+1] - x[i]
            G[i, i] = 1
            G[i, i + 1] = -1
            h[i] = x[i + 1] - x[i]
        else:  # "desc"
            # -z[i] + z[i+1] <= x[i] - x[i+1]
            G[i, i] = -1
            G[i, i + 1] = 1
            h[i] = x[i] - x[i + 1]

    # constraints for min and max
    if order == "asc":
        # min: -z[0] <= x[0]
        G[-2, 0] = -1
        h[-2] = x[0]
        # max: z[n-1] <= 1 - x[n-1]
        G[-1, n - 1] = 1
        h[-1] = 1 - x[-1]
    else:  # "desc"
        # max: z[0] <= 1 - x[0]
        G[-2, 0] = 1
        h[-2] = 1 - x[0]
        # min: -z[n-1] <= 1 - x[n-1]
        G[-1, n - 1] = -1
        h[-1] = x[-1]

    solvers.options["show_progress"] = False
    res = solvers.qp(P, q, G, h)
    z = np.array(res["x"]).reshape(-1)
    return post_process(x, z, order)


def smooth_curve(x, norm=1, order="asc"):
    # Ascending or descending order
    # Bounded by [0, 1]
    if order not in ["asc", "desc"]:
        raise ValueError(f"Unknown order: {order}")

    if norm == 1:
        return smooth_norm_1(x, order)
    elif norm == 2:
        return smooth_norm_2(x, order)
    else:
        raise ValueError(f"Unsupported norm: {norm}")
