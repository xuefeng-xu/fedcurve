import numpy as np
from scipy.stats import dlaplace
from sklearn.utils import check_random_state


# continuous
def Laplace(eps, sens, size, rng=None):
    # Laplace Mechanism
    lam = sens / eps
    rng = check_random_state(rng)
    return rng.laplace(loc=0, scale=lam, size=size)


# continuous
def DistributedLaplace(eps, sens, n_clients, size, rng=None):
    # Difference of two Gamma random variables
    # Reference: equation (3) and (4) in https://doi.org/10.1109/TDSC.2015.2484326
    k = 1 / n_clients
    theta = sens / eps
    rng = check_random_state(rng)
    rvs = rng.gamma(shape=k, scale=theta, size=(2, size))
    return rvs[0] - rvs[1]


# discrete
def Geometric(eps, sens, size, rng=None):
    # Geometric Mechanism: two-sided Geometric distribution
    # Discrete Laplace is equivalent to two-sided Geometric
    a = eps / sens
    return dlaplace.rvs(a=a, loc=0, size=size, random_state=rng)


# discrete
def DistributedGeometric(eps, sens, n_clients, size, rng=None):
    # Difference of two Polya (negative binomial) random variables
    # Reference: Theorem 5.1 in https://doi.org/10.1109/TDSC.2015.2484326
    n = 1 / n_clients
    p = 1 - np.exp(-eps / sens)
    rng = check_random_state(rng)
    rvs = rng.negative_binomial(n, p, size=(2, size))
    return rvs[0] - rvs[1]


def LocalDP_noise(eps, sens, size, noise_type="continuous", rng=None):
    if noise_type == "continuous":
        return Laplace(eps, sens, size, rng=rng)
    elif noise_type == "discrete":
        return Geometric(eps, sens, size, rng=rng)
    else:
        raise ValueError("noise_type must be 'continuous' or 'discrete'.")


def DistributedDP_noise(eps, sens, n_clients, size, noise_type="continuous", rng=None):
    if noise_type == "continuous":
        return DistributedLaplace(eps, sens, n_clients, size, rng=rng)
    elif noise_type == "discrete":
        return DistributedGeometric(eps, sens, n_clients, size, rng=rng)
    else:
        raise ValueError("noise_type must be 'continuous' or 'discrete'.")
