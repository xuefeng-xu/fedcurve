import numpy as np


# continuous
def Laplace(eps, sens, size):
    # Laplace Mechanism
    lam = sens / eps
    return np.random.laplace(loc=0, scale=lam, size=size)


# continuous
def DistrbutedLaplace(eps, sens, n_clients, size):
    # Difference of two Gamma random variables
    # Reference: equation (3) and (4) in https://doi.org/10.1109/TDSC.2015.2484326
    k = 1 / n_clients
    theta = sens / eps
    rvs = np.random.gamma(shape=k, scale=theta, size=(2, size))
    return rvs[0] - rvs[1]


# discrete
def Geometric(eps, sens, size):
    # Geometric Mechanism: two-sided geometric distribution
    # Reference: IBM Diffprivlib
    # https://diffprivlib.readthedocs.io/en/0.6.0/_modules/diffprivlib/mechanisms/geometric.html#Geometric.randomise
    scale = -eps / sens
    unif_rv = np.random.rand(size) - 0.5
    unif_rv *= 1 + np.exp(scale)
    sgn = np.sign(unif_rv).astype(int)
    return sgn * np.floor(np.log(sgn * unif_rv) / scale).astype(int)


# discrete
def DistrbutedGeometric(eps, sens, n_clients, size):
    # Difference of two Polya (negative binomial) random variables
    # Reference: Theorem 5.1 in https://doi.org/10.1109/TDSC.2015.2484326
    n = 1 / n_clients
    p = 1 - np.exp(-eps / sens)
    rvs = np.random.negative_binomial(n, p, size=(2, size))
    return rvs[0] - rvs[1]


def LocalDP_noise(eps, sens, size, noise_type="continuous"):
    if noise_type == "continuous":
        return Laplace(eps, sens, size)
    else:  # discrete
        return Geometric(eps, sens, size)


def DistrbutedDP_noise(eps, sens, n_clients, size, noise_type="continuous"):
    if noise_type == "continuous":
        return DistrbutedLaplace(eps, sens, n_clients, size)
    else:  # discrete
        return DistrbutedGeometric(eps, sens, n_clients, size)
