import numpy as np


def calc_il_to_rv(N, L):
    # Reference: https://gitlab.inria.fr/abarczew/ab_technical/-/blob/b957b869ab8024e003cd0fc69a356309429dd7be/medical-statistics-and-privacy/dp-cum/code/2024/utils/smooth.py#L11-28
    """
    il_to_rv[i,l] provides a linear index for the l-th random variable
    to be added to \ecdf(\tau_i) to obtain \ecdfdp(\tau_i)

    \ecdfdp(\tau_i) = \ecdf(\tau_i) + \sum_{l=0}^{L-1} \eta_{il_to_rv[i,l]}

    il_to_rv[i,l] is a linear number, while in the text random variables
    are indexed by a pair (floor(i/2^l),l)
    """
    il_to_rv = np.ndarray([N, L], "i")
    rv_cnt = -1
    for l in range(L):
        for i in range(N):
            if np.mod(i, 2**l) == 0:
                rv_cnt = rv_cnt + 1
            il_to_rv[i, l] = rv_cnt
    return (il_to_rv, rv_cnt + 1)


class rv_index:
    # Reference: https://gitlab.inria.fr/abarczew/ab_technical/-/blob/b957b869ab8024e003cd0fc69a356309429dd7be/medical-statistics-and-privacy/dp-cum/code/2024/utils/smooth.py#L42-52
    """
    class to pre-compute all indexation tables and similar info for
    binary tree organized random variables
    """

    def __init__(self, N):
        self.N = N
        self.L = int(np.ceil(np.log(N) / np.log(2))) + 1
        (self.il_to_rv, self.rv_cnt) = calc_il_to_rv(self.N, self.L)


def generate_noise(epsilon, rvidx):
    # Reference: https://gitlab.inria.fr/abarczew/ab_technical/-/blob/b957b869ab8024e003cd0fc69a356309429dd7be/medical-statistics-and-privacy/dp-cum/code/2024/utils/dp_counter.py#L4-26
    """
    generate DP noise

    :param epsilon: the epsilon parameter of the Laplace distribution
    :param rvidx: an rv_index structure

    :return:
        - eta:
        - sum_eta:
    """
    rv_cnt = rvidx.rv_cnt
    N = rvidx.N
    L = rvidx.L
    eta = np.random.laplace(0, 1 / epsilon, size=rv_cnt)
    sum_eta = np.zeros(N)
    for i in range(N):
        for j in range(L):
            sum_eta[i] = sum_eta[i] + eta[rvidx.il_to_rv[i, j]]
    return sum_eta
