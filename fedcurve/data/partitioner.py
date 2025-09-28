import numpy as np
from sklearn.utils import check_random_state


def IID_partitioner(y_true, y_score, n_clients, rng=None):
    if len(y_true) != len(y_score):
        raise RuntimeError("y_true and y_score must have the same length")

    if n_clients == 1:
        return [(y_true, y_score)]

    # Shuffle
    rng = check_random_state(rng)
    permuted_indices = rng.permutation(len(y_true))
    y_true, y_score = y_true[permuted_indices], y_score[permuted_indices]

    # Split into approximately equal parts
    y_true_splits = np.array_split(y_true, n_clients)
    y_score_splits = np.array_split(y_score, n_clients)

    return list(zip(y_true_splits, y_score_splits))
