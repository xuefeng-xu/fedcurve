import numpy as np


def IID_partitioner(y_true, y_score, n_clients):
    # Shuffle
    n_samples = len(y_true)
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    y_true, y_score = y_true[idx], y_score[idx]

    # Split
    y_clients = []
    start = end = 0
    for i in range(n_clients):
        start = end
        end = int(n_samples * (i + 1) / n_clients)
        y_clients.append((y_true[start:end], y_score[start:end]))

    return y_clients
