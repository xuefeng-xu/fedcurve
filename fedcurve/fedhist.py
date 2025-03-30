import numpy as np
from functools import partial
from .noise import DistributedDP_noise, LocalDP_noise


def get_n_quantile(n_q, hier_hist):
    height = len(hier_hist)
    branch = len(hier_hist[0])
    tol = 1 / (2 * branch**height)

    # Reference: method `get_quantile_estimate` in class `FOQuant`
    # https://figshare.com/s/607998e479b0778645f6
    def get_quantile(q_frac):
        lb, ub = 0, 1
        q = 0
        idx = 0

        for level in range(1, height + 1):
            interval = (ub - lb) / branch
            level_count = sum(hier_hist[level - 1])
            bin_frac = max(hier_hist[level - 1][idx] / level_count, 0.0)

            child_idx = branch * idx
            max_idx = idx + branch - 1

            while q + bin_frac < q_frac and idx < max_idx:
                q += bin_frac
                child_idx += branch
                idx += 1
                lb += interval
                bin_frac = max(hier_hist[level - 1][idx] / level_count, 0.0)

            ub = lb + interval
            idx = child_idx

            if abs(q_frac - q) < tol:
                break

        return lb

    return [get_quantile(q_frac) for q_frac in np.linspace(0, 1, n_q)]


def fedhist_server(hist_client, height, branch, privacy, post_processing=True):
    if privacy == "SA":
        # For Secure Aggregation, server builds the hierarchical histogram
        hier_hist = [np.sum(hist_client, axis=0)]
        for _ in range(height - 1):
            hist = hier_hist[0].reshape(-1, branch).sum(axis=1)
            # reshape(branch, -1).sum(axis=0) is incorrect
            hier_hist.insert(0, hist)

    elif privacy in ["DDP", "LDP"]:
        hier_hist = [
            sum(client[level - 1] for client in hist_client)
            for level in range(1, height + 1)
        ]

        if post_processing:
            # Weighted averaging (bottom up, skip leaf nodes)
            for level in range(height - 1, 0, -1):
                i = height - level + 1
                scale = (branch**i - branch ** (i - 1)) / (branch**i - 1)

                hier_hist[level - 1] = scale * hier_hist[level - 1] + (
                    1 - scale
                ) * hier_hist[level].reshape(-1, branch).sum(axis=1)

            # Mean Consistency (top down, skip root node)
            for level in range(2, height + 1):
                diff = (
                    hier_hist[level - 2]
                    - hier_hist[level - 1].reshape(-1, branch).sum(axis=1)
                ) / branch
                hier_hist[level - 1] = hier_hist[level - 1] + np.repeat(diff, branch)

    else:
        raise ValueError(f"Unknown privacy model: {privacy}")

    return hier_hist


def fedhist_client(
    y_score, height, branch, privacy, eps, n_clients, noise_type="continuous"
):
    if privacy == "SA":
        # For Secure Aggregation, client just sends the leaf nodes
        n_leaf = branch**height
        hist, _ = np.histogram(y_score, bins=np.linspace(0, 1, n_leaf + 1))
        return hist

    elif privacy in ["DDP", "LDP"]:
        # For histogram, the sensitivity is 1
        sens = 1

        if privacy == "DDP":
            noise = partial(
                DistributedDP_noise,
                eps=eps / height,
                sens=sens,
                n_clients=n_clients,
                noise_type=noise_type,
            )
        else:  # "LDP"
            noise = partial(
                LocalDP_noise, eps=eps / height, sens=sens, noise_type=noise_type
            )

        hier_hist = []
        for level in range(1, height + 1):
            n_node = branch**level
            hist, _ = np.histogram(y_score, bins=np.linspace(0, 1, n_node + 1))
            hier_hist.append(hist + noise(size=hist.size))
        return hier_hist

    else:
        raise ValueError(f"Unknown privacy model: {privacy}")
