import numpy as np
from functools import partial
from .noise import DistributedDP_noise, LocalDP_noise


def fedhist_server(hist_client, height, branch, privacy, post_processing=True):
    if privacy == "SA":
        # For Secure Aggregation, server builds the hierarchical histogram
        hier_hist = [np.sum(hist_client, axis=0)]
        for _ in range(height):
            hist = hier_hist[0].reshape(-1, branch).sum(axis=1)
            # reshape(branch, -1).sum(axis=0) is incorrect
            hier_hist.insert(0, hist)

    elif privacy in ["DDP", "LDP"]:
        hier_hist = [
            sum(client[level] for client in hist_client) for level in range(height)
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

        # Add root node
        root = np.array([hier_hist[0].sum()])
        hier_hist.insert(0, root)

    else:
        raise ValueError(f"Unknown privacy model: {privacy}")

    return hier_hist


def fedhist_client(
    y_score, height, branch, privacy, eps, n_clients, noise_type="continuous"
):
    if privacy == "SA":
        # For Secure Aggregation, client just sends the leaf nodes
        n_leaf = branch**height
        hist = np.histogram(y_score, bins=np.linspace(0, 1, n_leaf + 1))[0]
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
            hist = np.histogram(y_score, bins=np.linspace(0, 1, n_node + 1))[0]
            hier_hist.append(hist + noise(size=hist.size))
        return hier_hist

    else:
        raise ValueError(f"Unknown privacy model: {privacy}")
