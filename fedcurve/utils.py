import numpy as np
from pathlib import Path
from .data import load_data, IID_partitioner
from .classifier import predict_proba
from .hist import fedhist_client, fedhist_server, HierarchicalHistogram


def load_label_and_score(dataset, classifier, ratio):
    # Use absolute path to avoid issues with relative paths
    PROJECT_ROOT = Path(__file__).parent.parent
    y_file = PROJECT_ROOT / f"dataset/{dataset}/clf/{classifier}.npz"

    not_make_imb = np.isnan(ratio)

    if y_file.exists() and not_make_imb:
        # Use cached y_true and y_score
        y_data = np.load(y_file)
        y_true, y_score = y_data["y_true"], y_data["y_score"]
    else:
        X, y_true = load_data(dataset, ratio)
        y_score = predict_proba(X, y_true, classifier)

        if not_make_imb:
            y_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(y_file, y_true=y_true, y_score=y_score)
    return y_true, y_score


def fed_simulation(
    y_true,
    y_score,
    n_clients,
    height,
    branch,
    privacy,
    epsilon,
    noise_type,
    post_processing,
):
    # clients partioning
    y_clients = IID_partitioner(y_true, y_score, n_clients)

    # Split positive and negative data
    hist_pos_client, hist_neg_client = [], []
    for y_true_client, y_score_client in y_clients:
        pos_idx_client = y_true_client == 1
        y_score_pos_client = y_score_client[pos_idx_client]
        y_score_neg_client = y_score_client[~pos_idx_client]

        hist_pos_client.append(
            fedhist_client(
                y_score_pos_client,
                height,
                branch,
                privacy,
                epsilon,
                n_clients,
                noise_type,
            )
        )

        hist_neg_client.append(
            fedhist_client(
                y_score_neg_client,
                height,
                branch,
                privacy,
                epsilon,
                n_clients,
                noise_type,
            )
        )

    # Construct hierarchical histograms
    hier_hist_pos = fedhist_server(
        hist_pos_client,
        height,
        branch,
        privacy,
        post_processing,
    )
    hier_hist_neg = fedhist_server(
        hist_neg_client,
        height,
        branch,
        privacy,
        post_processing,
    )
    return HierarchicalHistogram(hier_hist_pos), HierarchicalHistogram(hier_hist_neg)
