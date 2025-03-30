import math
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from sklearn.metrics import roc_curve, precision_recall_curve
from fedcurve.data import load_data, IID_partitioner
from fedcurve.classifier import predict_proba
from fedcurve.fedhist import get_n_quantile, fedhist_client, fedhist_server
from fedcurve.approx import (
    roc_curve_approx,
    pr_curve_approx_separate,
    pr_curve_approx_combine,
)
from fedcurve.error import area_error_roc, area_error_pr


def save_results(args, area_error):
    file = Path(f"./result/{args.dataset}_{args.classifier}_{args.curve}.txt")

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("w") as f:
            f.write(
                "area_error,n_clients,privacy,epsilon,noise_type,post_processing,"
                "n_q,pr_strategy,height,branch,interp\n"
            )

    with file.open("a") as f:
        f.write(
            f"{area_error},{args.n_clients},{args.privacy},{args.epsilon},"
            f"{args.noise_type},{args.post_processing},{args.n_q},"
            f"{args.pr_strategy},{args.height},{args.branch},{args.interp}\n"
        )


def parse_arguments():
    parser = ArgumentParser(
        description="Run the experiment with configurable parameters."
    )
    parser.add_argument(
        "--curve",
        type=str,
        default="ROC",
        choices=["ROC", "PR"],
        help="Curve name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        choices=["adult", "bank", "cover", "sep", "oct", "nov"],
        help="Dataset name",
    )
    parser.add_argument(
        "--classifier", type=str, default="XGBClassifier", help="Classifier name"
    )
    parser.add_argument("--n_clients", type=int, default=1, help="Number of clients")
    parser.add_argument(
        "--privacy",
        type=str,
        default="EQ",
        choices=["EQ", "SA", "DDP", "LDP"],
        help="Privacy model",
    )
    parser.add_argument("--epsilon", type=float, default=1.0, help="Privacy budget")
    parser.add_argument(
        "--noise_type",
        type=str,
        default="discrete",
        choices=["continuous", "discrete"],
        help="Type of DP noise",
    )
    parser.add_argument(
        "--post_processing",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Apply post-processing for DP",
    )
    parser.add_argument("--n_q", type=int, default=5, help="Number of quantiles")
    parser.add_argument(
        "--pr_strategy",
        type=str,
        default="separate",
        choices=["separate", "combine"],
        help="PR curve quantile strategy",
    )
    parser.add_argument("--height", type=str, default="auto", help="Tree height")
    parser.add_argument("--branch", type=int, default=2, help="Tree branching factor")
    parser.add_argument(
        "--interp",
        type=str,
        default="pchip",
        choices=["midpoint", "linear", "pchip"],
        help="Interpolation method",
    )
    parser.add_argument("--n_reps", type=int, default=1, help="Number of repetitions")
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.privacy in ["EQ", "SA"]:
        args.epsilon = np.inf
        args.noise_type = "none"
        args.post_processing = np.nan

    if args.privacy == "EQ":
        args.n_clients = np.nan
        args.height = np.nan
        args.branch = np.nan
    elif args.height == "auto":
        args.height = math.ceil(math.log(args.n_q, args.branch)) + 3
    else:  # Convert height to int
        args.height = int(args.height)

    # ROC curve does not need the pr_strategy
    if args.curve == "ROC":
        args.pr_strategy = "none"

    y_file = Path(f"./dataset/{args.dataset}/clf/{args.classifier}.npz")
    if y_file.exists():
        # Use cached y_true and y_score
        y_data = np.load(y_file)
        y_true, y_score = y_data["y_true"], y_data["y_score"]
    else:
        X, y_true = load_data(args.dataset)
        y_score = predict_proba(X, y_true, args.classifier)
        y_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(y_file, y_true=y_true, y_score=y_score)

    # To compute point errors, don't drop suboptimal thresholds
    if args.curve == "ROC":
        fpr_true, tpr_true, roc_thresholds = roc_curve(
            y_true, y_score, drop_intermediate=False
        )
    else:  # "PR"
        precision_true, recall_true, pr_thresholds = precision_recall_curve(
            y_true, y_score, drop_intermediate=False
        )

    for _ in range(args.n_reps):
        if args.privacy == "EQ":
            # Compute the exact quantiles
            q_frac = np.linspace(0, 1, args.n_q)

            pos_idx = y_true == 1
            y_score_pos = y_score[pos_idx]
            n_pos = len(y_score_pos)
            q_pos = np.quantile(y_score_pos, q_frac)

            if args.pr_strategy == "combine":
                n_all = len(y_score)
                q_all = np.quantile(y_score, q_frac)
            else:  # args.pr_strategy in ["separate", "none"]
                y_score_neg = y_score[~pos_idx]
                n_neg = len(y_score_neg)
                q_neg = np.quantile(y_score_neg, q_frac)

        else:
            # clients partioning
            y_clients = IID_partitioner(y_true, y_score, args.n_clients)

            # For SA, DDP, and LDP, construct the histograms
            hist_client, hist_pos_client, hist_neg_client = [], [], []
            for y_true_client, y_score_client in y_clients:
                pos_idx_client = y_true_client == 1
                y_score_pos_client = y_score_client[pos_idx_client]

                hist_pos_client.append(
                    fedhist_client(
                        y_score_pos_client,
                        args.height,
                        args.branch,
                        args.privacy,
                        args.epsilon,
                        args.n_clients,
                        args.noise_type,
                    )
                )

                if args.pr_strategy == "combine":
                    hist_client.append(
                        fedhist_client(
                            y_score_client,
                            args.height,
                            args.branch,
                            args.privacy,
                            args.epsilon,
                            args.n_clients,
                            args.noise_type,
                        )
                    )
                else:  # args.pr_strategy in ["separate", "none"]
                    y_score_neg_client = y_score_client[~pos_idx_client]

                    hist_neg_client.append(
                        fedhist_client(
                            y_score_neg_client,
                            args.height,
                            args.branch,
                            args.privacy,
                            args.epsilon,
                            args.n_clients,
                            args.noise_type,
                        )
                    )

            # Compute the quantiles
            hier_hist_pos = fedhist_server(
                hist_pos_client,
                args.height,
                args.branch,
                args.privacy,
                args.post_processing,
            )
            q_pos = get_n_quantile(args.n_q, hier_hist_pos)
            n_pos = sum(hier_hist_pos[0])

            if args.pr_strategy == "combine":
                hier_hist = fedhist_server(
                    hist_client,
                    args.height,
                    args.branch,
                    args.privacy,
                    args.post_processing,
                )
                q_all = get_n_quantile(args.n_q, hier_hist)
                n_all = sum(hier_hist[0])
            else:  # args.pr_strategy in ["separate", "none"]
                hier_hist_neg = fedhist_server(
                    hist_neg_client,
                    args.height,
                    args.branch,
                    args.privacy,
                    args.post_processing,
                )
                q_neg = get_n_quantile(args.n_q, hier_hist_neg)
                n_neg = sum(hier_hist_neg[0])

        # Compute the area error
        if args.curve == "ROC":
            fpr_func_approx, tpr_func_approx = roc_curve_approx(
                q_neg, q_pos, args.n_q, args.interp
            )

            ae_roc = area_error_roc(
                fpr_true, tpr_true, fpr_func_approx, tpr_func_approx
            )
            save_results(args, ae_roc)

        else:  # "PR"
            if args.pr_strategy == "separate":
                precision_func_approx, recall_func_approx = pr_curve_approx_separate(
                    q_neg, q_pos, args.n_q, n_neg, n_pos, args.interp
                )
            else:  # "combine"
                precision_func_approx, recall_func_approx = pr_curve_approx_combine(
                    q_pos, q_all, args.n_q, n_pos, n_all, args.interp
                )

            ae_pr = area_error_pr(
                precision_true,
                recall_true,
                pr_thresholds,
                precision_func_approx,
                recall_func_approx,
            )
            save_results(args, ae_pr)


if __name__ == "__main__":
    main()
