import sys
from pathlib import Path

# Add the parent directory to the system path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import math
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from sklearn.metrics import roc_curve, precision_recall_curve
from fedcurve.utils import load_label_and_score, fed_simulation
from fedcurve.approx import (
    roc_curve_approx,
    pr_curve_approx_separate,
    pr_curve_approx_combine,
)
from fedcurve.error import area_error_roc, area_error_pr


def save_results(args, area_error):
    file = Path(f"./result/fedcurve/{args.dataset}_{args.classifier}_{args.curve}.txt")

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("w") as f:
            f.write(
                "area_error,n_clients,privacy,epsilon,"
                "noise_type,post_processing,n_q,"
                "pr_strategy,height,branch,interp\n"
            )

    with file.open("a") as f:
        f.write(
            f"{area_error},{args.n_clients},{args.privacy},{args.epsilon},"
            f"{args.noise_type},{args.post_processing},{args.n_q},"
            f"{args.pr_strategy},{args.height},{args.branch},{args.interp}\n"
        )


def get_common_parser():
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
        choices=["adult", "bank", "cover", "dota2"],
        help="Dataset name",
    )
    parser.add_argument(
        "--classifier", type=str, default="XGBClassifier", help="Classifier name"
    )
    parser.add_argument("--epsilon", type=float, default=1.0, help="Privacy budget")
    parser.add_argument("--n_reps", type=int, default=1, help="Number of repetitions")
    return parser


def parse_arguments():
    parser = get_common_parser()
    parser.add_argument("--n_clients", type=int, default=1, help="Number of clients")
    parser.add_argument(
        "--privacy",
        type=str,
        default="SA",
        choices=["EQ", "SA", "DDP", "LDP"],
        help="Privacy model",
    )
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
    parser.add_argument("--n_q", type=int, default=8, help="Number of quantiles")
    parser.add_argument(
        "--pr_strategy",
        type=str,
        default="separate",
        choices=["separate", "combine"],
        help="PR curve quantile strategy",
    )
    parser.add_argument("--branch", type=int, default=2, help="Tree branching factor")
    parser.add_argument(
        "--interp",
        type=str,
        default="pchip",
        choices=["midpoint", "linear", "pchip"],
        help="Interpolation method",
    )
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
    else:
        args.height = math.ceil(math.log(args.n_q, args.branch)) + 2

    # ROC curve does not need the pr_strategy
    if args.curve == "ROC":
        args.pr_strategy = "none"

    # Load y_true and y_score
    y_true, y_score = load_label_and_score(args.dataset, args.classifier)

    # To compute point errors, set drop_intermediate=False
    if args.curve == "ROC":
        fpr_true, tpr_true, roc_thresholds = roc_curve(y_true, y_score)
    else:  # "PR"
        precision_true, recall_true, pr_thresholds = precision_recall_curve(
            y_true, y_score
        )

    q_frac = np.linspace(0, 1, args.n_q)

    for _ in range(args.n_reps):
        if args.privacy == "EQ":
            # Split the positive and negative examples
            pos_idx = y_true == 1
            y_score_pos = y_score[pos_idx]
            y_score_neg = y_score[~pos_idx]

            # Compute the Exact Quantiles (EQ)
            n_pos = len(y_score_pos)
            q_pos = np.quantile(y_score_pos, q_frac)

            if args.pr_strategy == "combine":
                n_all = len(y_score)
                q_all = np.quantile(y_score, q_frac)
            else:  # args.pr_strategy in ["separate", "none"]
                n_neg = len(y_score_neg)
                q_neg = np.quantile(y_score_neg, q_frac)

        else:  # args.privacy in ["SA", "DDP", "LDP"]
            # Simulate federated learning
            hier_hist_pos, hier_hist_neg = fed_simulation(
                y_true,
                y_score,
                args.n_clients,
                args.height,
                args.branch,
                args.privacy,
                args.epsilon,
                args.noise_type,
                args.post_processing,
            )

            # Compute the quantiles
            q_pos = hier_hist_pos.n_quantile(q_frac)
            n_pos = hier_hist_pos.N

            if args.pr_strategy == "combine":
                hier_hist = hier_hist_pos.merge(hier_hist_neg)
                q_all = hier_hist.n_quantile(q_frac)
                n_all = hier_hist.N
            else:  # args.pr_strategy in ["separate", "none"]
                q_neg = hier_hist_neg.n_quantile(q_frac)
                n_neg = hier_hist_neg.N

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
