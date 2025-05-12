import sys
from pathlib import Path

# Add the parent directory to the system path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import numpy as np
from scipy.interpolate import interp1d
from functools import partial
from sklearn.metrics import roc_curve, precision_recall_curve
from fedcurve.main import get_common_parser
from fedcurve.utils import load_label_and_score
from fedcurve.approx import precision_func_separate
from fedcurve.error import area_error_roc, area_error_pr
from dpecdf.smooth import smooth_curve
from dpecdf.noise import rv_index, generate_noise


def save_results(args, area_error):
    file = Path(f"./result/dpecdf/{args.dataset}_{args.classifier}_{args.curve}.txt")

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("w") as f:
            f.write("area_error,ratio,epsilon,n_p,norm\n")

    with file.open("a") as f:
        f.write(f"{area_error},{args.ratio},{args.epsilon},{args.n_p},{args.norm}\n")


def parse_arguments():
    parser = get_common_parser()
    parser.add_argument("--n_p", type=int, default=8, help="Number of points")
    parser.add_argument(
        "--norm", type=int, default=1, choices=[1, 2], help="Norm for smoothing"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    y_true, y_score = load_label_and_score(args.dataset, args.classifier, args.ratio)

    # To compute point errors, set drop_intermediate=False
    if args.curve == "ROC":
        fpr_true, tpr_true, roc_thresholds = roc_curve(y_true, y_score)
    else:  # "PR"
        precision_true, recall_true, pr_thresholds = precision_recall_curve(
            y_true, y_score
        )

    pos_idx = y_true == 1
    y_score_pos = y_score[pos_idx]
    y_score_neg = y_score[~pos_idx]

    bins = np.linspace(0, 1, args.n_p + 1)
    hist_pos = np.histogram(y_score_pos, bins=bins)[0]
    hist_neg = np.histogram(y_score_neg, bins=bins)[0]

    tp = np.cumsum(hist_pos[::-1])
    fp = np.cumsum(hist_neg[::-1])

    rvidx = rv_index(args.n_p)
    scale = args.epsilon / rvidx.L

    for _ in range(args.n_reps):
        tp_noisy = tp + generate_noise(scale, rvidx)
        fp_noisy = fp + generate_noise(scale, rvidx)

        n_pos = max(tp_noisy)
        n_neg = max(fp_noisy)

        tpr_noisy = np.r_[0, tp_noisy] / n_pos
        fpr_noisy = np.r_[0, fp_noisy] / n_neg

        tpr_smooth = smooth_curve(tpr_noisy, norm=args.norm)
        fpr_smooth = smooth_curve(fpr_noisy, norm=args.norm)

        thresholds = np.linspace(1, 0, args.n_p + 1)
        tpr_func_approx = interp1d(
            thresholds, tpr_smooth, bounds_error=False, fill_value=(1, 0)
        )
        fpr_func_approx = interp1d(
            thresholds, fpr_smooth, bounds_error=False, fill_value=(1, 0)
        )

        # Compute the area error
        if args.curve == "ROC":
            ae_roc = area_error_roc(
                fpr_true, tpr_true, fpr_func_approx, tpr_func_approx
            )
            save_results(args, ae_roc)

        else:  # "PR"
            precision_func_approx = partial(
                precision_func_separate,
                fpr_func=fpr_func_approx,
                tpr_func=tpr_func_approx,
                n_neg=n_neg,
                n_pos=n_pos,
            )

            precision_func_separate
            ae_pr = area_error_pr(
                precision_true,
                recall_true,
                pr_thresholds,
                precision_func_approx,
                tpr_func_approx,  # tpr == recall
            )
            save_results(args, ae_pr)


if __name__ == "__main__":
    main()
