import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad_vec
from sklearn.metrics import auc
from functools import partial


def abs_diff_func(x, y_true, y_approx):
    return abs(y_true(x) - y_approx(x))


def area_error_roc(fpr_true, tpr_true, fpr_func_approx, tpr_func_approx, n_scores=1e4):
    roc_true = interp1d(fpr_true, tpr_true, kind="linear", assume_sorted=True)

    # Approximate tpr(score) ~ fpr(score)
    s = np.linspace(1, 0, int(n_scores))
    fpr_approx = fpr_func_approx(s)
    tpr_approx = tpr_func_approx(s)
    roc_approx = interp1d(
        fpr_approx, tpr_approx, kind="linear", bounds_error=False, fill_value=(0, 1)
    )

    abs_diff = partial(abs_diff_func, y_true=roc_true, y_approx=roc_approx)
    res, _ = quad_vec(abs_diff, 0, 1, workers=-1)
    return res


def auc_error_roc(fpr_true, tpr_true, fpr_func_approx, tpr_func_approx, n_scores=1e4):
    # thresholds in ROC are in descending order
    s = np.linspace(1, 0, int(n_scores))
    fpr_approx = fpr_func_approx(s)
    tpr_approx = tpr_func_approx(s)

    # difference of auc score
    auc_true = auc(fpr_true, tpr_true)
    auc_approx = auc(fpr_approx, tpr_approx)
    return abs(auc_true - auc_approx)


def area_error_pr(
    precision_true,
    recall_true,
    thresholds,
    precision_func_approx,
    recall_func_approx,
    n_scores=1e4,
):
    # Same as in sklearn.metrics.PrecisionRecallDisplay
    # drawstyle = "steps-post" --> kind = "previous"
    precision_func_true = interp1d(
        thresholds,
        precision_true[:-1],
        kind="previous",
        bounds_error=False,
        fill_value=(precision_true[0], 1),
    )

    recall_func_true = interp1d(
        thresholds,
        recall_true[:-1],
        kind="linear",
        bounds_error=False,
        fill_value=(1, 0),
    )

    # Approximate precision(score) ~ recall(score)
    s = np.linspace(1, 0, int(n_scores))
    pr_true = interp1d(
        recall_func_true(s),
        precision_func_true(s),
        kind="linear",
        bounds_error=False,
        fill_value=(1, precision_true[0]),
    )
    pr_approx = interp1d(
        recall_func_approx(s),
        precision_func_approx(s),
        kind="linear",
        bounds_error=False,
        fill_value=(1, precision_true[0]),
    )

    abs_diff = partial(abs_diff_func, y_true=pr_true, y_approx=pr_approx)
    res, _ = quad_vec(abs_diff, 0, 1, workers=-1)
    return res


def avg_prec(precision, recall):
    # sklearn.metrics.average_precision_score
    return max(0.0, -np.sum(np.diff(recall) * np.array(precision)[:-1]))


def auc_error_pr(
    precision_true, recall_true, precision_func_approx, recall_func_approx, n_scores=1e4
):
    # thresholds in PR are in ascending order
    s = np.linspace(0, 1, int(n_scores))
    precision_approx = precision_func_approx(s)
    recall_approx = recall_func_approx(s)

    # difference of average precision score
    avg_prec_true = avg_prec(precision_true, recall_true)
    avg_prec_approx = avg_prec(precision_approx, recall_approx)
    return abs(avg_prec_true - avg_prec_approx)
