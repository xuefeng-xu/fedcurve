import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae_mse(x_true, y_true, x_func_approx, y_func_approx, thresholds, sample_weight):
    x_approx = x_func_approx(thresholds)
    y_approx = y_func_approx(thresholds)

    mae_x = mean_absolute_error(x_true, x_approx, sample_weight=sample_weight)
    mae_y = mean_absolute_error(y_true, y_approx, sample_weight=sample_weight)

    mse_x = mean_squared_error(x_true, x_approx, sample_weight=sample_weight)
    mse_y = mean_squared_error(y_true, y_approx, sample_weight=sample_weight)

    return mae_x + mae_y, mse_x + mse_y


def mae_mse_roc(
    fpr_true, tpr_true, thresholds, fpr_func_approx, tpr_func_approx, y_score
):
    # The first point of ROC is always (0, 0) with threshold=np.inf
    fpr_true, tpr_true, thresholds = fpr_true[1:], tpr_true[1:], thresholds[1:]

    # Duplicate scores are merged, thus need to compute weighted average
    _, sample_weight = np.unique(y_score, return_counts=True)
    # threasholds in ROC are in descending order
    sample_weight = sample_weight[::-1]

    return mae_mse(
        fpr_true, tpr_true, fpr_func_approx, tpr_func_approx, thresholds, sample_weight
    )


def mae_mse_pr(
    precision_true,
    recall_true,
    thresholds,
    precision_func_approx,
    recall_func_approx,
    y_score,
):
    # The last point of PR is always (0, 1)
    precision_true, recall_true = precision_true[:-1], recall_true[:-1]

    # Duplicate scores are merged, thus need to compute weighted average
    _, sample_weight = np.unique(y_score, return_counts=True)
    # threasholds in PR are in ascending order

    return mae_mse(
        recall_true,
        precision_true,
        recall_func_approx,
        precision_func_approx,
        thresholds,
        sample_weight,
    )
