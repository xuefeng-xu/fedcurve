import numpy as np
from functools import partial
from .interp import piecewise_interp


def roc_curve_approx(q_neg, q_pos, n_q, interp):
    q_frac = np.linspace(1, 0, n_q)
    fpr_func_approx = piecewise_interp(q_neg, q_frac, interp)
    tpr_func_approx = piecewise_interp(q_pos, q_frac, interp)
    return fpr_func_approx, tpr_func_approx


def precision_func_separate(score, fpr_func, tpr_func, n_neg, n_pos):
    fp = fpr_func(score) * n_neg
    tp = tpr_func(score) * n_pos
    pp = fp + tp
    precision = np.ones_like(score)
    np.divide(tp, pp, out=precision, where=(pp != 0))
    return precision


def pr_curve_approx_separate(q_neg, q_pos, n_q, n_neg, n_pos, interp):
    # separate negative and positive samples
    q_frac = np.linspace(1, 0, n_q)
    fpr_func_approx = piecewise_interp(q_neg, q_frac, interp)
    recall_func_approx = piecewise_interp(q_pos, q_frac, interp)

    precision_func_approx = partial(
        precision_func_separate,
        fpr_func=fpr_func_approx,
        tpr_func=recall_func_approx,  # tpr == recall
        n_neg=n_neg,
        n_pos=n_pos,
    )
    return precision_func_approx, recall_func_approx


def precision_func_combine(score, tpr_func, ppr_func, n_pos, n_all):
    tp = tpr_func(score) * n_pos
    pp = ppr_func(score) * n_all
    precision = np.ones_like(score)
    np.divide(tp, pp, out=precision, where=(pp != 0))
    return np.clip(precision, 0.0, 1.0)


def pr_curve_approx_combine(q_pos, q_all, n_q, n_pos, n_all, interp):
    # combine negative and positive samples
    q_frac = np.linspace(1, 0, n_q)
    ppr_func_approx = piecewise_interp(q_all, q_frac, interp)
    recall_func_approx = piecewise_interp(q_pos, q_frac, interp)

    precision_func_approx = partial(
        precision_func_combine,
        tpr_func=recall_func_approx,  # tpr == recall
        ppr_func=ppr_func_approx,
        n_pos=n_pos,
        n_all=n_all,
    )
    return precision_func_approx, recall_func_approx
