import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser


def _check(method):
    if method not in ["fedcurve", "dpecdf"]:
        raise ValueError(f"Unsupported method: {method}")


def run_experiment(method, **params):
    _check(method)

    cmd = ["python", f"{method}/main.py"]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])

    subprocess.run(cmd, check=True)


def read_csv(method, dataset, classifier):
    _check(method)

    file_prefix = f"./result/{method}/{dataset}_{classifier}_"
    roc = pd.read_csv(file_prefix + "ROC.txt")
    pr = pd.read_csv(file_prefix + "PR.txt")
    return roc, pr


def run_fedcurve(dataset, classifier, n_q_list, epsilon, figsize):
    n_q_list = np.asarray(n_q_list)

    # Compare interp=["linear", "pchip"]
    # Fix privacy="SA", pr_strategy="separate"
    exp_interp_sa = [
        {
            "curve": curve,
            "dataset": dataset,
            "classifier": classifier,
            "privacy": "SA",
            "n_q": n_q,
            "pr_strategy": "separate",
            "interp": interp,
            "n_reps": 1,
        }
        for curve in ["ROC", "PR"]
        for n_q in n_q_list
        for interp in ["linear", "pchip"]
    ]
    for params in exp_interp_sa:
        run_experiment("fedcurve", **params)

    # Compare interp=["linear", "pchip"]
    # Fix privacy="DDP", epsilon=epsilon, pr_strategy="separate"
    exp_interp_ddp = [
        {
            "curve": curve,
            "dataset": dataset,
            "classifier": classifier,
            "privacy": "DDP",
            "epsilon": epsilon,
            "n_q": n_q,
            "pr_strategy": "separate",
            "interp": interp,
            "n_reps": 5,
        }
        for curve in ["ROC", "PR"]
        for n_q in n_q_list
        for interp in ["linear", "pchip"]
    ]
    for params in exp_interp_ddp:
        run_experiment("fedcurve", **params)

    # create plots
    roc, pr = read_csv("fedcurve", dataset, classifier)
    roc = roc[roc["ratio"].isna()]
    pr = pr[(pr["pr_strategy"] == "separate") & pr["ratio"].isna()]

    for curve in ["ROC", "PR"]:
        df = roc if curve == "ROC" else pr
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(n_q_list, 1 / n_q_list, "k:", label="1/Q")

        df_sa = df[df["privacy"] == "SA"]
        df_ddp = df[df["privacy"] == "DDP"]

        for interp in ["linear", "pchip"]:
            df_sa_interp = df_sa[df_sa["interp"] == interp]
            ax.plot(
                df_sa_interp["n_q"],
                df_sa_interp["area_error"],
                "o--",
                label=f"SA, {interp}",
            )

            df_ddp_interp = df_ddp[
                (df_ddp["interp"] == interp) & (df_ddp["epsilon"] == epsilon)
            ]
            ae_mean, ae_var = [], []
            for n_q in n_q_list:
                df_q = df_ddp_interp[df_ddp_interp["n_q"] == n_q]
                ae_mean.append(df_q["area_error"].mean())
                ae_var.append(df_q["area_error"].var())

            ax.errorbar(
                n_q_list,
                ae_mean,
                yerr=ae_var,
                fmt="o-",
                label=rf"DDP $\epsilon$={epsilon}, {interp}",
            )

        ax.set_xlabel("Q")
        ax.set_ylabel("Area Error")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid()
        img_path = Path(f"./img/interp/{dataset}_{classifier}_{curve}.pdf")
        if not img_path.parent.exists():
            img_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(img_path)
        plt.close(fig)

    ###########################################################################

    # Compare EQ, SA, epsilon=[0.1, 0.3, 1]
    # Fix privacy="EQ", interp="pchip", pr_strategy="separate"
    exp_eq = [
        {
            "curve": curve,
            "dataset": dataset,
            "classifier": classifier,
            "privacy": "EQ",
            "n_q": n_q,
            "pr_strategy": "separate",
            "interp": "pchip",
            "n_reps": 1,
        }
        for curve in ["ROC", "PR"]
        for n_q in n_q_list
    ]
    for params in exp_eq:
        run_experiment("fedcurve", **params)

    # Compare EQ, SA, epsilon=[0.1, 0.3, 1]
    # Fix privacy="DDP", interp="pchip", pr_strategy="separate"
    epsilon_list = [0.1, 0.3, 1]
    eps_exec = epsilon_list.copy()
    eps_exec.remove(epsilon)

    exp_epsilon_ddp = [
        {
            "curve": curve,
            "dataset": dataset,
            "classifier": classifier,
            "privacy": "DDP",
            "epsilon": epsilon,
            "n_q": n_q,
            "pr_strategy": "separate",
            "interp": "pchip",
            "n_reps": 5,
        }
        for curve in ["ROC", "PR"]
        for n_q in n_q_list
        for epsilon in eps_exec
    ]
    for params in exp_epsilon_ddp:
        run_experiment("fedcurve", **params)

    # create plots
    roc, pr = read_csv("fedcurve", dataset, classifier)
    roc = roc[(roc["interp"] == "pchip") & roc["ratio"].isna()]
    pr = pr[
        (pr["pr_strategy"] == "separate")
        & (pr["interp"] == "pchip")
        & pr["ratio"].isna()
    ]

    for curve in ["ROC", "PR"]:
        df = roc if curve == "ROC" else pr
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(n_q_list, 1 / n_q_list, "k:", label="1/Q")

        df_eq = df[(df["privacy"] == "EQ")]
        ax.plot(
            df_eq["n_q"],
            df_eq["area_error"],
            "o--",
            label=f"EQ",
        )

        df_sa = df[(df["privacy"] == "SA")]
        ax.plot(
            df_sa["n_q"],
            df_sa["area_error"],
            "o--",
            label=f"SA",
        )

        df_ddp = df[(df["privacy"] == "DDP")]
        for eps in epsilon_list:
            df_ddp_eps = df_ddp[df_ddp["epsilon"] == eps]
            ae_mean, ae_var = [], []
            for n_q in n_q_list:
                df_q = df_ddp_eps[df_ddp_eps["n_q"] == n_q]
                ae_mean.append(df_q["area_error"].mean())
                ae_var.append(df_q["area_error"].var())

            ax.errorbar(
                n_q_list,
                ae_mean,
                yerr=ae_var,
                fmt="o-",
                label=rf"DDP $\epsilon$={eps}",
            )

        ax.set_xlabel("Q")
        ax.set_ylabel("Area Error")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid()
        img_path = Path(f"./img/epsilon/{dataset}_{classifier}_{curve}.pdf")
        if not img_path.parent.exists():
            img_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(img_path)
        plt.close(fig)

    ###########################################################################

    # Compare pr_strategy=["separate", "combine"]
    # Fix curve="PR", privacy="SA", interp="pchip"
    exp_pr_strategy_sa = [
        {
            "curve": "PR",
            "dataset": dataset,
            "classifier": classifier,
            "privacy": "SA",
            "n_q": n_q,
            "pr_strategy": pr_strategy,
            "interp": "pchip",
            "n_reps": 1,
        }
        for pr_strategy in ["combine"]
        for n_q in n_q_list
    ]
    for params in exp_pr_strategy_sa:
        run_experiment("fedcurve", **params)

    # Compare pr_strategy=["separate", "combine"]
    # curve="PR", privacy="DDP", epsilon=epsilon, interp="pchip"
    exp_pr_strategy_ddp = [
        {
            "curve": "PR",
            "dataset": dataset,
            "classifier": classifier,
            "privacy": "DDP",
            "epsilon": epsilon,
            "n_q": n_q,
            "pr_strategy": pr_strategy,
            "interp": "pchip",
            "n_reps": 5,
        }
        for pr_strategy in ["combine"]
        for n_q in n_q_list
    ]
    for params in exp_pr_strategy_ddp:
        run_experiment("fedcurve", **params)

    # create plots
    _, pr = read_csv("fedcurve", dataset, classifier)
    df = pr[(pr["interp"] == "pchip") & pr["ratio"].isna()]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(n_q_list, 1 / n_q_list, "k:", label="1/Q")

    df_sa = df[df["privacy"] == "SA"]
    df_ddp = df[df["privacy"] == "DDP"]

    for pr_strategy in ["separate", "combine"]:
        df_sa_pr = df_sa[df_sa["pr_strategy"] == pr_strategy]
        ax.plot(
            df_sa_pr["n_q"],
            df_sa_pr["area_error"],
            "o--",
            label=f"SA, {pr_strategy}",
        )

        df_ddp_pr = df_ddp[
            (df_ddp["pr_strategy"] == pr_strategy) & (df_ddp["epsilon"] == epsilon)
        ]
        ae_mean, ae_var = [], []
        for n_q in n_q_list:
            df_q = df_ddp_pr[df_ddp_pr["n_q"] == n_q]
            ae_mean.append(df_q["area_error"].mean())
            ae_var.append(df_q["area_error"].var())

        ax.errorbar(
            n_q_list,
            ae_mean,
            yerr=ae_var,
            fmt="o-",
            label=rf"DDP $\epsilon$={epsilon}, {pr_strategy}",
        )

    ax.set_xlabel("Q")
    ax.set_ylabel("Area Error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    img_path = Path(f"./img/pr_strategy/{dataset}_{classifier}_PR.pdf")
    if not img_path.parent.exists():
        img_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)


def run_dpecdf(dataset, classifier, n_p_list, epsilon, figsize):
    n_p_list = np.asarray(n_p_list)

    exp_dp = [
        {
            "curve": curve,
            "dataset": dataset,
            "classifier": classifier,
            "epsilon": epsilon,
            "n_p": n_p,
            "norm": norm,
            "n_reps": 5,
        }
        for curve in ["ROC", "PR"]
        for n_p in n_p_list
        for norm in [1, 2]
    ]
    for params in exp_dp:
        run_experiment("dpecdf", **params)

    # create plots
    range_roc, range_pr = read_csv("dpecdf", dataset, classifier)
    quantile_roc, quantile_pr = read_csv("fedcurve", dataset, classifier)

    range_roc = range_roc[range_roc["ratio"].isna()]
    range_pr = range_pr[range_pr["ratio"].isna()]

    quantile_roc = quantile_roc[
        (quantile_roc["interp"] == "pchip") & quantile_roc["ratio"].isna()
    ]
    quantile_pr = quantile_pr[
        (quantile_pr["pr_strategy"] == "separate")
        & (quantile_pr["interp"] == "pchip")
        & quantile_pr["ratio"].isna()
    ]

    for curve in ["ROC", "PR"]:
        range_df, quantile_df = (
            (range_roc, quantile_roc) if curve == "ROC" else (range_pr, quantile_pr)
        )

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(n_p_list, 1 / n_p_list, "k:", label="1/Q")

        range_df_ddp = range_df[(range_df["epsilon"] == epsilon)]
        quantile_df_ddp = quantile_df[(quantile_df["epsilon"] == epsilon)]

        range_ae_mean_norm1, range_ae_var_norm1 = [], []
        range_ae_mean_norm2, range_ae_var_norm2 = [], []
        quantile_ae_mean, quantile_ae_var = [], []
        for n_q in n_q_list:
            range_df_q = range_df_ddp[range_df_ddp["n_p"] == n_q]
            range_df_q_norm1 = range_df_q[range_df_q["norm"] == 1]
            range_df_q_norm2 = range_df_q[range_df_q["norm"] == 2]

            range_ae_mean_norm1.append(range_df_q_norm1["area_error"].mean())
            range_ae_var_norm1.append(range_df_q_norm1["area_error"].var())
            range_ae_mean_norm2.append(range_df_q_norm2["area_error"].mean())
            range_ae_var_norm2.append(range_df_q_norm2["area_error"].var())

            quantile_df_q = quantile_df_ddp[quantile_df_ddp["n_q"] == n_q]
            quantile_ae_mean.append(quantile_df_q["area_error"].mean())
            quantile_ae_var.append(quantile_df_q["area_error"].var())

        ax.errorbar(
            n_q_list,
            range_ae_mean_norm1,
            yerr=range_ae_var_norm1,
            fmt="o-",
            label=rf"Range ($l_1$ norm) $\epsilon$={epsilon}",
        )
        ax.errorbar(
            n_q_list,
            range_ae_mean_norm2,
            yerr=range_ae_var_norm2,
            fmt="o-",
            label=rf"Range ($l_2$ norm) $\epsilon$={epsilon}",
        )
        ax.errorbar(
            n_q_list,
            quantile_ae_mean,
            yerr=quantile_ae_var,
            fmt="o-",
            label=rf"Quantile $\epsilon$={epsilon}",
        )

        ax.set_xlabel("Q(N)")
        ax.set_ylabel("Area Error")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid()
        img_path = Path(f"./img/range/{dataset}_{classifier}_{curve}.pdf")
        if not img_path.parent.exists():
            img_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(img_path)
        plt.close(fig)


def run_imb_ratio(dataset, classifier, n_q_list, ratio_list, figsize):
    exp_eq = [
        {
            "curve": "PR",
            "dataset": dataset,
            "ratio": ratio,
            "classifier": classifier,
            "privacy": "EQ",
            "n_q": n_q,
            "pr_strategy": "separate",
            "interp": "pchip",
            "n_reps": 5,
        }
        for ratio in ratio_list
        for n_q in n_q_list
    ]
    for params in exp_eq:
        run_experiment("fedcurve", **params)

    # create plots
    _, df = read_csv("fedcurve", dataset, classifier)
    df = df[(df["privacy"] == "EQ") & df["ratio"].notna()]

    fig, ax = plt.subplots(figsize=figsize)

    for ratio in ratio_list:
        df_ratio = df[df["ratio"] == ratio]
        ae_mean, ae_var = [], []
        for n_q in n_q_list:
            df_q = df_ratio[df_ratio["n_q"] == n_q]
            ae_mean.append(df_q["area_error"].mean())
            ae_var.append(df_q["area_error"].var())

        ax.errorbar(
            n_q_list,
            ae_mean,
            yerr=ae_var,
            fmt="o-",
            label=f"r={ratio}",
        )

    ax.set_xlabel("Q")
    ax.set_ylabel("Area Error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    img_path = Path(f"./img/ratio/{dataset}_{classifier}_PR.pdf")
    if not img_path.parent.exists():
        img_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description="Reproduce the results of the experiment.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        help="Dataset name",
    )
    parser.add_argument(
        "--classifier", type=str, default="XGBClassifier", help="Classifier name"
    )
    args = parser.parse_args()

    n_q_list = [2**i for i in range(2, 11)]
    ratio_list = [0.1, 0.05, 0.01]

    if args.dataset in ["cover", "dota2"]:
        epsilon = 0.3
    else:
        epsilon = 1.0

    figsize = (5, 5)
    run_fedcurve(args.dataset, args.classifier, n_q_list, epsilon, figsize)
    run_dpecdf(args.dataset, args.classifier, n_q_list, epsilon, figsize)
    run_imb_ratio(args.dataset, args.classifier, n_q_list, ratio_list, figsize)
