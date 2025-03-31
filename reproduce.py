import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser


def run_experiment(**params):
    cmd = ["python", "main.py"]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])

    subprocess.run(cmd, check=True)


def read_csv(dataset, classifier):
    file_prefix = f"./result/{dataset}_{classifier}_"
    # file_prefix = f"./result/archive/{dataset}/{classifier}/"
    roc = pd.read_csv(file_prefix + "ROC.txt")
    pr = pd.read_csv(file_prefix + "PR.txt")
    return roc, pr


def main(dataset, classifier, n_q_list):
    figsize = (5, 5)

    # Compare interp=["linear", "pchip"]
    # Fix: privacy="SA", pr_strategy="separate"
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
        run_experiment(**params)

    # Compare interp=["linear", "pchip"]
    # Fix: privacy="DDP", epsilon=1, pr_strategy="separate"
    exp_interp_ddp = [
        {
            "curve": curve,
            "dataset": dataset,
            "classifier": classifier,
            "privacy": "DDP",
            "epsilon": 1,
            "n_q": n_q,
            "pr_strategy": "separate",
            "interp": interp,
            "n_reps": 10,
        }
        for curve in ["ROC", "PR"]
        for n_q in n_q_list
        for interp in ["linear", "pchip"]
    ]
    for params in exp_interp_ddp:
        run_experiment(**params)

    # create plots
    roc, pr = read_csv(dataset, classifier)
    pr = pr[pr["pr_strategy"] == "separate"]

    for curve in ["ROC", "PR"]:
        df = roc if curve == "ROC" else pr
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(n_q_list, 1 / n_q_list, "k:", label="1/Q")
        # ax.plot(n_q_list, 1 / n_q_list**2, "y:", label="1/Q^2")

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

            epsilon = 1
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
        fig.savefig(img_path)
        plt.close(fig)

    ###########################################################################

    # Compare epsilon=[0.1, 0.3, 1]
    # Fix: privacy="DDP", interp="pchip", pr_strategy="separate"
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
            "n_reps": 10,
        }
        for curve in ["ROC", "PR"]
        for n_q in n_q_list
        for epsilon in [0.1, 0.3]
    ]
    for params in exp_epsilon_ddp:
        run_experiment(**params)

    # create plots
    roc, pr = read_csv(dataset, classifier)
    roc = roc[roc["interp"] == "pchip"]
    pr = pr[(pr["pr_strategy"] == "separate") & (pr["interp"] == "pchip")]

    for curve in ["ROC", "PR"]:
        df = roc if curve == "ROC" else pr
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(n_q_list, 1 / n_q_list, "k:", label="1/Q")
        # ax.plot(n_q_list, 1 / n_q_list**2, "y:", label="1/Q^2")

        df_sa = df[(df["privacy"] == "SA")]
        ax.plot(
            df_sa["n_q"],
            df_sa["area_error"],
            "o--",
            label=f"SA",
        )

        df_ddp = df[(df["privacy"] == "DDP")]
        for epsilon in [0.1, 0.3, 1]:
            df_ddp_epsilon = df_ddp[df_ddp["epsilon"] == epsilon]
            ae_mean, ae_var = [], []
            for n_q in n_q_list:
                df_q = df_ddp_epsilon[df_ddp_epsilon["n_q"] == n_q]
                ae_mean.append(df_q["area_error"].mean())
                ae_var.append(df_q["area_error"].var())

            ax.errorbar(
                n_q_list,
                ae_mean,
                yerr=ae_var,
                fmt="o-",
                label=rf"DDP $\epsilon$={epsilon}",
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
        fig.savefig(img_path)
        plt.close(fig)

    ###########################################################################

    # Compare pr_strategy=["separate", "combine"]
    # Fix: curve="PR", privacy="SA", interp="pchip"
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
        run_experiment(**params)

    # Compare pr_strategy=["separate", "combine"]
    # curve="PR", privacy="DDP", epsilon=1, interp="pchip"
    exp_pr_strategy_ddp = [
        {
            "curve": "PR",
            "dataset": dataset,
            "classifier": classifier,
            "privacy": "DDP",
            "epsilon": 1,
            "n_q": n_q,
            "pr_strategy": pr_strategy,
            "interp": "pchip",
            "n_reps": 10,
        }
        for pr_strategy in ["combine"]
        for n_q in n_q_list
    ]
    for params in exp_pr_strategy_ddp:
        run_experiment(**params)

    # create plots
    _, pr = read_csv(dataset, classifier)
    df = pr[pr["interp"] == "pchip"]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(n_q_list, 1 / n_q_list, "k:", label="1/Q")
    # ax.plot(n_q_list, 1 / n_q_list**2, "y:", label="1/Q^2")

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

        epsilon = 1
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
    fig.savefig(img_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description="Reproduce the results of the experiment.")
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
    args = parser.parse_args()

    if args.dataset in ["adult", "bank"]:
        n_q_list = np.array([5, 7, 10, 14, 19, 26, 37, 51, 72, 100])
    else:  # ["cover", "sep", "oct", "nov"]
        n_q_list = np.array(
            [5, 7, 10, 14, 19, 26, 37, 51, 72, 100, 136, 190, 264, 368, 514, 717, 1000]
        )

    main(args.dataset, args.classifier, n_q_list)
