import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


if __name__ == "__main__":
    dataset = "nov"
    # classifier = "LogisticRegression"
    classifier = "XGBClassifier"

    file = f"./result/{dataset}/{classifier}/PR.txt"
    df = pd.read_csv(file)

    n_clients_list = [100, 300, 1000, 3000]
    privacy_methods = ["DDP", "LDP"]
    epsilons = [0.1, 0.3, 1]
    noise_types = ["continuous", "discrete"]
    # pr_strategies = ["separate", "combine"]
    pr_strategies = ["separate"]
    n_q_list = np.array([5, 7, 10, 14, 19, 26, 37, 51, 72, 100])

    df_eq_sepa = df[(df["privacy"] == "EQ") & (df["pr_strategy"] == "separate")]
    df_eq_comb = df[(df["privacy"] == "EQ") & (df["pr_strategy"] == "combine")]
    df_sa_sepa = df[(df["privacy"] == "SA") & (df["pr_strategy"] == "separate")]
    df_sa_comb = df[(df["privacy"] == "SA") & (df["pr_strategy"] == "combine")]

    # compare number of clients
    for privacy_method in privacy_methods:
        for epsilon in epsilons:
            for noise_type in noise_types:
                for pr_strategy in pr_strategies:
                    df_filtered = df[
                        (df["privacy"] == privacy_method)
                        & (df["epsilon"] == epsilon)
                        & (df["noise_type"] == noise_type)
                        & (df["pr_strategy"] == pr_strategy)
                    ]

                    if pr_strategy == "separate":
                        df_sa = df_sa_sepa
                        df_eq = df_eq_sepa
                    else:
                        df_sa = df_sa_comb
                        df_eq = df_eq_comb

                    fig, ax = plt.subplots()
                    ax.plot(df_sa["n_q"], 1 / df_sa["n_q"], "y:", label="1/Q")
                    ax.plot(
                        df_sa["n_q"], 1 / df_sa["n_q"] ** 1.5, "k:", label="1/Q^1.5"
                    )
                    ax.plot(df_sa["n_q"], 1 / df_sa["n_q"] ** 2, "r:", label="1/Q^2")

                    ax.plot(df_eq["n_q"], df_eq["area_error"], "o--", label="EQ")
                    ax.plot(df_sa["n_q"], df_sa["area_error"], "o--", label="SA")

                    for n_clients in n_clients_list:
                        df_n_clients = df_filtered[
                            df_filtered["n_clients"] == n_clients
                        ]

                        area_error_mean, area_error_var = [], []
                        for n_q in n_q_list:
                            df_q = df_n_clients[df_n_clients["n_q"] == n_q]
                            area_error_mean.append(df_q["area_error"].mean())
                            area_error_var.append(df_q["area_error"].var())

                        ax.errorbar(
                            n_q_list,
                            area_error_mean,
                            yerr=area_error_var,
                            fmt="o-",
                            label=f"{n_clients} clients",
                        )

                    ax.set_xlabel("Q")
                    ax.set_ylabel("Area Error")
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.legend()
                    ax.set_title(
                        rf"{privacy_method}, $\epsilon$={epsilon}, Noise Type: {noise_type}, Strategy: {pr_strategy}"
                    )

                    img_path = Path(
                        f"./img/{dataset}/{classifier}/PR/n_clients/privacy_{privacy_method}_epsilon_{epsilon}_noise_type_{noise_type}_pr_strategy_{pr_strategy}.pdf"
                    )
                    if not img_path.parent.exists():
                        img_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(img_path)
                    plt.close(fig)

    # compare epsilon
    for n_clients in n_clients_list:
        for privacy_method in privacy_methods:
            for noise_type in noise_types:
                for pr_strategy in pr_strategies:
                    df_filtered = df[
                        (df["n_clients"] == n_clients)
                        & (df["privacy"] == privacy_method)
                        & (df["noise_type"] == noise_type)
                        & (df["pr_strategy"] == pr_strategy)
                    ]

                    if pr_strategy == "separate":
                        df_sa = df_sa_sepa
                        df_eq = df_eq_sepa
                    else:
                        df_sa = df_sa_comb
                        df_eq = df_eq_comb

                    fig, ax = plt.subplots()
                    ax.plot(df_sa["n_q"], 1 / df_sa["n_q"], "y:", label="1/Q")
                    ax.plot(
                        df_sa["n_q"], 1 / df_sa["n_q"] ** 1.5, "k:", label="1/Q^1.5"
                    )
                    ax.plot(df_sa["n_q"], 1 / df_sa["n_q"] ** 2, "r:", label="1/Q^2")

                    ax.plot(df_eq["n_q"], df_eq["area_error"], "o--", label="EQ")
                    ax.plot(df_sa["n_q"], df_sa["area_error"], "o--", label="SA")

                    for epsilon in epsilons:
                        df_epsilon = df_filtered[df_filtered["epsilon"] == epsilon]

                        area_error_mean, area_error_var = [], []
                        for n_q in n_q_list:
                            df_q = df_epsilon[df_epsilon["n_q"] == n_q]
                            area_error_mean.append(df_q["area_error"].mean())
                            area_error_var.append(df_q["area_error"].var())

                        ax.errorbar(
                            n_q_list,
                            area_error_mean,
                            yerr=area_error_var,
                            fmt="o-",
                            label=rf"$\epsilon$={epsilon}",
                        )

                    ax.set_xlabel("Q")
                    ax.set_ylabel("Area Error")
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.legend()
                    ax.set_title(
                        rf"{privacy_method}, #Clients={n_clients}, Noise Type: {noise_type}, Strategy: {pr_strategy}"
                    )

                    img_path = Path(
                        f"./img/{dataset}/{classifier}/PR/epsilon/n_clients_{n_clients}_privacy_{privacy_method}_noise_type_{noise_type}_pr_strategy_{pr_strategy}.pdf"
                    )
                    if not img_path.parent.exists():
                        img_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(img_path)
                    plt.close(fig)

    # compare pr strategies
    for n_clients in n_clients_list:
        for privacy_method in privacy_methods:
            for noise_type in noise_types:
                for epsilon in epsilons:
                    df_filtered = df[
                        (df["n_clients"] == n_clients)
                        & (df["privacy"] == privacy_method)
                        & (df["noise_type"] == noise_type)
                        & (df["epsilon"] == epsilon)
                    ]

                    fig, ax = plt.subplots()
                    ax.plot(n_q_list, 1 / n_q_list, "y:", label="1/Q")
                    ax.plot(n_q_list, 1 / n_q_list**1.5, "k:", label="1/Q^1.5")
                    ax.plot(n_q_list, 1 / n_q_list**2, "r:", label="1/Q^2")

                    ax.plot(
                        df_eq_sepa["n_q"],
                        df_eq_sepa["area_error"],
                        "o--",
                        label="EQ separate",
                    )
                    ax.plot(
                        df_eq_comb["n_q"],
                        df_eq_comb["area_error"],
                        "o--",
                        label="EQ combine",
                    )
                    ax.plot(
                        df_sa_sepa["n_q"],
                        df_sa_sepa["area_error"],
                        "o--",
                        label="SA separate",
                    )
                    ax.plot(
                        df_sa_comb["n_q"],
                        df_sa_comb["area_error"],
                        "o--",
                        label="SA combine",
                    )

                    for pr_strategy in pr_strategies:
                        df_pr_strategy = df_filtered[
                            df_filtered["pr_strategy"] == pr_strategy
                        ]

                        area_error_mean, area_error_var = [], []
                        for n_q in n_q_list:
                            df_q = df_pr_strategy[df_pr_strategy["n_q"] == n_q]
                            area_error_mean.append(df_q["area_error"].mean())
                            area_error_var.append(df_q["area_error"].var())

                        ax.errorbar(
                            n_q_list,
                            area_error_mean,
                            yerr=area_error_var,
                            fmt="o-",
                            label=f"pr_strategy {pr_strategy}",
                        )

                    ax.set_xlabel("Q")
                    ax.set_ylabel("Area Error")
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.legend()
                    ax.set_title(
                        rf"{privacy_method}, $\epsilon$={epsilon}, #Clients={n_clients}, Noise Type: {noise_type}"
                    )

                    img_path = Path(
                        f"./img/{dataset}/{classifier}/PR/pr_strategy/n_clients_{n_clients}_privacy_{privacy_method}_epsilon_{epsilon}_noise_type_{noise_type}.pdf"
                    )
                    if not img_path.parent.exists():
                        img_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(img_path)
                    plt.close(fig)

    # compare noise type
    for privacy_method in privacy_methods:
        for epsilon in epsilons:
            for n_clients in n_clients_list:
                for pr_strategy in pr_strategies:
                    df_filtered = df[
                        (df["privacy"] == privacy_method)
                        & (df["epsilon"] == epsilon)
                        & (df["n_clients"] == n_clients)
                        & (df["pr_strategy"] == pr_strategy)
                    ]

                    if pr_strategy == "separate":
                        df_sa = df_sa_sepa
                        df_eq = df_eq_sepa
                    else:
                        df_sa = df_sa_comb
                        df_eq = df_eq_comb

                    fig, ax = plt.subplots()
                    ax.plot(df_sa["n_q"], 1 / df_sa["n_q"], "y:", label="1/Q")
                    ax.plot(
                        df_sa["n_q"], 1 / df_sa["n_q"] ** 1.5, "k:", label="1/Q^1.5"
                    )
                    ax.plot(df_sa["n_q"], 1 / df_sa["n_q"] ** 2, "r:", label="1/Q^2")

                    ax.plot(df_eq["n_q"], df_eq["area_error"], "o--", label="EQ")
                    ax.plot(df_sa["n_q"], df_sa["area_error"], "o--", label="SA")

                    for noise_type in noise_types:
                        df_noise = df_filtered[df_filtered["noise_type"] == noise_type]

                        area_error_mean, area_error_var = [], []
                        for n_q in n_q_list:
                            df_q = df_noise[df_noise["n_q"] == n_q]
                            area_error_mean.append(df_q["area_error"].mean())
                            area_error_var.append(df_q["area_error"].var())

                        ax.errorbar(
                            n_q_list,
                            area_error_mean,
                            yerr=area_error_var,
                            fmt="o-",
                            label=f"noise type: {noise_type}",
                        )

                    ax.set_xlabel("Q")
                    ax.set_ylabel("Area Error")
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.legend()
                    ax.set_title(
                        rf"{privacy_method}, $\epsilon$={epsilon}, #Clients={n_clients}, Strategy: {pr_strategy}"
                    )

                    img_path = Path(
                        f"./img/{dataset}/{classifier}/PR/noise/privacy_{privacy_method}_epsilon_{epsilon}_n_clients_{n_clients}_pr_strategy_{pr_strategy}.pdf"
                    )
                    if not img_path.parent.exists():
                        img_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(img_path)
                    plt.close(fig)
