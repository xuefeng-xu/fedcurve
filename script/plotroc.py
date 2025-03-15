import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


if __name__ == "__main__":
    dataset = "nov"
    # classifier = "LogisticRegression"
    classifier = "XGBClassifier"

    file = f"./result/{dataset}/{classifier}/ROC.txt"
    df = pd.read_csv(file)

    n_clients_list = [100, 300, 1000, 3000]
    privacy_methods = ["DDP", "LDP"]
    epsilons = [0.1, 0.3, 1]
    noise_types = ["continuous"]
    n_q_list = [5, 7, 10, 14, 19, 26, 37, 51, 72, 100]

    df_eq = df[df["privacy"] == "EQ"]
    df_sa = df[df["privacy"] == "SA"]

    # compare number of clients
    for privacy_method in privacy_methods:
        for epsilon in epsilons:
            for noise_type in noise_types:
                df_filtered = df[
                    (df["privacy"] == privacy_method)
                    & (df["epsilon"] == epsilon)
                    & (df["noise_type"] == noise_type)
                ]

                fig, ax = plt.subplots()
                ax.plot(df_sa["n_q"], 1 / df_sa["n_q"], "y:", label="1/Q")
                ax.plot(df_sa["n_q"], 1 / df_sa["n_q"] ** 1.5, "k:", label="1/Q^1.5")
                ax.plot(df_sa["n_q"], 1 / df_sa["n_q"] ** 2, "r:", label="1/Q^2")

                ax.plot(df_eq["n_q"], df_eq["area_error"], "o--", label="EQ")
                ax.plot(df_sa["n_q"], df_sa["area_error"], "o--", label="SA")

                for n_clients in n_clients_list:
                    df_n_clients = df_filtered[df_filtered["n_clients"] == n_clients]

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
                        label=f"#Clients={n_clients}",
                    )

                ax.set_xlabel("Q")
                ax.set_ylabel("Area Error")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.legend()
                ax.set_title(
                    rf"{privacy_method}, $\epsilon$={epsilon}, Noise Type: {noise_type}"
                )

                img_path = Path(
                    f"./img/{dataset}/{classifier}/ROC/n_clients/privacy_{privacy_method}_epsilon_{epsilon}_noise_type_{noise_type}.pdf"
                )
                if not img_path.parent.exists():
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(img_path)
                plt.close(fig)

    # compare epsilon
    for n_clients in n_clients_list:
        for privacy_method in privacy_methods:
            for noise_type in noise_types:
                df_filtered = df[
                    (df["n_clients"] == n_clients)
                    & (df["privacy"] == privacy_method)
                    & (df["noise_type"] == noise_type)
                ]

                fig, ax = plt.subplots()
                ax.plot(df_sa["n_q"], 1 / df_sa["n_q"], "y:", label="1/Q")
                ax.plot(df_sa["n_q"], 1 / df_sa["n_q"] ** 1.5, "k:", label="1/Q^1.5")
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
                    rf"{privacy_method}, #Clients={n_clients}, Noise Type: {noise_type}"
                )

                img_path = Path(
                    f"./img/{dataset}/{classifier}/ROC/epsilon/n_clients_{n_clients}_privacy_{privacy_method}_noise_type_{noise_type}.pdf"
                )
                if not img_path.parent.exists():
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(img_path)
                plt.close(fig)
