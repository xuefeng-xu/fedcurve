import subprocess
import itertools

datasets = ["nov", "oct", "sep", "cover", "bank", "adult"]
classifiers = ["LogisticRegression", "XGBClassifier"]
n_clients_list = [1]
# privacy_methods = ["DDP", "LDP"]
privacy_methods = ["DDP"]
epsilons = [0.1, 0.3, 1]
noise_types = ["continuous", "discrete"]
n_q_list = [5, 7, 10, 14, 19, 26, 37, 51, 72, 100]
pr_strategies = ["separate", "combine"]


def run_experiment(
    dataset, classifier, n_clients, privacy, epsilon, noise_type, n_q, pr_strategy
):
    n_reps = 1 if privacy == "SA" else 10

    cmd = [
        "python",
        "main.py",
        "--curve",
        "PR",
        "--dataset",
        dataset,
        "--classifier",
        classifier,
        "--n_clients",
        str(n_clients),
        "--privacy",
        privacy,
        "--epsilon",
        str(epsilon),
        "--noise_type",
        noise_type,
        "--interp",
        "pchip",
        "--n_q",
        str(n_q),
        "--pr_strategy",
        pr_strategy,
        "--n_reps",
        str(n_reps),
        "--post_processing",
        "True",
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # run SA
    # for params in itertools.product(
    #     datasets, classifiers, [2], ["SA"], [1], ["continuous"], n_q_list, pr_strategies
    # ):
    #     run_experiment(*params)
    # run DDP
    for params in itertools.product(
        datasets,
        classifiers,
        n_clients_list,
        privacy_methods,
        epsilons,
        noise_types,
        n_q_list,
        pr_strategies,
    ):
        run_experiment(*params)
