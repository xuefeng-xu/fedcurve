import zipfile
import gzip
import shutil
import subprocess
import pandas as pd
from pathlib import Path
from urllib.request import urlretrieve
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def load_adult():
    file = Path("./dataset/adult/adult.data")
    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)

        zip_file = file.parent / "adult.zip"
        if not zip_file.exists():
            try:
                urlretrieve(
                    "https://archive.ics.uci.edu/static/public/2/adult.zip", zip_file
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download dataset: {e}") from e

        try:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(file.parent)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Failed to unzip file {zip_file}: {e}") from e

    X = pd.read_csv(file, header=None)

    y = X.pop(14)
    y = y.map({" >50K": 1, " <=50K": 0})

    objcol = X.select_dtypes(exclude=["float", "int"]).columns

    encoder = OrdinalEncoder()
    X[objcol] = encoder.fit_transform(X[objcol])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def load_bank():
    file = Path("./dataset/bank/bank-full.csv")
    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)

        zip_file = file.parent / "bank+marketing.zip"
        if not zip_file.exists():
            try:
                urlretrieve(
                    "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
                    zip_file,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download dataset: {e}") from e

        try:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(file.parent)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Failed to unzip file {zip_file}: {e}") from e

        zip_sub_file = file.parent / "bank.zip"
        try:
            with zipfile.ZipFile(zip_sub_file, "r") as zip_ref:
                zip_ref.extractall(file.parent)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Failed to unzip file {zip_sub_file}: {e}") from e

    X = pd.read_csv(file, sep=";")

    y = X.pop("y")
    y = y.map({"yes": 1, "no": 0})

    objcol = X.select_dtypes(exclude=["float", "int"]).columns

    encoder = OrdinalEncoder()
    X[objcol] = encoder.fit_transform(X[objcol])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def load_cover():
    file = Path("./dataset/cover/covtype.data")
    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)

        zip_file = file.parent / "covertype.zip"
        if not zip_file.exists():
            try:
                urlretrieve(
                    "https://archive.ics.uci.edu/static/public/31/covertype.zip",
                    zip_file,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download dataset: {e}") from e

        try:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(file.parent)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Failed to unzip file {zip_file}: {e}") from e

        gz_file = file.parent / "covtype.data.gz"
        try:
            with gzip.open(gz_file, "rb") as f_in:
                with open(file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            raise RuntimeError(f"Failed to unzip file {gz_file}: {e}") from e

    X = pd.read_csv(file, header=None)

    y = X.pop(54)
    y = y.apply((lambda x: 1 if x == 1 else 0))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def load_kaggle(tag):
    file = Path(f"./dataset/{tag}/train.csv")
    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)

        zip_file = file.parent / f"tabular-playground-series-{tag}-2021.zip"
        if not zip_file.exists():
            cmd = [
                "kaggle",
                "competitions",
                "download",
                "-c",
                f"tabular-playground-series-{tag}-2021",
                "-p",
                file.parent,
            ]
            try:
                result = subprocess.run(cmd, check=True)
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to download dataset: {e.stderr}") from e

        try:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(file.parent)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Failed to unzip file {zip_file}: {e}") from e

    return pd.read_csv(file)


def load_sep():
    X = load_kaggle("sep")

    X.pop("id")
    y = X.pop("claim")

    imputer = SimpleImputer()
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def load_oct():
    X = load_kaggle("oct")

    X.pop("id")
    y = X.pop("target")

    return X, y


def load_nov():
    X = load_kaggle("nov")

    X.pop("id")
    y = X.pop("target")

    return X, y


def load_data(dataset):
    if dataset == "adult":
        X, y = load_adult()
    elif dataset == "bank":
        X, y = load_bank()
    elif dataset == "cover":
        X, y = load_cover()
    elif dataset == "sep":
        X, y = load_sep()
    elif dataset == "oct":
        X, y = load_oct()
    elif dataset == "nov":
        X, y = load_nov()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return X, y
