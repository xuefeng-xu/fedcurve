import zipfile
import gzip
import shutil
import numpy as np
from pandas import read_csv
from pathlib import Path
from urllib.request import urlretrieve
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from imblearn.datasets import fetch_datasets, make_imbalance


def download(url, zip_file):
    try:
        urlretrieve(url, zip_file)
    except Exception as e:
        raise RuntimeError(f"Failed to download from {url}: {e}") from e


def extract(zip_file, extract_path):
    file_type = zip_file.suffix
    try:
        if file_type == ".zip":
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(extract_path)
        elif file_type == ".gz":
            with gzip.open(zip_file, "rb") as f_in, open(extract_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            raise ValueError(f"Unknown file type: {file_type}")

    except Exception as e:
        raise RuntimeError(f"Failed to extract {zip_file}: {e}") from e


def download_and_extract(url, zip_file, extract_path):
    download(url, zip_file)
    extract(zip_file, extract_path)


def load_adult(dataset_dir):
    file = dataset_dir / "adult.data"

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        zip_file = file.parent / "adult.zip"

        download_and_extract(
            "https://archive.ics.uci.edu/static/public/2/adult.zip",
            zip_file,
            file.parent,
        )

    X = read_csv(file, header=None)

    y = X.pop(14).map({" >50K": 1, " <=50K": 0})

    objcol = X.select_dtypes(exclude=["float", "int"]).columns
    X[objcol] = OrdinalEncoder().fit_transform(X[objcol])
    X = StandardScaler().fit_transform(X)

    return X, y


def load_bank(dataset_dir):
    file = dataset_dir / "bank-full.csv"

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        zip_file = file.parent / "bank+marketing.zip"

        download_and_extract(
            "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
            zip_file,
            file.parent,
        )

        zip_sub_file = file.parent / "bank.zip"
        extract(zip_sub_file, file.parent)

    X = read_csv(file, sep=";")

    y = X.pop("y").map({"yes": 1, "no": 0})

    objcol = X.select_dtypes(exclude=["float", "int"]).columns
    X[objcol] = OrdinalEncoder().fit_transform(X[objcol])
    X = StandardScaler().fit_transform(X)

    return X, y


def load_cover(dataset_dir):
    file = dataset_dir / "covtype.data"

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        zip_file = file.parent / "covertype.zip"

        download_and_extract(
            "https://archive.ics.uci.edu/static/public/31/covertype.zip",
            zip_file,
            file.parent,
        )

        gz_file = file.parent / "covtype.data.gz"
        extract(gz_file, file)

    X = read_csv(file, header=None)

    y = X.pop(54).apply((lambda x: 1 if x == 1 else 0))

    X = StandardScaler().fit_transform(X)

    return X, y


def load_dota2(dataset_dir):
    file = dataset_dir / "dota2Train.csv"

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        zip_file = file.parent / "dota2+games+results.zip"

        download_and_extract(
            "https://archive.ics.uci.edu/static/public/367/dota2+games+results.zip",
            zip_file,
            file.parent,
        )

    X = read_csv(file, header=None)

    y = X.pop(0).map({1: 1, -1: 0})

    X = StandardScaler().fit_transform(X)

    return X, y


def load_imblearn_data(dataset_dir, dataset):
    dataobj = fetch_datasets(data_home=dataset_dir)[dataset]

    X = dataobj.data

    y = dataobj.target
    y[y == -1] = 0

    X = StandardScaler().fit_transform(X)

    return X, y


def load_data(dataset, ratio=np.nan, rng=None):
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    dataset_dir = PROJECT_ROOT / f"dataset/{dataset}"

    uci_data = {
        "adult": lambda: load_adult(dataset_dir),
        "bank": lambda: load_bank(dataset_dir),
        "cover": lambda: load_cover(dataset_dir),
        "dota2": lambda: load_dota2(dataset_dir),
    }

    if dataset in uci_data:
        X, y = uci_data[dataset]()
    else:
        X, y = load_imblearn_data(dataset_dir.parent.stem, dataset)

    if not np.isnan(ratio):
        n_neg = sum(y == 0)
        n_pos = round(n_neg * ratio)
        if n_pos < 1:
            raise ValueError(f"Ratio {ratio} is too small for the dataset {dataset}")
        X, y = make_imbalance(X, y, sampling_strategy={1: n_pos}, random_state=rng)

    return X, y
