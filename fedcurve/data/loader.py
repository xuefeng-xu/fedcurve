import zipfile
import gzip
import shutil
import subprocess
from pandas import read_csv
from pathlib import Path
from urllib.request import urlretrieve
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


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


def load_kaggle(dataset_dir, tag):
    file = dataset_dir / "train.csv"

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        zip_file = file.parent / f"tabular-playground-series-{tag}-2021.zip"

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

        extract(zip_file, file.parent)

    return read_csv(file)


def load_sep(dataset_dir):
    X = load_kaggle(dataset_dir, "sep")

    X.pop("id")
    y = X.pop("claim")

    X = SimpleImputer().fit_transform(X)
    X = StandardScaler().fit_transform(X)

    return X, y


def load_oct(dataset_dir):
    X = load_kaggle(dataset_dir, "oct")

    X.pop("id")
    y = X.pop("target")

    return X, y


def load_nov(dataset_dir):
    X = load_kaggle(dataset_dir, "nov")

    X.pop("id")
    y = X.pop("target")

    return X, y


def load_data(dataset):
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    dataset_dir = PROJECT_ROOT / f"dataset/{dataset}"

    datasets = {
        "adult": lambda: load_adult(dataset_dir),
        "bank": lambda: load_bank(dataset_dir),
        "cover": lambda: load_cover(dataset_dir),
        "sep": lambda: load_sep(dataset_dir),
        "oct": lambda: load_oct(dataset_dir),
        "nov": lambda: load_nov(dataset_dir),
    }

    if dataset not in datasets:
        raise ValueError(f"Unknown dataset: {dataset}")

    return datasets[dataset]()
