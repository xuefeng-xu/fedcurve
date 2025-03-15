# Federated ROC & PR Curve

## Installation

1. Create a Python env

```bash
conda create --name fedcurve python=3.10
conda activate fedcurve
```

2. Install the requirements

```bash
pip install -r requirements.txt
```

3. Set up [Kaggle API](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#api-credentials), then accept the competition rules for dataset: [sep](https://www.kaggle.com/competitions/tabular-playground-series-sep-2021/rules), [oct](https://www.kaggle.com/competitions/tabular-playground-series-oct-2021/rules), [nov](https://www.kaggle.com/competitions/tabular-playground-series-nov-2021/rules).


## Usage

```bash
python main.py
```

## Parameters

| Parameter |Value|
|---|---|
|`curve`|Name of curve: "ROC" or "PR"|
|`dataset`|Name of dataset: ["adult"](https://archive.ics.uci.edu/dataset/2/adult), ["bank"](https://archive.ics.uci.edu/dataset/222/bank+marketing), ["cover"](https://archive.ics.uci.edu/dataset/31/covertype), ["sep"](https://www.kaggle.com/competitions/tabular-playground-series-sep-2021), ["oct"](https://www.kaggle.com/competitions/tabular-playground-series-oct-2021), ["nov"](https://www.kaggle.com/competitions/tabular-playground-series-nov-2021)|Name of dataset|
|`n_clients`|Number of clients|
