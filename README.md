# Federated Computation of ROC and PR Curves

## Installation

1. Create a Python environment:

```bash
conda create --name fedcurve python=3.10
conda activate fedcurve
```

2. Clone the repository:

```bash
git clone https://github.com/xuefeng-xu/fedcurve.git
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up the Kaggle API
- Follow the instructions [here](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#api-credentials)
- Accept the competition rules for the following datasets:
  - [sep](https://www.kaggle.com/competitions/tabular-playground-series-sep-2021/rules), [oct](https://www.kaggle.com/competitions/tabular-playground-series-oct-2021/rules), [nov](https://www.kaggle.com/competitions/tabular-playground-series-nov-2021/rules)

## Usage

See [fedcurve/README](./fedcurve/README.md) for details.

## Reproduction

To reproduce results for the `adult` dataset with `XGBClassifier`:

```bash
python reproduce.py --dataset adult --classifier XGBClassifier
```

Plots are saved under `./img/*/{dataset}_{classifier}_{curve}.pdf`
