# Federated ROC & PR Curves

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
  - [sep](https://www.kaggle.com/competitions/tabular-playground-series-sep-2021/rules)
  - [oct](https://www.kaggle.com/competitions/tabular-playground-series-oct-2021/rules)
  - [nov](https://www.kaggle.com/competitions/tabular-playground-series-nov-2021/rules)

## Usage

Run with customize parameters:

```bash
python main.py --curve ROC --dataset adult --privacy DDP --epsilon 1
```

Results are saved under `./result/{dataset}_{classifier}_{curve}.txt`

## Parameters

| Parameter | Description | Values |
|---|---|---|
| `curve` | Curve name | `ROC`, `PR` |
| `dataset` | Dataset name | [`adult`](https://archive.ics.uci.edu/dataset/2/adult), [`bank`](https://archive.ics.uci.edu/dataset/222/bank+marketing), [`cover`](https://archive.ics.uci.edu/dataset/31/covertype), [`sep`](https://www.kaggle.com/competitions/tabular-playground-series-sep-2021), [`oct`](https://www.kaggle.com/competitions/tabular-playground-series-oct-2021), [`nov`](https://www.kaggle.com/competitions/tabular-playground-series-nov-2021) |
| `classifier` | Classifier name | `XGBClassifier`, `LogisticRegression`, etc. |
| `n_clients` | Number of clients | Integer (e.g., `1`, `10`) |
| `privacy` | Privacy model | `EQ`, `SA`, `DDP`, `LDP` |
| `epsilon` | Privacy budget | Float (e.g., `1.0`, `0.5`) |
| `noise_type` | Type of DP noise | `discrete`, `continuous` |
| `post_processing` | Apply post-processing for DP | Bool: `true` or `false` |
| `n_q` | Number of quantiles | Integer (e.g., `10`, `30`) |
| `pr_strategy` | PR curve quantile strategy | `separate`, `combine` |
| `height` | Tree height | `auto` or integer (e.g., `5`, `8`) |
| `branch` | Tree branching factor | Integer (e.g., `2`, `4`) |
| `interp` | Interpolation method | `linear`, `pchip` |
| `n_reps` | Number of repetitions | Integer (e.g., `1`, `3`) |

## Reproduction

To reproduce results for the `adult` dataset with `XGBClassifier`:

```bash
python reproduce.py --dataset adult --classifier XGBClassifier
```

Plots are saved under `./img/*/{dataset}_{classifier}_{curve}.pdf`
