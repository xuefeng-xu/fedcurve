# FedCurve

This code implements a method for federated computation of ROC and PR curves with differential privacy.

## Usage

Run with customize parameters:

```bash
python main.py --curve ROC --dataset adult --privacy DDP --epsilon 1
```

Results are saved under `./result/fedcurve/{dataset}_{classifier}_{curve}.txt`

## Parameters

| Parameter | Description | Values |
|---|---|---|
| `curve` | Curve name | `ROC` or `PR` |
| `dataset` | Dataset name | [`adult`](https://archive.ics.uci.edu/dataset/2/adult), [`bank`](https://archive.ics.uci.edu/dataset/222/bank+marketing), [`cover`](https://archive.ics.uci.edu/dataset/31/covertype), [`sep`](https://www.kaggle.com/competitions/tabular-playground-series-sep-2021), [`oct`](https://www.kaggle.com/competitions/tabular-playground-series-oct-2021), [`nov`](https://www.kaggle.com/competitions/tabular-playground-series-nov-2021) |
| `classifier` | Classifier name | `XGBClassifier`, `LogisticRegression`, etc. |
| `n_clients` | Number of clients | Integer (e.g., `1`, `10`) |
| `privacy` | Privacy model | `EQ`, `SA` or `DDP` |
| `epsilon` | Privacy budget | Float (e.g., `1.0`, `0.5`) |
| `noise_type` | Type of DP noise | `discrete` or `continuous` |
| `post_processing` | Apply post-processing for DP | `true` or `false` |
| `n_q` | Number of quantiles | Integer (e.g., `10`, `30`) |
| `pr_strategy` | PR curve quantile strategy | `separate` or `combine` |
| `branch` | Tree branching factor | Integer (e.g., `2`, `4`) |
| `interp` | Interpolation method | `linear` or `pchip` |
| `n_reps` | Number of repetitions | Integer (e.g., `1`, `3`) |
