# DP-ECDF

This code reproduces the methods from the paper "[Differentially Private Empirical Cumulative Distribution Functions](https://arxiv.org/pdf/2502.06651v1)" by Antoine Barczewski, Amal Mawass, and Jan Ramon.

## Usage

Run with customize parameters:

```bash
python main.py --curve ROC --dataset adult --epsilon 1 --norm 1
```

Results are saved under `./result/dpecdf/{dataset}_{classifier}_{curve}.txt`

## Parameters

| Parameter | Description | Values |
|---|---|---|
| `curve` | Curve name | `ROC` or `PR` |
| `dataset` | Dataset name | [`adult`](https://archive.ics.uci.edu/dataset/2/adult), [`bank`](https://archive.ics.uci.edu/dataset/222/bank+marketing), [`cover`](https://archive.ics.uci.edu/dataset/31/covertype), [`sep`](https://www.kaggle.com/competitions/tabular-playground-series-sep-2021), [`oct`](https://www.kaggle.com/competitions/tabular-playground-series-oct-2021), [`nov`](https://www.kaggle.com/competitions/tabular-playground-series-nov-2021) |
| `classifier` | Classifier name | `XGBClassifier`, `LogisticRegression`, etc. |
| `epsilon` | Privacy budget | Float (e.g., `1.0`, `0.5`) |
| `norm` | Norm for smoothing | `1` or `2` |
| `n_reps` | Number of repetitions | Integer (e.g., `1`, `3`) |

## Disclaimer

This implementation is adapted from the authors' [GitLab repository](https://gitlab.inria.fr/abarczew/ab_technical/-/tree/b957b869ab8024e003cd0fc69a356309429dd7be/medical-statistics-and-privacy/dp-cum/code/2024). We have modified the code for compatibility with our library to facilitate simulation and testing. However, our implementation may not be identical to the original, and the authors may have updated their code since our last review.

Last checked: April 19th, 2025.
