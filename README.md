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

## Usage

See [fedcurve/README](./fedcurve/README.md) for details.

## Reproduction

To reproduce results for the `adult` dataset with `XGBClassifier`:

```bash
python reproduce.py --dataset adult --classifier XGBClassifier
```

Plots are saved under `./img/*/{dataset}_{classifier}_{curve}.pdf`
