# CLAUDE.md

## Project Overview

Deep Learning homework assignments for the HSE Deep Learning course.
Author: Konstantin Anisimov

## Repository Structure

```
hw01/               - Image augmentation homework (torchvision)
hw03 root           - Cryptocurrency price prediction (time series)
  data/             - train/val/test feature CSVs
  checkpoints/      - saved model weights (.pt / .pth)
  experiments/      - exploratory notebooks
  prediction_results/ - CSV outputs from inference
  wandb/            - W&B run logs
  wandb_exports/    - exported W&B history/summary CSVs
  uberdataset.py    - custom feature-engineering orchestration class
```

## Key Source Files

- `uberdataset.py` — `UberDatasetFuhrer`: dependency-aware feature registry wrapping a Pandas DataFrame. Features are registered with `@udf.feature(deps=[...], produces=[...])` and built in topological order via `udf.build(...)`.
- `feture_engineeering.ipynb` — feature construction pipeline
- `model_training.ipynb` / `model_training_2.ipynb` — training loops
- `multi_coin_features.ipynb` / `multi_coin_features_1.ipynb` — multi-asset feature engineering
- `full_recursive_prediction.ipynb` — recursive multi-step forecasting

## Models

| Checkpoint | Description |
|---|---|
| `MEDIUM_GRU.pt` | Medium-capacity GRU |
| `MEDIUM_GRU_TRUNC.pt` | BPTT-truncated variant |
| `MEGA_GRU.pt` | Larger GRU |
| `Grumbert.pt` | GRU + Transformer hybrid |

## Tech Stack

- Python (3.14 based on `__pycache__`)
- PyTorch (GRU models)
- torchvision (hw01 augmentations)
- Pandas / NumPy
- W&B (`wandb`) for experiment tracking
- Optuna for hyperparameter search (see `experiments/hw03_with_optuna.ipynb`)

## Data

- Target assets: BTC-USD, ETH-USD
- Splits: `train_features.csv`, `val_features.csv`, `test_features.csv`
- `make_splits()` on `UberDatasetFuhrer` assigns deterministic labels (seed=42, 80/10/10 default)

## Common Patterns

### Using UberDatasetFuhrer

```python
from uberdataset import UberDatasetFuhrer

udf = UberDatasetFuhrer(df, feature_fn_prefix="f_")

@udf.feature(deps=["price"], produces=["returns"])
def f_returns(df):
    df["returns"] = df["price"].pct_change()

udf.build("returns")   # builds deps automatically
udf.build_all()        # builds everything registered
udf.make_splits()      # adds _split column
```

### Running notebooks

Open Jupyter Lab / Notebook in the repo root. Notebooks assume the working directory is the repo root for relative data paths (`./data/...`).
