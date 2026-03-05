"""
dlframework — micro-framework for crypto time-series prediction.

Extracts shared logic (models, features, losses, training, prediction)
from the project's Jupyter notebooks into importable Python modules.

Quick start in a notebook:
    from dlframework import (
        register_features, compute_all_features,
        build_model, MODEL_REGISTRY,
        LOSS_REGISTRY,
        MultiCoinDataset, load_splits,
        run_trial, evaluate, evaluate_per_ticker,
        FullRecursivePredictor, predict_non_recursive,
        FEATURE_COLS, TARGET_COL,
    )

On Google Colab, make sure the repo root is on sys.path:
    import sys; sys.path.insert(0, '/content/your-repo')
"""

from dlframework.constants import *  # noqa: F401,F403
from dlframework.features import register_features, compute_all_features, entropy_of_window
from dlframework.models import (
    GRU_Basic,
    LSTM_Basic,
    TransformerModel,
    LearnedPositionalEncoding,
    build_model,
    MODEL_REGISTRY,
)
from dlframework.losses import (
    LOSS_REGISTRY,
    HuberDirectionalLoss,
    MapeLoss,
    QuantileLoss,
    LogCoshLoss,
    ScaledMSELoss,
)
from dlframework.dataset import MultiCoinDataset, load_splits
from dlframework.training import (
    train_one_epoch,
    evaluate,
    evaluate_per_ticker,
    run_trial,
    directional_accuracy,
)
from dlframework.prediction import FullRecursivePredictor, predict_non_recursive
