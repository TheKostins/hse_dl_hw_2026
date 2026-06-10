"""
models.py — importable architectures, loss, dataset and helpers for the
multi-coin next-day **direction** model (HW3 finalized).

Everything here is extracted verbatim (same forward contracts) from the
exploratory notebooks so that `03_direction_model.ipynb` and the provenance
notebooks in `experiments/` share one source of truth:

- `WindowDataset`            — per-ticker sliding windows (never spans tickers).
- `AdditiveAttention`        — Bahdanau additive attention over time.
- `GRUWithAttention`         — headline direction model; forward -> (pred, attn).
- `VolatilityGRU`            — magnitude (|return|) model, Softplus output.
- `SoftDirectionalHuberLoss` — Huber + differentiable tanh sign-agreement term.
- `CURATED_FEATURES`         — the 8-feature high-signal core set.
- helpers: `get_device`, `directional_accuracy`, `make_loaders`, `z_feature_cols`.

Forward contract (shared by all CV/training code in `timeseries_cv.py`):
    preds, _ = model(x)           # attention map is optional / may be None
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------- #
# Curated high-signal feature set (the "core 8").                              #
# Every model consumes these z-scored / regime columns. They exist in the     #
# saved data/*.csv for all tickers.                                           #
# --------------------------------------------------------------------------- #
CURATED_FEATURES: list[str] = [
    "bull_regime",          # regime / trend  (strongest single predictor)
    "z_bb_pct_b",           # momentum: position within Bollinger band
    "z_volume_z20",         # volume anomaly (orthogonal to price)
    "z_log_close_return_1", # yesterday's return (short-term mean reversion)
    "z_range",              # intraday volatility magnitude
    "z_ret_autocorr",       # trending vs mean-reverting regime
    "dow_cos",              # calendar (weak but orthogonal)
    "high_vol_regime",      # binary volatility regime
]


def z_feature_cols(df: pd.DataFrame) -> list[str]:
    """All z-scored feature columns (the full ~95-feature set)."""
    drop = {"target", "target_scale", "target_mu", "ticker", "_split"}
    return [c for c in df.columns if c.startswith("z_") and c not in drop]


def get_device() -> torch.device:
    return (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )


# --------------------------------------------------------------------------- #
# Dataset — windows are built WITHIN each ticker, never across tickers.        #
# --------------------------------------------------------------------------- #
class WindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        seq_len: int = 20,
        target_col: str = "target",
    ):
        self.seq_len = seq_len
        self.samples: list[tuple[npt.NDArray[np.float32], np.float32]] = []

        for ticker, group in df.groupby("ticker"):
            group = group.sort_index()
            X = group[feature_cols].values.astype(np.float32)
            y = group[target_col].values.astype(np.float32)

            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[mask]
            y = y[mask]

            for i in range(seq_len, len(group)):
                window = X[i - seq_len : i]
                target = y[i]
                self.samples.append((window, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y)


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = 20,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    kw = dict(feature_cols=feature_cols, seq_len=seq_len)
    loader_kw = dict(batch_size=batch_size, num_workers=0)  # num_workers=0 on MPS
    return (
        DataLoader(WindowDataset(train_df, **kw), shuffle=True, **loader_kw),
        DataLoader(WindowDataset(val_df, **kw), shuffle=False, **loader_kw),
        DataLoader(WindowDataset(test_df, **kw), shuffle=False, **loader_kw),
    )


# --------------------------------------------------------------------------- #
# Attention + direction model.                                                #
# --------------------------------------------------------------------------- #
class AdditiveAttention(nn.Module):
    """
    Bahdanau-style attention over the time dimension.

    Score:  e_t = v · tanh(W · h_t)
    Weight: α   = softmax(e)  (over T)
    Output: c   = Σ α_t · h_t
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.v(torch.tanh(self.W(hidden_states)))  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)              # (B, T, 1)
        context = (weights * hidden_states).sum(dim=1)      # (B, H)
        return context, weights.squeeze(-1)


class GRUWithAttention(nn.Module):
    """GRU encoder + additive attention. forward(x) -> (pred, attn_map)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        linear_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AdditiveAttention(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, linear_hidden),
            nn.GELU(),
            nn.LayerNorm(linear_hidden),
            nn.Dropout(dropout),
            nn.Linear(linear_hidden, 1),
        )

    def forward(self, x: torch.Tensor):
        hidden_states, _ = self.gru(x)          # (B, T, H)
        context, attn = self.attention(hidden_states)
        out = self.head(context).squeeze(-1)    # (B,)
        return out, attn


class VolatilityGRU(nn.Module):
    """
    Small GRU predicting next-day |return| (magnitude). Softplus output keeps
    predictions positive. At inference combine as sign(dir) * vol to get a
    signed return estimate. forward(x) -> (pred, None).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 16,
        num_layers: int = 1,
        linear_hidden: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, linear_hidden),
            nn.GELU(),
            nn.LayerNorm(linear_hidden),
            nn.Dropout(dropout),
            nn.Linear(linear_hidden, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor):
        hidden_states, _ = self.gru(x)
        last = hidden_states[:, -1, :]          # last timestep (no attention)
        out = self.head(last).squeeze(-1)
        return out, None


# --------------------------------------------------------------------------- #
# Loss.                                                                        #
# --------------------------------------------------------------------------- #
class SoftDirectionalHuberLoss(nn.Module):
    """Huber + differentiable directional penalty.

    The directional term uses tanh(sharpness·x) as a soft, differentiable
    surrogate for sign(x), so it contributes real gradient (unlike a hard
    sign()). Agreement is positive when preds/targets share a sign.
    """

    def __init__(self, delta: float = 1.01, dir_weight: float = 0.3, sharpness: float = 10.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction="none")
        self.dir_weight = dir_weight
        self.sharpness = sharpness  # higher = closer to hard sign

    def forward(self, preds, targets):
        huber = self.huber(preds, targets).mean()
        agreement = torch.tanh(preds * self.sharpness) * torch.tanh(targets * self.sharpness)
        dir_loss = -agreement.mean()            # minimize -> maximize agreement
        return huber + self.dir_weight * dir_loss


# --------------------------------------------------------------------------- #
# Metric helper.                                                               #
# --------------------------------------------------------------------------- #
def directional_accuracy(preds, targets) -> float:
    """Fraction of predictions with the correct sign vs target."""
    p = np.asarray(preds).reshape(-1)
    t = np.asarray(targets).reshape(-1)
    m = np.isfinite(p) & np.isfinite(t)
    if m.sum() == 0:
        return float("nan")
    return float((np.sign(p[m]) == np.sign(t[m])).mean())


# --------------------------------------------------------------------------- #
# Checkpoint export / import.                                                  #
# A checkpoint is self-describing: it stores the architecture name, the        #
# constructor kwargs, the feature columns and seq_len, plus the weights — so   #
# `load_checkpoint` can rebuild the exact model without re-specifying config.  #
# --------------------------------------------------------------------------- #
ARCH_REGISTRY: dict[str, type[nn.Module]] = {
    "GRUWithAttention": GRUWithAttention,
    "VolatilityGRU": VolatilityGRU,
}


def save_checkpoint(model, path, *, config, feature_cols, seq_len,
                    arch="GRUWithAttention", metrics=None):
    """Save weights + everything needed to rebuild the model.

    `config` is the dict of constructor kwargs (e.g. input_size, hidden_size,
    num_layers, linear_hidden, dropout). Returns the path written.
    """
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": arch,
        "config": dict(config),
        "feature_cols": list(feature_cols),
        "seq_len": int(seq_len),
        "state_dict": model.state_dict(),
        "metrics": dict(metrics or {}),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path, device=None, map_location=None):
    """Rebuild a model from a checkpoint written by `save_checkpoint`.

    Returns (model, payload). `payload` carries feature_cols / seq_len / metrics
    so the caller stays consistent with how the model was trained.
    """
    device = device or get_device()
    ml = map_location or device
    try:
        payload = torch.load(path, map_location=ml, weights_only=False)
    except TypeError:  # older torch without the weights_only kwarg
        payload = torch.load(path, map_location=ml)
    cls = ARCH_REGISTRY[payload["arch"]]
    model = cls(**payload["config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
