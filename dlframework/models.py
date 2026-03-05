"""
Model architectures and registry.

How to add a new model:
1. Define your nn.Module class in this file
2. Add a config entry to MODEL_REGISTRY with a unique name
3. Add an elif branch in build_model()
"""

import torch
import torch.nn as nn


class GRU_Basic(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        linear_hiddien_size: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout_2d = nn.Dropout2d(dropout)
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, linear_hiddien_size)
        self.norm = nn.LayerNorm(linear_hiddien_size)
        self.bias = nn.Parameter(torch.ones(linear_hiddien_size))
        self.fc2 = nn.Linear(linear_hiddien_size, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.dropout_2d(x.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.norm(x)
        x = x + self.bias
        return self.fc2(x)


class LSTM_Basic(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        linear_hiddien_size: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout_2d = nn.Dropout2d(dropout)
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, linear_hiddien_size)
        self.norm = nn.LayerNorm(linear_hiddien_size)
        self.fc2 = nn.Linear(linear_hiddien_size, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.dropout_2d(x.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.norm(x)
        return self.fc2(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions)


class TransformerModel(nn.Module):
    """
    Transformer encoder for a single flat feature vector (no time axis).
    Each feature becomes a separate token with learned positional encoding.
    Supports CLS token readout or mean pooling.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.use_cls_token = use_cls_token
        seq_len = num_features + (1 if use_cls_token else 0)

        self.feature_embed = nn.Linear(1, d_model)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = LearnedPositionalEncoding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F = x.shape
        tokens = self.feature_embed(x.unsqueeze(-1))
        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_enc(tokens)
        encoded = self.encoder(tokens)
        readout = encoded[:, 0, :] if self.use_cls_token else encoded.mean(dim=1)
        return self.head(readout)


# ── Pre-configured model configs ──────────────────────────────────────────────
MODEL_REGISTRY = {
    "gru_basic": {
        "model_type": "gru_basic",
        "hidden_size": 64,
        "num_layers": 2,
        "linear_hidden_size": 64,
        "dropout": 0.1,
    },
    "lstm_basic": {
        "model_type": "lstm_basic",
        "hidden_size": 64,
        "num_layers": 2,
        "linear_hidden_size": 64,
        "dropout": 0.1,
    },
    "MEGA_GRU": {
        "model_type": "gru_basic",
        "hidden_size": 1024,
        "num_layers": 3,
        "linear_hidden_size": 512,
        "dropout": 0.1,
        "loss": "huber",
        "lr": 1e-4,
        "weight_decay": 1e-5,
    },
    "MEDIUM_GRU": {
        "model_type": "gru_basic",
        "hidden_size": 256,
        "num_layers": 2,
        "linear_hidden_size": 128,
        "dropout": 0.1,
        "loss": "huber_directional",
        "lr": 5e-4,
        "weight_decay": 1e-5,
    },
    "transformer_v1": {
        "model_type": "transformer",
        "d_model": 2,
        "nhead": 2,
        "num_layers": 4,
        "dim_feedforward": 256,
        "dropout": 0.2,
        "use_cls_token": True,
    },
    "transformer_large": {
        "model_type": "transformer",
        "d_model": 64,
        "nhead": 4,
        "num_layers": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "use_cls_token": True,
    },
}


def build_model(cfg: dict, input_dim: int) -> nn.Module:
    """
    Construct a model from a config dict.

    Args:
        cfg: dict with 'model_type' key and architecture-specific params.
        input_dim: number of input features.

    Returns:
        nn.Module instance (not moved to device).
    """
    model_type = cfg["model_type"]
    if model_type == "gru_basic":
        return GRU_Basic(
            input_size=input_dim,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            linear_hiddien_size=cfg.get("linear_hidden_size", 64),
            dropout=cfg.get("dropout", 0.0),
        )
    elif model_type == "lstm_basic":
        return LSTM_Basic(
            input_size=input_dim,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            linear_hiddien_size=cfg.get("linear_hidden_size", 64),
            dropout=cfg.get("dropout", 0.0),
        )
    elif model_type == "transformer":
        return TransformerModel(
            num_features=input_dim,
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg.get("dropout", 0.1),
            use_cls_token=cfg.get("use_cls_token", True),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
