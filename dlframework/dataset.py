"""
Dataset and data loading utilities.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class MultiCoinDataset(Dataset):
    """
    Sliding-window dataset for time-series prediction.

    Each sample is (X_seq, y) where:
      X_seq: (seq_len, n_features) float32 tensor
      y: (1,) float32 scalar target
    """

    def __init__(self, df, feature_cols, target_col, seq_len=20):
        df = df.dropna(subset=feature_cols + [target_col])
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        X_seq = self.X[idx : idx + self.seq_len]
        y = self.y[idx + self.seq_len]
        return torch.tensor(X_seq), torch.tensor(y).unsqueeze(0)


def load_splits(data_dir, feature_cols, target_col, batch_size, seq_len=20, device=None):
    """
    Load train/val/test CSVs, build MultiCoinDatasets, return DataLoaders.

    Args:
        data_dir: Path to directory containing {train,val,test}_features.csv
        feature_cols: list of column names for input features
        target_col: column name for the target
        batch_size: batch size for DataLoaders
        seq_len: sequence length for MultiCoinDataset
        device: torch.device (used to configure num_workers and pin_memory)

    Returns:
        (train_loader, val_loader, test_loader, test_df)
    """
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train_features.csv", index_col=0)
    val_df = pd.read_csv(data_dir / "val_features.csv", index_col=0)
    test_df = pd.read_csv(data_dir / "test_features.csv", index_col=0)

    train_ds = MultiCoinDataset(train_df, feature_cols, target_col, seq_len)
    val_ds = MultiCoinDataset(val_df, feature_cols, target_col, seq_len)
    test_ds = MultiCoinDataset(test_df, feature_cols, target_col, seq_len)

    is_cuda = device is not None and device.type == "cuda"
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=4 if is_cuda else 0,
        pin_memory=is_cuda,
        persistent_workers=is_cuda,
    )
    train_loader = DataLoader(train_ds, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, test_df
