"""
Recursive multi-step prediction.

The FullRecursivePredictor uses compute_all_features() from dlframework.features
as the single source of truth for feature computation, eliminating the duplicated
inline implementation that was in the prediction notebook.

How to add a new volume strategy:
1. Add an elif branch in FullRecursivePredictor._estimate_volume()
"""

from typing import Optional

import numpy as np
import pandas as pd
import torch

from dlframework.features import compute_all_features


class FullRecursivePredictor:
    """
    Recursive multi-step prediction with full feature recomputation.

    At each step:
    1. Extract feature window from history
    2. Model predicts scaled log return
    3. Descale using last known target_scale
    4. Estimate OHLCV for the predicted day
    5. Append synthetic row to history
    6. Recompute ALL features via compute_all_features()
    7. Repeat
    """

    def __init__(
        self,
        model,
        feature_cols: list,
        device,
        seq_len: int = 20,
        volume_strategy: str = "rolling_mean",
    ):
        self.model = model
        self.feature_cols = feature_cols
        self.device = device
        self.seq_len = seq_len
        self.volume_strategy = volume_strategy
        self.model.eval()

        # State
        self.history_df = None
        self.features_df = None
        self.last_target_scale = None
        self._scale_warned = False

    def initialize(self, historical_ohlcv: pd.DataFrame):
        """
        Initialize with historical OHLCV data.

        Args:
            historical_ohlcv: DataFrame with columns [Close, High, Low, Open, Volume]
                              and DatetimeIndex.
        """
        required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in required_cols:
            if col not in historical_ohlcv.columns:
                raise ValueError(f"Missing required column: {col}")

        self.history_df = historical_ohlcv[required_cols].copy()
        self.history_df = self.history_df.sort_index()
        if len(self.history_df) < self.seq_len + 1:
            raise ValueError(
                f"Need at least seq_len + 1 rows of history, got {len(self.history_df)}"
            )

        self._recompute_features()

        print(f"Initialized with {len(self.history_df)} historical rows")
        print(f"Date range: {self.history_df.index[0]} -> {self.history_df.index[-1]}")

    def _recompute_features(self):
        """Recompute all features from current history."""
        self.features_df = compute_all_features(self.history_df, drop_na=False)

    def _get_latest_target_scale(self) -> float:
        """
        Last-known target scale.

        Priority:
        1) latest finite scale from recomputed features
        2) previously cached scale
        3) fallback 1.0
        """
        if self.features_df is not None and 'target_scale' in self.features_df.columns:
            scales = self.features_df['target_scale'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(scales) > 0:
                scale = float(scales.iloc[-1])
                self.last_target_scale = scale
                return scale

        if self.last_target_scale is not None and np.isfinite(self.last_target_scale):
            return float(self.last_target_scale)

        if not self._scale_warned:
            print("Warning: target_scale unavailable, fallback to 1.0")
            self._scale_warned = True

        self.last_target_scale = 1.0
        return 1.0

    def _estimate_high_low(self, predicted_close: float) -> tuple:
        """Estimate High/Low from predicted Close using recent volatility."""
        recent = self.history_df.tail(20)
        range_ratio = ((recent['High'] - recent['Low']) / recent['Close']).mean()

        half_range = predicted_close * range_ratio / 2
        estimated_high = predicted_close + half_range
        estimated_low = predicted_close - half_range

        return estimated_high, estimated_low

    def _estimate_volume(self) -> float:
        """Estimate volume for the predicted day."""
        if self.volume_strategy == "repeat_last":
            return self.history_df['Volume'].iloc[-1]
        elif self.volume_strategy == "rolling_mean":
            return self.history_df['Volume'].tail(20).mean()
        elif self.volume_strategy == "zero_change":
            return self.history_df['Volume'].iloc[-1]
        else:
            return self.history_df['Volume'].tail(20).mean()

    def _get_next_date(self) -> pd.Timestamp:
        """Get the next calendar day."""
        last_date = self.history_df.index[-1]
        return last_date + pd.Timedelta(days=1)

    def get_current_features(self) -> np.ndarray:
        """Get the current feature sequence matching MultiCoinDataset logic."""
        if self.features_df is None:
            raise ValueError("Predictor not initialized. Call initialize() first.")

        current_date = self.history_df.index[-1]
        feature_rows = self.features_df.dropna(subset=self.feature_cols)
        feature_rows = feature_rows.loc[feature_rows.index < current_date]
        if len(feature_rows) < self.seq_len:
            raise ValueError(
                f"Not enough feature rows to build a sequence of length {self.seq_len}"
            )

        features = feature_rows[self.feature_cols].tail(self.seq_len).values.astype(np.float32)
        return features

    def predict_next(self) -> dict:
        """
        Predict next day.

        Returns:
            dict with scaled prediction, unscaled prediction, and scale used.
        """
        features = self.get_current_features()

        if np.any(np.isnan(features)):
            features = np.nan_to_num(features, nan=0.0)

        with torch.no_grad():
            X = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            pred_scaled = float(self.model(X).cpu().numpy().flatten()[0])

        target_scale = self._get_latest_target_scale()
        pred_unscaled = pred_scaled * target_scale

        return {
            'pred_scaled': pred_scaled,
            'pred_unscaled': pred_unscaled,
            'target_scale': target_scale,
        }

    def step(
        self,
        predicted_log_return_unscaled: float,
        predicted_log_return_scaled: Optional[float] = None,
        target_scale_used: Optional[float] = None,
    ) -> dict:
        """
        Advance simulation by one day.

        Args:
            predicted_log_return_unscaled: descaled model output (raw log return)
            predicted_log_return_scaled: optional scaled output for logging
            target_scale_used: optional applied scale for logging

        Returns:
            dict with predicted OHLCV values and return fields
        """
        current_close = self.history_df['Close'].iloc[-1]
        predicted_close = current_close * np.exp(predicted_log_return_unscaled)

        estimated_high, estimated_low = self._estimate_high_low(predicted_close)
        estimated_volume = self._estimate_volume()
        estimated_open = current_close

        next_date = self._get_next_date()
        new_row = pd.DataFrame(
            {
                'Close': [predicted_close],
                'High': [estimated_high],
                'Low': [estimated_low],
                'Open': [estimated_open],
                'Volume': [estimated_volume],
            },
            index=[next_date],
        )

        self.history_df = pd.concat([self.history_df, new_row])
        self._recompute_features()

        return {
            'date': next_date,
            'close': predicted_close,
            'high': estimated_high,
            'low': estimated_low,
            'open': estimated_open,
            'volume': estimated_volume,
            'log_return': predicted_log_return_unscaled,
            'log_return_unscaled': predicted_log_return_unscaled,
            'log_return_scaled': predicted_log_return_scaled,
            'target_scale_used': target_scale_used,
        }

    def run_recursive_prediction(self, n_days: int, verbose: bool = True) -> pd.DataFrame:
        """Run full recursive prediction for n_days."""
        results = []

        for i in range(n_days):
            pred = self.predict_next()
            step_result = self.step(
                predicted_log_return_unscaled=pred['pred_unscaled'],
                predicted_log_return_scaled=pred['pred_scaled'],
                target_scale_used=pred['target_scale'],
            )
            results.append(step_result)

            if verbose and (i + 1) % 5 == 0:
                print(f"  Day {i+1}/{n_days}: predicted close = ${step_result['close']:,.2f}")

        return pd.DataFrame(results)


@torch.no_grad()
def predict_non_recursive(model, df, feature_cols, target_col, device, seq_len):
    """
    Predict using actual features (non-recursive baseline).

    Returns:
        (dates, pred_scaled, actual_scaled, pred_unscaled, actual_unscaled, scales)
    """
    model.eval()
    df = df.dropna(subset=feature_cols + [target_col, 'target_scale']).copy()
    if len(df) <= seq_len:
        raise ValueError(f"Need more than {seq_len} rows to build inference sequences")

    X_seq = []
    actual_scaled = []
    scales = []
    prediction_dates = []
    for idx in range(len(df) - seq_len):
        X_seq.append(df[feature_cols].iloc[idx : idx + seq_len].values.astype(np.float32))
        actual_scaled.append(float(df[target_col].iloc[idx + seq_len]))
        scales.append(float(df['target_scale'].iloc[idx + seq_len]))
        prediction_dates.append(df.index[idx + seq_len] + pd.Timedelta(days=1))

    X = torch.tensor(np.stack(X_seq), dtype=torch.float32).to(device)
    pred_scaled = model(X).cpu().numpy().flatten()
    actual_scaled = np.asarray(actual_scaled, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)

    pred_unscaled = pred_scaled * scales
    actual_unscaled = actual_scaled * scales

    return (
        pd.Index(prediction_dates),
        pred_scaled,
        actual_scaled,
        pred_unscaled,
        actual_unscaled,
        scales,
    )
