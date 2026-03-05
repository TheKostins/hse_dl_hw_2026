"""
Feature engineering pipeline.

All feature definitions live here as the single source of truth.
Both the feature generation notebook and the recursive prediction use
these same definitions.

How to add a new feature:
1. Add a new inner function inside register_features() and call ds.register()
2. If it should be standardized, add its column(s) to standardization_in_columns
   in constants.py and its feature name to standardization_deps
3. If it should be used for training, add the z-scored column name to your
   notebook's FEATURE_COLS list
"""

import numpy as np
import pandas as pd

from dlframework.constants import (
    EPS,
    return_horizons,
    log_close_return_columns,
    log_low_return_columns,
    log_high_return_columns,
    log_open_return_columns,
    log_return_columns,
    vol_windows,
    vol_columns,
    dev_windows,
    dev_columns,
    range_columns,
    close_position_columns,
    log_volume_columns,
    volume_return_columns,
    volume_z20_columns,
    rsi_columns,
    macd_columns,
    roc10_columns,
    entropy_columns,
    skew_columns,
    kurt_columns,
    bull_regime_columns,
    high_vol_regime_columns,
    macd_fast,
    macd_slow,
    macd_signal_w,
    entropy_window,
    standardization_window,
    standardization_deps,
    standardization_in_columns,
    standardization_out_columns,
)


def entropy_of_window(x: np.ndarray) -> float:
    """Compute Shannon entropy of a rolling window of returns."""
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    hist, _ = np.histogram(x, bins=10, density=True)
    p = hist.astype(np.float64)
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    p = p / p.sum()
    return float(-(p * np.log(p + EPS)).sum())


def register_features(ds) -> None:
    """
    Register all feature functions on an UberDatasetFuhrer instance.

    This is the SINGLE SOURCE OF TRUTH for feature definitions.
    Used by:
      - Feature generation notebook: ds = UberDatasetFuhrer(df); register_features(ds); ds.build_all()
      - Recursive prediction: via compute_all_features()

    Args:
        ds: UberDatasetFuhrer instance with a DataFrame containing OHLCV columns.
    """

    # 1. Log returns
    def ft_log_returns(df):
        for h, name in zip(return_horizons, log_close_return_columns):
            df[name] = np.log(df['Close'] / df['Close'].shift(h) + EPS)
        for h, name in zip(return_horizons, log_low_return_columns):
            df[name] = np.log(df['Low'] / df['Low'].shift(h) + EPS)
        for h, name in zip(return_horizons, log_high_return_columns):
            df[name] = np.log(df['High'] / df['High'].shift(h) + EPS)
        for h, name in zip(return_horizons, log_open_return_columns):
            df[name] = np.log(df['Open'] / df['Open'].shift(h) + EPS)

    ds.register(ft_log_returns, name='log_returns', produces=log_return_columns)

    # 2. Volatility (depends on log_returns for log_close_return_1)
    def ft_volatility(df):
        for w, name in zip(vol_windows, vol_columns):
            df[name] = df['log_close_return_1'].rolling(w).std()

    ds.register(ft_volatility, name='volatility', deps=['log_returns'], produces=vol_columns)

    # 3. Trend deviation
    def ft_trend_deviation(df):
        for w, name in zip(dev_windows, dev_columns):
            ma = df['Close'].rolling(w).mean()
            df[name] = (df['Close'] - ma) / (ma + EPS)

    ds.register(ft_trend_deviation, name='trend_deviation', produces=dev_columns)

    # 4. Candle structure
    def ft_range(df):
        df['range'] = (df['High'] - df['Low']) / (df['Close'] + EPS)

    ds.register(ft_range, name='range', produces=range_columns)

    def ft_close_position(df):
        denom = (df['High'] - df['Low']).replace(0, np.nan)
        df['close_position'] = ((df['Close'] - df['Low']) / denom).clip(0.0, 1.0)

    ds.register(ft_close_position, name='close_position', produces=close_position_columns)

    # 5. Volume features
    def ft_log_volume(df):
        df['log_volume'] = np.log(df['Volume'] + 1.0)

    ds.register(ft_log_volume, name='log_volume', produces=log_volume_columns)

    def ft_volume_return(df):
        df['volume_return'] = np.log((df['Volume'] + 1.0) / (df['Volume'].shift(1) + 1.0))

    ds.register(ft_volume_return, name='volume_return', produces=volume_return_columns)

    def ft_volume_z20(df):
        rm = df['Volume'].rolling(20).mean()
        rs = df['Volume'].rolling(20).std().replace(0, np.nan) + EPS
        df['volume_z20'] = (df['Volume'] - rm) / rs

    ds.register(ft_volume_z20, name='volume_z20', produces=volume_z20_columns)

    # 6. Momentum
    def ft_rsi_14(df):
        delta = df['Close'].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + EPS)
        df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))

    ds.register(ft_rsi_14, name='rsi_14', produces=rsi_columns)

    def ft_macd_hist(df):
        ema_f = df['Close'].ewm(span=macd_fast, adjust=False).mean()
        ema_s = df['Close'].ewm(span=macd_slow, adjust=False).mean()
        macd = ema_f - ema_s
        sig = macd.ewm(span=macd_signal_w, adjust=False).mean()
        df['macd_hist'] = macd - sig

    ds.register(ft_macd_hist, name='macd_hist', produces=macd_columns)

    def ft_roc_10(df):
        df['roc_10'] = df['Close'].pct_change(10)

    ds.register(ft_roc_10, name='roc_10', produces=roc10_columns)

    # 7. Distribution-based
    def ft_entropy(df):
        df['entropy'] = (
            df['log_close_return_1']
            .rolling(entropy_window)
            .apply(lambda s: entropy_of_window(s.to_numpy()), raw=False)
        )

    ds.register(ft_entropy, name='entropy', deps=['log_returns'], produces=entropy_columns)

    def ft_skew(df):
        df['skew'] = df['entropy'].rolling(entropy_window).skew()

    ds.register(ft_skew, name='skew', deps=['entropy'], produces=skew_columns)

    def ft_kurt(df):
        df['kurt'] = df['entropy'].rolling(entropy_window).kurt()

    ds.register(ft_kurt, name='kurt', deps=['entropy'], produces=kurt_columns)

    # 8. Regime flags
    def ft_bull_regime(df):
        ma50 = df['Close'].rolling(50).mean()
        ma200 = df['Close'].rolling(200).mean()
        df['bull_regime'] = (ma50 > ma200).astype(int)

    ds.register(ft_bull_regime, name='bull_regime', produces=bull_regime_columns)

    def ft_high_vol_regime(df):
        sigma = df['vol_20']
        df['high_vol_regime'] = (sigma > sigma.rolling(200).mean()).astype(int)

    ds.register(
        ft_high_vol_regime,
        name='high_vol_regime',
        deps=['volatility'],
        produces=high_vol_regime_columns,
    )

    # 9. Target
    def ft_target(df):
        df['target'] = np.log(df['Close'].shift(-1) / df['Close'] + EPS)

    ds.register(ft_target, name='target', produces=['target'])

    # 10. Rolling standardization
    def ft_standardization(df):
        for in_col, out_col in zip(standardization_in_columns, standardization_out_columns):
            rm = df[in_col].rolling(standardization_window).mean()
            rs = df[in_col].rolling(standardization_window).std().replace(0, np.nan)
            df[out_col] = (df[in_col] - rm) / (rs + EPS)

    ds.register(
        ft_standardization,
        name='standardization',
        deps=standardization_deps,
        produces=standardization_out_columns,
    )

    # 11. Scale target by rolling std
    def ft_target_scaling(df):
        target_std = df['target'].rolling(standardization_window).std().replace(0, np.nan)
        df['target'] = df['target'] / (target_std + EPS)
        df['target_scale'] = target_std

    ds.register(
        ft_target_scaling,
        name='target_scaling',
        deps=['target', 'standardization'],
        produces=['target', 'target_scale'],
    )


def compute_all_features(df: pd.DataFrame, drop_na: bool = False) -> pd.DataFrame:
    """
    Compute all features from a raw OHLCV DataFrame.

    This replaces the duplicated compute_all_features() that was in the
    prediction notebook. Under the hood it uses UberDatasetFuhrer with the
    shared register_features() pipeline.

    Args:
        df: DataFrame with columns [Close, High, Low, Open, Volume].
        drop_na: If True, drop NaN rows after building. Set to False for
                 recursive prediction where the tail row may have NaN features.

    Returns:
        DataFrame with all features computed.
    """
    from uberdataset import UberDatasetFuhrer

    ds = UberDatasetFuhrer(df.copy(), feature_fn_prefix='ft_')
    register_features(ds)
    ds.build_all(drop_na=drop_na)
    return ds.df
