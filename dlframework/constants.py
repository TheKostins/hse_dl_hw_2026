"""
Shared constants for the deep-learning framework.

Every magic number and column-name list that appears in both the
feature-generation and training notebooks is defined here exactly once.
"""

# ── tiny numerical guard ────────────────────────────────────────────
EPS = 1e-8

# ── return horizons & column names ──────────────────────────────────
return_horizons = (1, 3, 5, 10, 20)

log_close_return_columns = [f"log_close_return_{h}" for h in return_horizons]
log_low_return_columns   = [f"log_low_return_{h}"   for h in return_horizons]
log_high_return_columns  = [f"log_high_return_{h}"  for h in return_horizons]
log_open_return_columns  = [f"log_open_return_{h}"  for h in return_horizons]

log_return_columns = (
    log_close_return_columns
    + log_low_return_columns
    + log_high_return_columns
    + log_open_return_columns
)

# ── volatility ──────────────────────────────────────────────────────
vol_windows = (5, 10, 20, 60)
vol_columns = [f"vol_{w}" for w in vol_windows]

# ── trend deviation ─────────────────────────────────────────────────
dev_windows = (10, 20, 50, 100)
dev_columns = [f"dev_{w}" for w in dev_windows]

# ── single-column feature groups ────────────────────────────────────
range_columns           = ["range"]
close_position_columns  = ["close_position"]
log_volume_columns      = ["log_volume"]
volume_return_columns   = ["volume_return"]
volume_z20_columns      = ["volume_z20"]
rsi_columns             = ["rsi_14"]
macd_columns            = ["macd_hist"]
roc10_columns           = ["roc_10"]
entropy_columns         = ["entropy"]
skew_columns            = ["skew"]
kurt_columns            = ["kurt"]
bull_regime_columns     = ["bull_regime"]
high_vol_regime_columns = ["high_vol_regime"]

# ── indicator hyper-parameters ──────────────────────────────────────
macd_fast, macd_slow, macd_signal_w = 12, 26, 9
entropy_window = 20
standardization_window = 100

# ── standardization pipeline ────────────────────────────────────────
standardization_deps = [
    "log_returns",
    "volatility",
    "trend_deviation",
    "range",
    "close_position",
    "log_volume",
    "volume_return",
    "rsi_14",
    "macd_hist",
    "roc_10",
    "entropy",
    "skew",
    "kurt",
    "volume_z20",
]

standardization_in_columns = (
    log_return_columns
    + vol_columns
    + dev_columns
    + range_columns
    + close_position_columns
    + log_volume_columns
    + volume_return_columns
    + rsi_columns
    + macd_columns
    + roc10_columns
    + entropy_columns
    + skew_columns
    + kurt_columns
    + volume_z20_columns
)

standardization_out_columns = [f"z_{c}" for c in standardization_in_columns]

# ── training constants ──────────────────────────────────────────────
FEATURE_COLS = [
    "z_log_close_return_1",
    "z_log_close_return_3",
    "z_log_close_return_5",
    "z_log_close_return_10",
    "z_log_close_return_20",
    "z_log_high_return_1",
    "z_log_low_return_1",
    "z_log_open_return_1",
    "z_log_volume",
    "z_volume_z20",
]

TARGET_COL = "target"

# Alias: the default 10-feature training set
DEFAULT_FEATURE_COLS = FEATURE_COLS
