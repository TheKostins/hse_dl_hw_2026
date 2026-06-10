"""
Time-Series Cross-Validation: Expanding & Sliding Window
=========================================================

Two strategies, same interface — drop-in comparison.

EXPANDING window (train set grows each fold):
  Fold 0:  TRAIN [========]         VAL [===]
  Fold 1:  TRAIN [============]     VAL [===]
  Fold 2:  TRAIN [================] VAL [===]

SLIDING window (train set is fixed-size, slides forward):
  Fold 0:  TRAIN [========]         VAL [===]
  Fold 1:       TRAIN [========]    VAL [===]
  Fold 2:            TRAIN [========] VAL [===]

When to prefer which:
  - Expanding → when more history helps (stable regimes, slow drift)
  - Sliding   → when old data hurts (regime shifts, structural breaks)
  - If sliding wins by a lot, your market regime changed and old data
    is actively confusing the model.

Usage:
  from timeseries_cv import cross_validate, summarize_cv, compare_strategies
  from timeseries_cv import load_fold_model, plot_fold_predictions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# 1.  Fold definitions
# ---------------------------------------------------------------------------



@dataclass
class CVFold:
    """One CV fold with concrete date boundaries."""
    fold_idx:    int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    val_start:   pd.Timestamp
    val_end:     pd.Timestamp
    strategy:    str  # "expanding" or "sliding"


def expanding_window_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    min_train_frac: float = 0.40,
    val_frac: float = 0.10,
    gap_days: int = 0,
) -> list[CVFold]:
    """
    Training window starts at the beginning and GROWS each fold.

    Parameters
    ----------
    min_train_frac : fraction of dates used for the first fold's training.
    val_frac : fraction of dates used as validation in each fold.
    gap_days : embargo gap between train end and val start.
    """
    dates = pd.DatetimeIndex(df.index).sort_values().unique()
    n_dates = len(dates)

    val_size  = max(1, int(n_dates * val_frac))
    min_train = max(1, int(n_dates * min_train_frac))

    first_val_start = min_train + gap_days
    last_val_start  = n_dates - val_size

    if last_val_start <= first_val_start:
        raise ValueError(
            f"Not enough data for {n_folds} expanding folds. "
            f"Need more data or reduce n_folds/min_train_frac."
        )

    val_starts = np.linspace(first_val_start, last_val_start, n_folds, dtype=int)

    folds: list[CVFold] = []
    for i, vs in enumerate(val_starts):
        ve = min(vs + val_size, n_dates) - 1
        te = vs - 1 - gap_days
        folds.append(CVFold(
            fold_idx    = i,
            train_start = dates[0],          # ← always starts at the beginning
            train_end   = dates[te],
            val_start   = dates[vs],
            val_end     = dates[ve],
            strategy    = "expanding",
        ))
    return folds


def sliding_window_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    train_frac: float = 0.40,
    val_frac: float = 0.10,
    gap_days: int = 0,
) -> list[CVFold]:
    """
    Training window has FIXED SIZE and slides forward each fold.

    Parameters
    ----------
    train_frac : fraction of dates used as training in EVERY fold (fixed).
    val_frac : fraction of dates used as validation in each fold.
    gap_days : embargo gap between train end and val start.
    """
    dates = pd.DatetimeIndex(df.index).sort_values().unique()
    n_dates = len(dates)

    train_size = max(1, int(n_dates * train_frac))
    val_size   = max(1, int(n_dates * val_frac))
    window     = train_size + gap_days + val_size

    if window > n_dates:
        raise ValueError(
            f"Not enough data for sliding window. "
            f"train({train_size}) + gap({gap_days}) + val({val_size}) = "
            f"{window} > {n_dates} dates."
        )

    # First fold starts at index 0, last fold's val must end at the last date
    first_start = 0
    last_start  = n_dates - window

    if last_start <= first_start and n_folds > 1:
        raise ValueError(
            f"Not enough room for {n_folds} sliding folds. "
            f"Reduce n_folds or train_frac."
        )

    starts = np.linspace(first_start, last_start, n_folds, dtype=int)

    folds: list[CVFold] = []
    for i, s in enumerate(starts):
        te = s + train_size - 1
        vs = te + 1 + gap_days
        ve = min(vs + val_size - 1, n_dates - 1)
        folds.append(CVFold(
            fold_idx    = i,
            train_start = dates[s],          # ← slides forward each fold
            train_end   = dates[te],
            val_start   = dates[vs],
            val_end     = dates[ve],
            strategy    = "sliding",
        ))
    return folds


def apply_fold(
    df: pd.DataFrame,
    fold: CVFold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice all tickers by the fold's date boundaries."""
    idx = pd.DatetimeIndex(df.index)
    train_mask = (idx >= fold.train_start) & (idx <= fold.train_end)
    val_mask   = (idx >= fold.val_start)   & (idx <= fold.val_end)
    return df.loc[train_mask], df.loc[val_mask]


# ---------------------------------------------------------------------------
# 2.  Unified CV training loop
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_idx:     int
    strategy:     str
    train_loss:   float
    val_loss:     float
    val_mae:      float
    val_dir_acc:  float
    n_train:      int
    n_val:        int
    train_start:  str
    train_end:    str
    val_start:    str
    val_end:      str
    # --- new: for plotting & further use ---
    best_model_state: dict | None           = None   # state_dict of best checkpoint
    val_preds:        np.ndarray | None      = None   # (n_val_samples,) predictions
    val_targets:      np.ndarray | None      = None   # (n_val_samples,) actuals
    val_index:        pd.DatetimeIndex | None = None   # date index aligned to preds/targets


def cross_validate(
    full_df:        pd.DataFrame,
    feature_cols:   list[str],
    model_factory,                      # callable() -> nn.Module
    dataset_cls:    type[Dataset],      # your WindowDataset class
    strategy:       Literal["expanding", "sliding"] = "expanding",
    n_folds:        int   = 5,
    train_frac:     float = 0.40,       # min_train_frac for expanding, fixed for sliding
    val_frac:       float = 0.10,
    gap_days:       int   = 0,
    seq_len:        int   = 20,
    batch_size:     int   = 64,
    lr:             float = 1e-4,
    epochs:         int   = 50,
    patience:       int   = 10,
    huber_delta:    float = 1.0,
    device:         torch.device = torch.device("cpu"),
    verbose:        bool  = True,
    criterion       = None,  # if None, will use DirectionalHuberLoss with huber_delta
    weight_decay:   float = 1e-2,
    warmup_epochs:  int   = 5,
) -> list[FoldResult]:
    """
    Run time-series CV with either expanding or sliding window.

    Parameters
    ----------
    strategy : "expanding" or "sliding".
    train_frac : for expanding, this is the *minimum* train fraction (first fold).
        For sliding, this is the *fixed* train fraction (every fold).
    All other params same as before.
    """
    if strategy == "expanding":
        folds = expanding_window_folds(
            full_df, n_folds=n_folds,
            min_train_frac=train_frac, val_frac=val_frac, gap_days=gap_days,
        )
    elif strategy == "sliding":
        folds = sliding_window_folds(
            full_df, n_folds=n_folds,
            train_frac=train_frac, val_frac=val_frac, gap_days=gap_days,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    results: list[FoldResult] = []

    for fold in folds:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  [{fold.strategy.upper()}] Fold {fold.fold_idx}  |  "
                  f"train {fold.train_start.date()} → {fold.train_end.date()}  |  "
                  f"val {fold.val_start.date()} → {fold.val_end.date()}")
            print(f"{'='*60}")

        train_df, val_df = apply_fold(full_df, fold)

        train_ds = dataset_cls(train_df, feature_cols=feature_cols, seq_len=seq_len)
        val_ds   = dataset_cls(val_df,   feature_cols=feature_cols, seq_len=seq_len)

        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"  ⚠ Fold {fold.fold_idx} skipped (empty dataset).")
            continue

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

        # Fresh model each fold
        model     = model_factory()
        criterion = nn.HuberLoss(delta=huber_delta) if criterion is None else criterion
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )

        best_val_loss = float("inf")
        patience_ctr  = 0
        best_state    = None
        best_metrics  = (float("inf"), float("inf"), float("inf"), 0.0)
        best_preds    = np.array([])
        best_targets  = np.array([])

        for epoch in range(1, epochs + 1):
            # --- train ---
            model.train()
            total_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                preds, _ = model(x)
                loss = criterion(preds, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(y)
            train_loss = total_loss / len(train_ds)

            # --- evaluate ---
            model.eval()
            all_p, all_t = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    p, _ = model(x)
                    all_p.append(p.cpu())
                    all_t.append(y.cpu())
            preds_t   = torch.cat(all_p)
            targets_t = torch.cat(all_t)
            val_loss  = criterion(preds_t, targets_t).item()
            val_mae   = (preds_t - targets_t).abs().mean().item()
            val_dir   = (preds_t.sign() == targets_t.sign()).float().mean().item()

            scheduler.step()

            if verbose:
                print(f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                      f"val_loss={val_loss:.4f}  dir_acc={val_dir:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr  = 0
                best_metrics  = (train_loss, val_loss, val_mae, val_dir)
                best_preds    = preds_t.numpy()
                best_targets  = targets_t.numpy()
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    if verbose:
                        print(f"  Early stop at epoch {epoch}.")
                    break

        tl, vl, vm, vd = best_metrics

        # Date index aligned to predictions: window dataset drops the
        # first seq_len rows as lookback, so targets start at index seq_len.
        val_dates = pd.DatetimeIndex(val_df.index).sort_values().unique()
        val_idx   = val_dates[seq_len:] if len(val_dates) > seq_len else val_dates

        results.append(FoldResult(
            fold_idx    = fold.fold_idx,
            strategy    = fold.strategy,
            train_loss  = tl,
            val_loss    = vl,
            val_mae     = vm,
            val_dir_acc = vd,
            n_train     = len(train_ds),
            n_val       = len(val_ds),
            train_start = str(fold.train_start.date()),
            train_end   = str(fold.train_end.date()),
            val_start   = str(fold.val_start.date()),
            val_end     = str(fold.val_end.date()),
            best_model_state = best_state,
            val_preds        = best_preds,
            val_targets      = best_targets,
            val_index        = val_idx,
        ))

    return results


# ---------------------------------------------------------------------------
# 3.  Summary & comparison helpers
# ---------------------------------------------------------------------------

def summarize_cv(results: list[FoldResult]) -> pd.DataFrame:
    """Pretty-print fold results and return a summary DataFrame."""
    if not results:
        print("No results to summarize.")
        return pd.DataFrame()

    strategy = results[0].strategy
    rows = [{k: v for k, v in vars(r).items()
             if k not in ('model', 'val_preds', 'val_targets', 'val_dates')}
            for r in results]
    df = pd.DataFrame(rows)

    print(f"\n{'='*60}")
    print(f"  {strategy.upper()}-WINDOW CV SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  Fold {r.fold_idx}  |  val_loss={r.val_loss:.4f}  "
              f"dir_acc={r.val_dir_acc:.3f}  |  "
              f"train={r.n_train:,}  val={r.n_val:,}")

    mean_loss = df["val_loss"].mean()
    std_loss  = df["val_loss"].std()
    mean_dir  = df["val_dir_acc"].mean()
    std_dir   = df["val_dir_acc"].std()

    print(f"\n  val_loss  : {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"  dir_acc   : {mean_dir:.3f} ± {std_dir:.3f}")
    print(f"{'='*60}")

    return df


def compare_strategies(
    expanding_results: list[FoldResult],
    sliding_results:   list[FoldResult],
) -> pd.DataFrame:
    """
    Side-by-side comparison of expanding vs sliding window.

    Returns a DataFrame with one row per strategy.
    """
    def _agg(results, name):
        losses = [r.val_loss for r in results]
        accs   = [r.val_dir_acc for r in results]
        return {
            "strategy":        name,
            "val_loss_mean":   np.mean(losses),
            "val_loss_std":    np.std(losses),
            "dir_acc_mean":    np.mean(accs),
            "dir_acc_std":     np.std(accs),
            "n_folds":         len(results),
            "best_fold_acc":   max(accs),
            "worst_fold_acc":  min(accs),
        }

    df = pd.DataFrame([
        _agg(expanding_results, "expanding"),
        _agg(sliding_results,   "sliding"),
    ])

    print(f"\n{'='*60}")
    print("  STRATEGY COMPARISON")
    print(f"{'='*60}")
    for _, row in df.iterrows():
        print(f"  {row['strategy']:10s}  |  "
              f"dir_acc = {row['dir_acc_mean']:.3f} ± {row['dir_acc_std']:.3f}  |  "
              f"val_loss = {row['val_loss_mean']:.4f} ± {row['val_loss_std']:.4f}")

    # Interpretation hint
    exp_acc = df.loc[df.strategy == "expanding", "dir_acc_mean"].values[0]
    sli_acc = df.loc[df.strategy == "sliding",   "dir_acc_mean"].values[0]
    exp_std = df.loc[df.strategy == "expanding", "dir_acc_std"].values[0]
    sli_std = df.loc[df.strategy == "sliding",   "dir_acc_std"].values[0]

    print()
    if sli_acc - exp_acc > 0.005:
        print("  → Sliding wins: older data may be hurting. Consider shorter history.")
    elif exp_acc - sli_acc > 0.005:
        print("  → Expanding wins: model benefits from more history.")
    else:
        print("  → No clear winner: similar performance regardless of window strategy.")

    if sli_std < exp_std - 0.005:
        print("  → Sliding is more stable across folds (lower variance).")
    elif exp_std < sli_std - 0.005:
        print("  → Expanding is more stable across folds (lower variance).")

    print(f"{'='*60}")

    return df

@dataclass
class EvalResult:
    """Metrics from a single train -> eval run."""
    train_loss:   float
    eval_loss:    float
    eval_mae:     float
    eval_dir_acc: float
    n_train:      int
    n_eval:       int
    train_period: str
    eval_period:  str
    ticker:       str          # "ALL" or specific ticker
    epochs_run:   int
    per_ticker:   dict | None  # {ticker: {loss, mae, dir_acc, n}} if multiple 
    # --- for plotting predictions vs actuals ---
    model:         nn.Module | None     = None
    eval_preds:    np.ndarray | None    = None
    eval_targets:  np.ndarray | None    = None
    eval_dates:    np.ndarray | None    = None

def evaluate_holdout(
    train_df:       pd.DataFrame,
    eval_df:        pd.DataFrame,
    feature_cols:   list[str],
    model_factory,                      # callable() -> nn.Module
    dataset_cls:    type[Dataset],
    ticker:         str | list[str] | None = None,
    seq_len:        int   = 20,
    batch_size:     int   = 64,
    lr:             float = 1e-3,
    epochs:         int   = 1000,
    patience:       int   = 20,
    huber_delta:    float = 1.01,
    device:         torch.device = torch.device("cpu"),
    verbose:        bool  = True,
    criterion       = None,
    weight_decay:   float = 2e-2,
    warmup_epochs:  int   = 5,
) -> EvalResult:
    """
    Train on train_df, evaluate on eval_df.  Optionally filter by ticker.

    Parameters
    ----------
    train_df : DataFrame with DatetimeIndex, 'ticker' column, features, and 'target'.
    eval_df  : Same schema.  Held-out evaluation period.
    ticker   : None  -> use all tickers.
               str   -> filter to one ticker (e.g. "BTC-USD").
               list  -> filter to these tickers.
    All other params match cross_validate.

    Returns
    -------
    EvalResult with aggregate and per-ticker metrics.

    Usage
    -----
        from timeseries_cv import evaluate_holdout

        result = evaluate_holdout(
            train_df=cv_df, eval_df=test_df,
            feature_cols=feature_cols,
            model_factory=model_factory,
            dataset_cls=WindowDataset,
            ticker="BTC-USD",       # or None for all, or ["BTC-USD", "ETH-USD"]
            lr=1e-3, device=DEVICE,
        )
    """
    # --- filter by ticker if requested ---
    if ticker is not None:
        tickers = [ticker] if isinstance(ticker, str) else list(ticker)
        train_df = train_df[train_df["ticker"].isin(tickers)]
        eval_df  = eval_df[eval_df["ticker"].isin(tickers)]
        ticker_label = ", ".join(tickers)
    else:
        ticker_label = "ALL"
        tickers = None

    if len(train_df) == 0 or len(eval_df) == 0:
        raise ValueError(
            f"Empty data after ticker filter. "
            f"train={len(train_df)}, eval={len(eval_df)}"
        )

    train_period = f"{train_df.index.min().date()} -> {train_df.index.max().date()}"
    eval_period  = f"{eval_df.index.min().date()} -> {eval_df.index.max().date()}"

    if verbose:
        print(f"\n{'='*60}")
        print(f"  HOLDOUT EVALUATION  |  ticker={ticker_label}")
        print(f"  train {train_period}  ({len(train_df):,} rows)")
        print(f"  eval  {eval_period}  ({len(eval_df):,} rows)")
        print(f"{'='*60}")

    # --- build datasets & loaders ---
    train_ds = dataset_cls(train_df, feature_cols=feature_cols, seq_len=seq_len)
    eval_ds  = dataset_cls(eval_df,  feature_cols=feature_cols, seq_len=seq_len)

    if len(train_ds) == 0 or len(eval_ds) == 0:
        raise ValueError(
            f"Empty dataset after windowing. "
            f"train_ds={len(train_ds)}, eval_ds={len(eval_ds)}"
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    eval_loader  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # --- train ---
    model = model_factory()
    if criterion is None:
        criterion = nn.HuberLoss(delta=huber_delta)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

    best_eval_loss = float("inf")
    patience_ctr   = 0
    best_state     = None
    best_metrics   = (float("inf"), float("inf"), float("inf"), 0.0)
    final_epoch    = 0

    for epoch in range(1, epochs + 1):
        # train step
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds, _ = model(x)
            loss = criterion(preds, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
        train_loss = total_loss / len(train_ds)

        # eval step
        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                p, _ = model(x)
                all_p.append(p.cpu())
                all_t.append(y.cpu())
        preds_t   = torch.cat(all_p)
        targets_t = torch.cat(all_t)
        eval_loss = criterion(preds_t, targets_t).item()
        eval_mae  = (preds_t - targets_t).abs().mean().item()
        eval_dir  = (preds_t.sign() == targets_t.sign()).float().mean().item()

        scheduler.step()

        if verbose:
            print(f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"eval_loss={eval_loss:.4f}  dir_acc={eval_dir:.3f}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr   = 0
            best_metrics   = (train_loss, eval_loss, eval_mae, eval_dir)
            final_epoch    = epoch
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}.")
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- final inference for plotting ---
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            p, _ = model(x)
            all_p.append(p.cpu())
            all_t.append(y.cpu())
    final_preds   = torch.cat(all_p).numpy()
    final_targets = torch.cat(all_t).numpy()

    # Date index: eval_df rows after the lookback window
    eval_dates = eval_df.index[seq_len - 1:] if hasattr(eval_df, 'index') else None
    if eval_dates is not None and len(eval_dates) > len(final_preds):
        eval_dates = eval_dates[:len(final_preds)]

    # --- per-ticker breakdown ---
    per_ticker_metrics = None
    eval_tickers = eval_df["ticker"].unique()
    if len(eval_tickers) > 1:
        per_ticker_metrics = {}
        model.eval()
        for tk in sorted(eval_tickers):
            tk_df = eval_df[eval_df["ticker"] == tk]
            tk_ds = dataset_cls(tk_df, feature_cols=feature_cols, seq_len=seq_len)
            if len(tk_ds) == 0:
                continue
            tk_loader = DataLoader(tk_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            tk_p, tk_t = [], []
            with torch.no_grad():
                for x, y in tk_loader:
                    x, y = x.to(device), y.to(device)
                    p, _ = model(x)
                    tk_p.append(p.cpu())
                    tk_t.append(y.cpu())
            tk_preds   = torch.cat(tk_p)
            tk_targets = torch.cat(tk_t)
            per_ticker_metrics[tk] = {
                "loss":    criterion(tk_preds, tk_targets).item(),
                "mae":     (tk_preds - tk_targets).abs().mean().item(),
                "dir_acc": (tk_preds.sign() == tk_targets.sign()).float().mean().item(),
                "n":       len(tk_ds),
            }

    tl, el, em, ed = best_metrics
    result = EvalResult(
        train_loss   = tl,
        eval_loss    = el,
        eval_mae     = em,
        eval_dir_acc = ed,
        n_train      = len(train_ds),
        n_eval       = len(eval_ds),
        train_period = train_period,
        eval_period  = eval_period,
        ticker       = ticker_label,
        epochs_run   = final_epoch,
        per_ticker   = per_ticker_metrics,
        model        = model,
        eval_preds   = final_preds,
        eval_targets = final_targets,
        eval_dates   = eval_dates.values if hasattr(eval_dates, 'values') else eval_dates,
    )

    # --- print summary ---
    if verbose:
        print(f"\n{'='*60}")
        print(f"  HOLDOUT RESULT  |  ticker={ticker_label}")
        print(f"{'='*60}")
        print(f"  eval_loss : {el:.4f}")
        print(f"  eval_mae  : {em:.4f}")
        print(f"  dir_acc   : {ed:.3f}")
        print(f"  n_train   : {len(train_ds):,}")
        print(f"  n_eval    : {len(eval_ds):,}")
        print(f"  best epoch: {final_epoch}")

        if per_ticker_metrics:
            print(f"\n  {'ticker':12s}  {'loss':>8s}  {'dir_acc':>8s}  {'n':>6s}")
            for tk, m in sorted(per_ticker_metrics.items()):
                print(f"  {tk:12s}  {m['loss']:8.4f}  {m['dir_acc']:8.3f}  {m['n']:6d}")

        # dummy baseline
        dummy_loss = criterion(
            torch.zeros_like(targets_t), targets_t
        ).item()
        dummy_dir = 0.500
        print(f"\n  dummy_loss: {dummy_loss:.4f}  (predict zero)")
        delta_loss = el - dummy_loss
        print(f"  delta loss: {delta_loss:+.4f}  "
              f"({'better' if el < dummy_loss else 'worse'} than dummy)")
        print(f"  delta dir : {ed - dummy_dir:+.3f}")
        print(f"{'='*60}")

    return result


# ---------------------------------------------------------------------------
# 4.  Fold model loading & prediction plotting helpers
# ---------------------------------------------------------------------------

def load_fold_model(
    fold_result: FoldResult,
    model_factory,
) -> nn.Module:
    """
    Reconstruct the best model from a CV fold.

    Usage:
        model = load_fold_model(results[0], model_factory)
        model.eval()
    """
    model = model_factory()
    if fold_result.best_model_state is not None:
        model.load_state_dict(fold_result.best_model_state)
    model.eval()
    return model


def plot_fold_predictions(
    results: list[FoldResult],
    fold_indices: list[int] | None = None,
    title: str = "CV Fold Predictions vs Actuals",
):
    """
    Plot predictions vs actuals for one or more CV folds using Plotly.

    Parameters
    ----------
    results : list of FoldResult from cross_validate.
    fold_indices : which folds to plot (default: all).
    title : plot title.

    Returns
    -------
    plotly Figure (call .show() to display).
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("pip install plotly  to use this helper.")

    if fold_indices is None:
        fold_indices = list(range(len(results)))

    n = len(fold_indices)
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=False,
        subplot_titles=[f"Fold {results[i].fold_idx}  |  dir_acc={results[i].val_dir_acc:.3f}"
                        for i in fold_indices],
        vertical_spacing=0.06,
    )

    for row, fi in enumerate(fold_indices, start=1):
        r = results[fi]
        if r.val_preds is None or r.val_targets is None:
            continue

        x = r.val_index if r.val_index is not None else np.arange(len(r.val_preds))

        fig.add_trace(go.Scatter(
            x=x, y=r.val_targets, mode="lines",
            name=f"Actual (fold {r.fold_idx})",
            line=dict(color="#636EFA"),
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=x, y=r.val_preds, mode="lines",
            name=f"Predicted (fold {r.fold_idx})",
            line=dict(color="#EF553B", dash="dot"),
        ), row=row, col=1)

        fig.add_hline(y=0, line_dash="dash", line_color="grey",
                      opacity=0.4, row=row, col=1)

    fig.update_layout(
        title=title,
        height=300 * n,
        template="plotly_dark",
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# 5.  Ensemble: multi-seed averaging
# ---------------------------------------------------------------------------

@dataclass
class EnsembleFoldResult:
    """Ensemble metrics for a single CV fold."""
    fold_idx:         int
    strategy:         str
    ensemble_dir_acc: float
    ensemble_mae:     float
    per_seed_dir_acc: list[float]
    n_val:            int
    val_start:        str
    val_end:          str
    ensemble_preds:   np.ndarray
    targets:          np.ndarray


@dataclass
class EnsembleCVResult:
    """Full result from ensemble cross-validation."""
    seeds:            list[int]
    strategy:         str
    n_seeds:          int
    n_folds:          int
    fold_results:     list[EnsembleFoldResult]
    mean_dir_acc:     float
    std_dir_acc:      float
    per_seed_mean:    dict          # {seed: mean_dir_acc across folds}
    per_seed_results: dict          # {seed: list[FoldResult]} for full access


def ensemble_cross_validate(
    seeds:          list[int],
    make_model,                     # callable(seed: int) -> nn.Module
    full_df:        pd.DataFrame,
    feature_cols:   list[str],
    dataset_cls:    type[Dataset],
    strategy:       Literal["expanding", "sliding"] = "expanding",
    n_folds:        int   = 5,
    verbose:        bool  = True,
    **cv_kwargs,
) -> EnsembleCVResult:
    """
    Train N models (one per seed) and average their predictions per fold.

    Parameters
    ----------
    seeds : list of random seeds (e.g. [42, 1337, 7777, 2024]).
    make_model : callable(seed) -> nn.Module. Called once per fold per seed
        to create a fresh, seeded model.
    full_df, feature_cols, dataset_cls : same as cross_validate.
    strategy : "expanding" or "sliding".
    **cv_kwargs : all other params forwarded to cross_validate
        (val_frac, seq_len, batch_size, lr, epochs, patience, etc.)

    Returns
    -------
    EnsembleCVResult with ensemble metrics, per-seed breakdown, and
    access to individual FoldResults for further analysis.

    Usage
    -----
        def make_model(seed):
            torch.manual_seed(seed)
            return GRUWithAttention(
                input_size=len(feature_cols), hidden_size=96,
                num_layers=3, linear_hidden=96, dropout=0.3,
                skip_days=10, skip_dropout=0.5,
            ).to(DEVICE)

        ens = ensemble_cross_validate(
            seeds=[42, 1337, 7777, 2024],
            make_model=make_model,
            full_df=cv_df, feature_cols=feature_cols,
            dataset_cls=WindowDataset,
            strategy="expanding",
            n_folds=5, val_frac=0.10, seq_len=60,
            batch_size=128, lr=1e-3, epochs=1000,
            patience=13, huber_delta=1.01, device=DEVICE,
            criterion=criterion, weight_decay=1e-1, warmup_epochs=10,
        )
    """
    all_seed_results: dict[int, list[FoldResult]] = {}

    for si, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'#'*60}")
            print(f"  ENSEMBLE SEED {si+1}/{len(seeds)}  seed={seed}")
            print(f"{'#'*60}")

        # Create a model_factory closure for this seed
        def model_factory(_seed=seed):
            import random as _random
            _random.seed(_seed)
            np.random.seed(_seed)
            torch.manual_seed(_seed)
            torch.cuda.manual_seed_all(_seed)
            return make_model(_seed)

        seed_results = cross_validate(
            full_df=full_df,
            feature_cols=feature_cols,
            model_factory=model_factory,
            dataset_cls=dataset_cls,
            strategy=strategy,
            n_folds=n_folds,
            verbose=verbose,
            **cv_kwargs,
        )
        all_seed_results[seed] = seed_results

    # --- Aggregate predictions across seeds per fold ---
    fold_ensemble: list[EnsembleFoldResult] = []

    for fi in range(n_folds):
        fold_preds_per_seed = []
        fold_targets = None
        per_seed_acc = []

        for seed in seeds:
            r = all_seed_results[seed][fi]
            if r.val_preds is not None:
                fold_preds_per_seed.append(r.val_preds)
                if fold_targets is None:
                    fold_targets = r.val_targets
                per_seed_acc.append(r.val_dir_acc)

        if len(fold_preds_per_seed) == 0 or fold_targets is None:
            continue

        # Average predictions across seeds
        avg_preds = np.mean(fold_preds_per_seed, axis=0)
        ens_dir = float(np.mean(np.sign(avg_preds) == np.sign(fold_targets)))
        ens_mae = float(np.mean(np.abs(avg_preds - fold_targets)))

        ref = all_seed_results[seeds[0]][fi]
        fold_ensemble.append(EnsembleFoldResult(
            fold_idx=fi,
            strategy=strategy,
            ensemble_dir_acc=ens_dir,
            ensemble_mae=ens_mae,
            per_seed_dir_acc=per_seed_acc,
            n_val=ref.n_val,
            val_start=ref.val_start,
            val_end=ref.val_end,
            ensemble_preds=avg_preds,
            targets=fold_targets,
        ))

    ens_accs = [f.ensemble_dir_acc for f in fold_ensemble]
    mean_acc = float(np.mean(ens_accs))
    std_acc  = float(np.std(ens_accs))

    per_seed_mean = {}
    for seed in seeds:
        accs = [r.val_dir_acc for r in all_seed_results[seed]]
        per_seed_mean[seed] = float(np.mean(accs))

    result = EnsembleCVResult(
        seeds=seeds,
        strategy=strategy,
        n_seeds=len(seeds),
        n_folds=n_folds,
        fold_results=fold_ensemble,
        mean_dir_acc=mean_acc,
        std_dir_acc=std_acc,
        per_seed_mean=per_seed_mean,
        per_seed_results=all_seed_results,
    )

    # --- Print summary ---
    if verbose:
        print(f"\n{'='*60}")
        print(f"  ENSEMBLE CV SUMMARY  ({len(seeds)} seeds × {n_folds} folds)")
        print(f"{'='*60}")

        print(f"\n  Per-seed individual averages:")
        for seed in seeds:
            print(f"    seed {seed:>12d}  →  dir_acc = {per_seed_mean[seed]:.3f}")

        print(f"\n  Per-fold ensemble (averaged predictions):")
        for f in fold_ensemble:
            seed_str = "  ".join(f"{a:.3f}" for a in f.per_seed_dir_acc)
            print(f"    Fold {f.fold_idx}  |  ensemble={f.ensemble_dir_acc:.3f}  "
                  f"|  individuals: [{seed_str}]")

        print(f"\n  ENSEMBLE dir_acc : {mean_acc:.3f} ± {std_acc:.3f}")

        # Compare to single-seed average
        single_avg = float(np.mean(list(per_seed_mean.values())))
        delta = mean_acc - single_avg
        print(f"  Single-seed avg  : {single_avg:.3f}")
        print(f"  Ensemble gain    : {delta:+.3f}")
        print(f"{'='*60}")

    return result


@dataclass
class EnsembleEvalResult:
    """Result from ensemble holdout evaluation."""
    seeds:            list[int]
    n_seeds:          int
    ensemble_dir_acc: float
    ensemble_mae:     float
    ensemble_loss:    float | None
    per_seed_dir_acc: list[float]
    per_seed_mae:     list[float]
    ticker:           str
    train_period:     str
    eval_period:      str
    n_train:          int
    n_eval:           int
    ensemble_preds:   np.ndarray
    targets:          np.ndarray
    eval_dates:       np.ndarray | None
    per_seed_results: list[EvalResult]  # full access to individual runs
    per_ticker:       dict | None       # per-ticker ensemble metrics


def ensemble_evaluate_holdout(
    seeds:          list[int],
    make_model,                     # callable(seed: int) -> nn.Module
    train_df:       pd.DataFrame,
    eval_df:        pd.DataFrame,
    feature_cols:   list[str],
    dataset_cls:    type[Dataset],
    verbose:        bool  = True,
    **eval_kwargs,
) -> EnsembleEvalResult:
    """
    Train N models (one per seed) and average their holdout predictions.

    Parameters
    ----------
    seeds : list of random seeds.
    make_model : callable(seed) -> nn.Module.
    train_df, eval_df : training and held-out evaluation DataFrames.
    **eval_kwargs : forwarded to evaluate_holdout
        (ticker, seq_len, batch_size, lr, epochs, patience, etc.)

    Returns
    -------
    EnsembleEvalResult with ensemble and per-seed metrics.

    Usage
    -----
        ens = ensemble_evaluate_holdout(
            seeds=[42, 1337, 7777, 2024],
            make_model=make_model,
            train_df=cv_df, eval_df=test_df,
            feature_cols=feature_cols,
            dataset_cls=WindowDataset,
            ticker=["BTC-USD", "ETH-USD"],
            lr=1e-3, device=DEVICE,
        )
    """
    all_results: list[EvalResult] = []

    for si, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'#'*60}")
            print(f"  ENSEMBLE SEED {si+1}/{len(seeds)}  seed={seed}")
            print(f"{'#'*60}")

        def model_factory(_seed=seed):
            import random as _random
            _random.seed(_seed)
            np.random.seed(_seed)
            torch.manual_seed(_seed)
            torch.cuda.manual_seed_all(_seed)
            return make_model(_seed)

        result = evaluate_holdout(
            train_df=train_df,
            eval_df=eval_df,
            feature_cols=feature_cols,
            model_factory=model_factory,
            dataset_cls=dataset_cls,
            verbose=verbose,
            **eval_kwargs,
        )
        all_results.append(result)

    # --- Aggregate ---
    all_preds = np.stack([r.eval_preds for r in all_results])  # (n_seeds, n_samples)
    avg_preds = all_preds.mean(axis=0)
    targets   = all_results[0].eval_targets

    ens_dir = float(np.mean(np.sign(avg_preds) == np.sign(targets)))
    ens_mae = float(np.mean(np.abs(avg_preds - targets)))

    # Try computing ensemble loss with the criterion if available
    ens_loss = None
    criterion = eval_kwargs.get("criterion")
    if criterion is not None:
        ens_loss = criterion(
            torch.from_numpy(avg_preds), torch.from_numpy(targets)
        ).item()

    per_seed_dir_acc = [r.eval_dir_acc for r in all_results]
    per_seed_mae     = [r.eval_mae for r in all_results]

    # --- Per-ticker ensemble metrics ---
    per_ticker_ens = None
    ticker_kwarg = eval_kwargs.get("ticker")
    if ticker_kwarg is not None:
        tickers = [ticker_kwarg] if isinstance(ticker_kwarg, str) else list(ticker_kwarg)
    else:
        tickers = eval_df["ticker"].unique().tolist()

    if len(tickers) > 1:
        per_ticker_ens = {}
        seq_len = eval_kwargs.get("seq_len", 20)
        batch_size = eval_kwargs.get("batch_size", 64)
        device = eval_kwargs.get("device", torch.device("cpu"))

        for tk in sorted(tickers):
            tk_eval = eval_df[eval_df["ticker"] == tk]
            tk_ds = dataset_cls(tk_eval, feature_cols=feature_cols, seq_len=seq_len)
            if len(tk_ds) == 0:
                continue
            tk_loader = DataLoader(tk_ds, batch_size=batch_size, shuffle=False, num_workers=0)

            tk_all_preds = []
            tk_targets_ref = None
            for r in all_results:
                if r.model is None:
                    continue
                r.model.eval()
                tk_p, tk_t = [], []
                with torch.no_grad():
                    for x, y in tk_loader:
                        x, y = x.to(device), y.to(device)
                        p, _ = r.model(x)
                        tk_p.append(p.cpu())
                        tk_t.append(y.cpu())
                tk_preds = torch.cat(tk_p).numpy()
                tk_all_preds.append(tk_preds)
                if tk_targets_ref is None:
                    tk_targets_ref = torch.cat(tk_t).numpy()

            if len(tk_all_preds) > 0 and tk_targets_ref is not None:
                tk_avg = np.mean(tk_all_preds, axis=0)
                per_ticker_ens[tk] = {
                    "dir_acc": float(np.mean(np.sign(tk_avg) == np.sign(tk_targets_ref))),
                    "mae":     float(np.mean(np.abs(tk_avg - tk_targets_ref))),
                    "n":       len(tk_ds),
                }

    ref = all_results[0]
    ens_result = EnsembleEvalResult(
        seeds=seeds,
        n_seeds=len(seeds),
        ensemble_dir_acc=ens_dir,
        ensemble_mae=ens_mae,
        ensemble_loss=ens_loss,
        per_seed_dir_acc=per_seed_dir_acc,
        per_seed_mae=per_seed_mae,
        ticker=ref.ticker,
        train_period=ref.train_period,
        eval_period=ref.eval_period,
        n_train=ref.n_train,
        n_eval=ref.n_eval,
        ensemble_preds=avg_preds,
        targets=targets,
        eval_dates=ref.eval_dates,
        per_seed_results=all_results,
        per_ticker=per_ticker_ens,
    )

    # --- Print summary ---
    if verbose:
        print(f"\n{'='*60}")
        print(f"  ENSEMBLE HOLDOUT  ({len(seeds)} seeds)")
        print(f"{'='*60}")

        print(f"\n  Per-seed results:")
        for seed, acc, mae in zip(seeds, per_seed_dir_acc, per_seed_mae):
            print(f"    seed {seed:>12d}  →  dir_acc={acc:.3f}  mae={mae:.4f}")

        print(f"\n  ENSEMBLE dir_acc : {ens_dir:.3f}")
        print(f"  ENSEMBLE mae     : {ens_mae:.4f}")
        if ens_loss is not None:
            print(f"  ENSEMBLE loss    : {ens_loss:.4f}")

        single_avg = float(np.mean(per_seed_dir_acc))
        delta = ens_dir - single_avg
        print(f"\n  Single-seed avg  : {single_avg:.3f}")
        print(f"  Ensemble gain    : {delta:+.3f}")

        if per_ticker_ens:
            print(f"\n  {'ticker':12s}  {'dir_acc':>8s}  {'mae':>8s}  {'n':>6s}")
            for tk, m in sorted(per_ticker_ens.items()):
                print(f"  {tk:12s}  {m['dir_acc']:8.3f}  {m['mae']:8.4f}  {m['n']:6d}")

        print(f"{'='*60}")

    return ens_result