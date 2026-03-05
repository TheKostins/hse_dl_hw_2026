"""
Training loop, evaluation, and experiment management.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn

from dlframework.losses import LOSS_REGISTRY
from dlframework.models import build_model


def directional_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of predictions with correct sign."""
    return (torch.sign(pred) == torch.sign(target)).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device) -> dict:
    """
    Train one epoch. Returns dict of metrics:
    {loss, dir_acc, mape, quantile_0.5, log_cosh, scaled_mse, mae}
    """
    model.train()
    total_loss, total_dir, total_mape, n = 0.0, 0.0, 0.0, 0
    total_quantile, total_log_cosh, total_scaled_mse = 0.0, 0.0, 0.0
    total_mae = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        batch_size = len(X)
        mare = LOSS_REGISTRY["mape"](pred, y)
        quantile = LOSS_REGISTRY["quantile_0.5"](pred, y)
        log_cosh = LOSS_REGISTRY["log_cosh"](pred, y)
        scaled_mse = LOSS_REGISTRY["scaled_mse"](pred, y)
        mae = LOSS_REGISTRY["mae"](pred, y)

        total_loss += loss.item() * batch_size
        total_dir += directional_accuracy(pred.detach(), y) * batch_size
        total_mape += mare.item() * batch_size
        total_quantile += quantile.item() * batch_size
        total_log_cosh += log_cosh.item() * batch_size
        total_scaled_mse += scaled_mse.item() * batch_size
        total_mae += mae.item() * batch_size
        n += batch_size

    return {
        "loss": total_loss / n,
        "dir_acc": total_dir / n,
        "mape": total_mape / n,
        "quantile_0.5": total_quantile / n,
        "log_cosh": total_log_cosh / n,
        "scaled_mse": total_scaled_mse / n,
        "mae": total_mae / n,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    """Evaluate on a DataLoader. Returns same metrics dict as train_one_epoch."""
    model.eval()
    total_loss, total_dir, total_mape, n = 0.0, 0.0, 0.0, 0
    total_quantile, total_log_cosh, total_scaled_mse = 0.0, 0.0, 0.0
    total_mae = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_loss += criterion(pred, y).item() * len(X)
        total_dir += directional_accuracy(pred, y) * len(X)
        total_mape += LOSS_REGISTRY["mape"](pred, y).item() * len(X)
        total_quantile += LOSS_REGISTRY["quantile_0.5"](pred, y).item() * len(X)
        total_log_cosh += LOSS_REGISTRY["log_cosh"](pred, y).item() * len(X)
        total_scaled_mse += LOSS_REGISTRY["scaled_mse"](pred, y).item() * len(X)
        total_mae += LOSS_REGISTRY["mae"](pred, y).item() * len(X)
        n += len(X)

    return {
        "loss": total_loss / n,
        "dir_acc": total_dir / n,
        "mape": total_mape / n,
        "quantile_0.5": total_quantile / n,
        "log_cosh": total_log_cosh / n,
        "scaled_mse": total_scaled_mse / n,
        "mae": total_mae / n,
    }


@torch.no_grad()
def evaluate_per_ticker(model, test_df, feature_cols, target_col, criterion, device) -> dict:
    """Per-ticker evaluation on raw DataFrame. Returns {ticker: metrics_dict}."""
    results = {}
    model.eval()
    for ticker, group in test_df.groupby("ticker"):
        group = group.dropna(subset=feature_cols + [target_col])
        if len(group) < 10:
            continue
        X = torch.tensor(group[feature_cols].values, dtype=torch.float32).to(device)
        y = torch.tensor(group[target_col].values, dtype=torch.float32).unsqueeze(1).to(device)
        pred = model(X)
        results[ticker] = {
            "loss": criterion(pred, y).item(),
            "dir_acc": directional_accuracy(pred, y),
            "mape": LOSS_REGISTRY["mape"](pred, y).item(),
            "quantile_0.5": LOSS_REGISTRY["quantile_0.5"](pred, y).item(),
            "log_cosh": LOSS_REGISTRY["log_cosh"](pred, y).item(),
            "scaled_mse": LOSS_REGISTRY["scaled_mse"](pred, y).item(),
            "mae": LOSS_REGISTRY["mae"](pred, y).item(),
            "n": len(group),
        }
    return results


def run_trial(
    cfg,
    train_loader,
    val_loader,
    device,
    checkpoint_dir="./checkpoints",
    run_name="trial",
    max_epochs=100,
    patience=10,
    wandb_project=None,
    wandb_entity=None,
    wandb_group="default",
):
    """
    Full training run: build model, train with early stopping, optionally log to W&B.

    Args:
        cfg: config dict (must have 'model_type', 'loss', 'lr', 'weight_decay', etc.)
        train_loader, val_loader: DataLoaders
        device: torch.device
        checkpoint_dir: where to save best model .pt
        run_name: name for W&B run and checkpoint file
        max_epochs: maximum training epochs
        patience: early stopping patience
        wandb_project: W&B project name (None to skip W&B logging)
        wandb_entity: W&B entity/org
        wandb_group: W&B run group

    Returns:
        (best_val_loss, checkpoint_path)
    """
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    input_dim = train_loader.dataset.X.shape[1]
    model = build_model(cfg, input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)
    criterion = LOSS_REGISTRY[cfg["loss"]]

    # W&B init (optional)
    if wandb_project:
        import wandb

        wandb.init(
            project=wandb_project,
            name=run_name,
            config=cfg,
            reinit=True,
            group=wandb_group,
            entity=wandb_entity,
        )

    best_val_loss = math.inf
    best_ckpt = checkpoint_dir / f"{run_name}.pt"
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        # W&B logging
        if wandb_project:
            import wandb

            wandb_stats = {}
            for k, v in train_metrics.items():
                wandb_stats[f"train/{k}"] = v
            for k, v in val_metrics.items():
                wandb_stats[f"val/{k}"] = v
            wandb_stats["lr"] = optimizer.param_groups[0]["lr"]
            wandb_stats["epoch"] = epoch
            wandb.log(wandb_stats)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_ckpt)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    if wandb_project:
        import wandb

        wandb.finish(quiet=True)

    return best_val_loss, str(best_ckpt)
