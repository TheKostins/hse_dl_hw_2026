"""
Two-Head GRU: Direction + Magnitude
====================================

Architecture:
  GRU backbone → shared representation
    ├─ Direction head (binary classification) → P(up)
    └─ Magnitude head (regression) → E[|return|]

Final prediction = sign_from_direction × magnitude

Loss = α · BCE(direction) + (1-α) · Huber(magnitude)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoHeadGRU(nn.Module):
    """
    Shared GRU backbone with separate direction and magnitude heads.
    
    The direction head predicts P(return > 0) via sigmoid.
    The magnitude head predicts E[|return|] (always positive, via softplus).
    
    Parameters
    ----------
    input_size : number of features
    hidden_size : GRU hidden dimension
    num_layers : GRU depth
    head_hidden : hidden dim for each head's MLP
    dropout : dropout rate for GRU and head MLPs
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        head_hidden: int = 16,
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
        
        self.norm = nn.LayerNorm(hidden_size)
        
        # Direction head: shared_repr → P(up)
        self.dir_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
            # No sigmoid here — use BCEWithLogitsLoss for numerical stability
        )
        
        # Magnitude head: shared_repr → E[|return|]
        self.mag_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
            nn.Softplus(),  # Ensures positive output
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name and 'gru' in name:
                nn.init.zeros_(p)
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)
        
        Returns
        -------
        dir_logit : (batch, 1) — raw logit for P(up), pass through sigmoid for probability
        magnitude : (batch, 1) — predicted |return|, always positive
        combined  : (batch, 1) — signed prediction = sign(dir) * magnitude
        hidden    : final GRU hidden state
        """
        gru_out, hidden = self.gru(x)
        
        # Use last timestep
        last = gru_out[:, -1, :]
        last = self.norm(last)
        
        dir_logit = self.dir_head(last)       # (batch, 1)
        magnitude = self.mag_head(last)        # (batch, 1), always > 0
        
        # Combined prediction: convert logit to sign, multiply by magnitude
        dir_prob = torch.sigmoid(dir_logit)
        sign = 2.0 * dir_prob - 1.0           # maps [0,1] → [-1,1] (soft sign)
        combined = sign * magnitude
        
        return dir_logit, magnitude, combined, hidden


class TwoHeadLoss(nn.Module):
    """
    Combined loss for the two-head model.
    
    L = alpha * BCE(dir_logit, target_sign) + (1 - alpha) * Huber(magnitude, |target|)
    
    Parameters
    ----------
    alpha : weight for direction loss (0.5 = equal weighting)
    huber_delta : delta for Huber loss on magnitude
    """
    
    def __init__(self, alpha: float = 0.5, huber_delta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.huber = nn.HuberLoss(delta=huber_delta)
    
    def forward(self, dir_logit, magnitude, target):
        """
        Parameters
        ----------
        dir_logit : (batch, 1) — raw logit from direction head
        magnitude : (batch, 1) — predicted |return| from magnitude head
        target    : (batch, 1) — actual z-scored return (signed)
        """
        # Direction: binary target (1 if return > 0, else 0)
        target_sign = (target > 0).float()
        dir_loss = self.bce(dir_logit, target_sign)
        
        # Magnitude: predict absolute value
        target_mag = target.abs()
        mag_loss = self.huber(magnitude, target_mag)
        
        combined_loss = self.alpha * dir_loss + (1 - self.alpha) * mag_loss
        
        return combined_loss, dir_loss, mag_loss


# ---------------------------------------------------------------------------
#  Metrics helper
# ---------------------------------------------------------------------------

def two_head_metrics(dir_logit, magnitude, target):
    """Compute metrics for both heads."""
    with torch.no_grad():
        # Direction accuracy
        pred_up = (dir_logit > 0).float()
        actual_up = (target > 0).float()
        dir_acc = (pred_up == actual_up).float().mean().item()
        
        # Direction confidence (how decisive is the model?)
        dir_prob = torch.sigmoid(dir_logit)
        dir_confidence = (dir_prob - 0.5).abs().mean().item() * 2  # 0=random, 1=certain
        
        # Magnitude MAE
        mag_mae = (magnitude - target.abs()).abs().mean().item()
        
        # Combined prediction correlation with target
        sign = (2.0 * dir_prob - 1.0)
        combined = sign * magnitude
        
        # Pearson correlation
        t = target.squeeze()
        c = combined.squeeze()
        if t.std() > 0 and c.std() > 0:
            corr = torch.corrcoef(torch.stack([t, c]))[0, 1].item()
        else:
            corr = 0.0
        
    return {
        'dir_acc': dir_acc,
        'dir_confidence': dir_confidence,
        'mag_mae': mag_mae,
        'pred_target_corr': corr,
    }




# ---------------------------------------------------------------------------
#  Example training loop integration for cross_validate
# ---------------------------------------------------------------------------

EXAMPLE_USAGE = """
# ── Model factory ──────────────────────────────────────────
def model_factory():
    torch.manual_seed(SEED)
    return TwoHeadGRU(
        input_size=len(feature_cols),
        hidden_size=32,       # bump from 16, the two heads need more capacity
        num_layers=2,
        head_hidden=16,
        dropout=0.2,          # lighter than 0.3 — let the model express magnitude
    ).to(DEVICE)


# ── Loss & alpha schedule ─────────────────────────────────
# Start alpha high (focus on direction first), anneal down 
# so magnitude head kicks in once backbone is stable.
#
#   alpha=0.7  →  70% direction, 30% magnitude
#
criterion = TwoHeadLoss(alpha=0.7, huber_delta=1.0)


# ── Inside training loop (replaces single-head forward pass) ──
# TRAIN:
model.train()
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    dir_logit, magnitude, combined, _ = model(x)
    loss, dir_loss, mag_loss = criterion(dir_logit, magnitude, y)
    loss.backward()
    ...

# EVAL:
model.eval()
all_dir, all_mag, all_tgt = [], [], []
with torch.no_grad():
    for x, y in val_loader:
        dir_logit, magnitude, combined, _ = model(x)
        all_dir.append(dir_logit.cpu())
        all_mag.append(magnitude.cpu())
        all_tgt.append(y.cpu())

dir_logits = torch.cat(all_dir)
magnitudes = torch.cat(all_mag)
targets    = torch.cat(all_tgt)

val_loss, val_dir_loss, val_mag_loss = criterion(dir_logits, magnitudes, targets)
metrics = two_head_metrics(dir_logits, magnitudes, targets)

print(f"  dir_acc={metrics['dir_acc']:.3f}  "
      f"dir_conf={metrics['dir_confidence']:.3f}  "
      f"mag_mae={metrics['mag_mae']:.3f}  "
      f"corr={metrics['pred_target_corr']:.3f}")


# ── Key metrics to watch ──────────────────────────────────
#
# 1. dir_acc        — should match or beat your current ~54%
# 2. dir_confidence — if this stays near 0, model is predicting 0.5 for everything
# 3. mag_mae        — compare vs baseline of predicting mean(|target|) 
# 4. pred_target_corr — the ultimate metric: does combined prediction 
#                        correlate with actual returns? This is what matters
#                        for trading.
#
# ── Alpha tuning guide ────────────────────────────────────
#
#   alpha=1.0  → pure direction (ignore magnitude entirely)
#   alpha=0.7  → direction-dominant (good starting point)
#   alpha=0.5  → equal weight 
#   alpha=0.3  → magnitude-dominant (risky: may hurt direction)
#
# Watch for: if dir_acc drops when you lower alpha, the magnitude
# gradients are corrupting the backbone. Raise alpha back up, or
# consider freezing the backbone for a few epochs first.
"""

