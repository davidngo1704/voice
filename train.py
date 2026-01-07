import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import WakeWordNet
from dataset import load_data

# ===== CONFIG =====
BATCH_SIZE = 16
EPOCHS = 60
LR = 1e-3
POS_WEIGHT = 3.0   # positive quan trọng gấp mấy lần negative
DEVICE = "cpu"

# ===== LOAD DATA =====
X, y = load_data()   # X: (N, 100, 13), y: (N,)

# ===== MFCC NORMALIZATION (BẮT BUỘC) =====
mean = X.mean(dim=(0, 1), keepdim=True)
std = X.std(dim=(0, 1), keepdim=True) + 1e-6
X = (X - mean) / std

# ===== DATASET =====
dataset = TensorDataset(X, y)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

# ===== MODEL =====
model = WakeWordNet().to(DEVICE)

# ===== LOSS (CHUẨN) =====
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE)
)

optimizer = optim.Adam(model.parameters(), lr=LR)

# ===== TRAIN LOOP =====
print("Start training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        logits = model(xb)          # (B,)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    # quick sanity metric
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        pos_mean = probs[yb == 1].mean().item() if (yb == 1).any() else 0.0
        neg_mean = probs[yb == 0].mean().item() if (yb == 0).any() else 0.0

    print(
        f"Epoch {epoch+1:03d} | "
        f"loss={avg_loss:.4f} | "
        f"pos≈{pos_mean:.3f} neg≈{neg_mean:.3f}"
    )

# ===== SAVE =====

torch.save({
    "model": model.state_dict(),
    "mean": mean,
    "std": std
}, "wakeword.pt")

print("Model saved -> wakeword.pt")
