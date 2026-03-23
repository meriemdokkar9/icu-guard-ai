import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


X_PATH = "X_raw_3sig.npy"
Y_PATH = "y_raw_3sig.npy"
G_PATH = "caseid_raw_3sig.npy"


class CNN1D(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, C, T)
        h = self.backbone(x)
        a = self.gap(h)
        m = self.gmp(h)
        h = torch.cat([a, m], dim=1)  # (B, 256, 1)
        return self.head(h).squeeze(1)


def zscore_per_channel(X_train, X_test):
    # X: (N,T,C)
    mu = X_train.reshape(-1, X_train.shape[2]).mean(axis=0)
    sd = X_train.reshape(-1, X_train.shape[2]).std(axis=0) + 1e-8
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


def main():
    print("START CNN TRAINING...")

    X = np.load(X_PATH)  # (N,T,3)
    y = np.load(Y_PATH).astype(np.int64)
    g = np.load(G_PATH).astype(np.int64)

    print("Loaded:", X.shape, y.shape, "labels:", np.bincount(y))

    # ---- Add diff channels (trend) -> (N,T,6)
    dX = np.diff(X, axis=1, prepend=X[:, :1, :])
    X = np.concatenate([X, dX], axis=2).astype(np.float32)
    print("After diff:", X.shape)

    # Group split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=g))

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # Normalize per channel (train only)
    X_tr, X_te, mu, sd = zscore_per_channel(X_tr, X_te)

    # To torch: (N,C,T)
    X_tr_t = torch.from_numpy(X_tr.transpose(0, 2, 1)).float()
    X_te_t = torch.from_numpy(X_te.transpose(0, 2, 1)).float()
    y_tr_t = torch.from_numpy(y_tr).float()
    y_te_t = torch.from_numpy(y_te).float()

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    test_ds = TensorDataset(X_te_t, y_te_t)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = CNN1D(in_ch=X.shape[2]).to(device)

    # imbalance
    neg = (y_tr == 0).sum()
    pos = (y_tr == 1).sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4, verbose=True)

    best_auc = -1.0
    best_state = None
    patience = 12
    patience_left = patience

    for epoch in range(1, 61):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        # Eval
        model.eval()
        probs_all = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs_all.append(torch.sigmoid(logits).cpu().numpy())

        y_prob = np.concatenate(probs_all)
        auc = roc_auc_score(y_te, y_prob)

        scheduler.step(auc)

        print(f"Epoch {epoch:02d} | loss={np.mean(losses):.4f} | test_auc={auc:.4f}")

        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {
                "model": model.state_dict(),
                "mu": mu,
                "sd": sd,
                "best_auc": float(best_auc),
            }
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    # Final report (best)
    model.load_state_dict(best_state["model"])
    model.eval()

    with torch.no_grad():
        probs_all = []
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
    y_prob = np.concatenate(probs_all)

    final_auc = roc_auc_score(y_te, y_prob)
    print("\nBest/Final Test ROC-AUC:", final_auc)

    y_pred = (y_prob >= 0.5).astype(int)
    print("\nClassification report:\n", classification_report(y_te, y_pred))

    torch.save(best_state, "cnn1d_raw_3sig_torch.pt")
    print("\nSaved: cnn1d_raw_3sig_torch.pt")


if __name__ == "__main__":
    main()

