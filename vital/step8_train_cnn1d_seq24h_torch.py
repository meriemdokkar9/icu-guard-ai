import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =========================
# PATHS
# =========================
X_PATH = "X_seq_24h.npy"
Y_PATH = "y_seq_24h.npy"
G_PATH = "patient_seq_24h.npy"

# =========================
# CONFIG
# =========================
BATCH_SIZE = 256
EPOCHS = 25
LR = 1e-3
TEST_SIZE = 0.2
RANDOM_STATE = 42


class CNN1D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.net(x)
        x = self.head(x)
        return x.squeeze(1)


def standardize_by_train(X_tr, X_te):
    # X: (N,T,F)
    mu = X_tr.reshape(-1, X_tr.shape[2]).mean(axis=0)
    sd = X_tr.reshape(-1, X_tr.shape[2]).std(axis=0) + 1e-8
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd
    return X_tr, X_te, mu, sd


def find_best_threshold_f1(y_true, y_prob):
    best_thr, best_f1, best_p, best_r = 0.5, -1, 0, 0
    for thr in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= thr).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        if f1 > best_f1:
            best_f1, best_thr, best_p, best_r = f1, thr, p, r
    return best_thr, best_f1, best_p, best_r


def main():
    print("Loading sequences...")
    X = np.load(X_PATH).astype(np.float32)  # (N,24,F)
    y = np.load(Y_PATH).astype(np.int64)
    g = np.load(G_PATH).astype(np.int64)

    print("X:", X.shape, "y:", y.shape, "labels:", np.bincount(y))
    N, T, F = X.shape

    # Group split by patient
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(X, y, groups=g))
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    print("Train:", X_tr.shape, "Test:", X_te.shape)
    print("Train labels:", np.bincount(y_tr), "Test labels:", np.bincount(y_te))

    # Standardize
    X_tr, X_te, mu, sd = standardize_by_train(X_tr, X_te)

    # CNN expects (N,C,T) not (N,T,F)
    X_tr = X_tr.transpose(0, 2, 1)  # (N,F,T)
    X_te = X_te.transpose(0, 2, 1)

    # Torch tensors
    X_tr_t = torch.from_numpy(X_tr).float()
    X_te_t = torch.from_numpy(X_te).float()
    y_tr_t = torch.from_numpy(y_tr).float()
    y_te_t = torch.from_numpy(y_te).float()

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    test_ds = TensorDataset(X_te_t, y_te_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = CNN1D(in_ch=F).to(device)

    # imbalance weight
    neg = (y_tr == 0).sum()
    pos = (y_tr == 1).sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_auc = -1.0
    best_state = None
    patience = 6
    patience_left = patience

    for epoch in range(1, EPOCHS + 1):
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

        model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs.append(torch.sigmoid(logits).cpu().numpy())

        y_prob = np.concatenate(probs)

        auc = roc_auc_score(y_te, y_prob)
        pr_auc = average_precision_score(y_te, y_prob)

        print(f"Epoch {epoch:02d} | loss={np.mean(losses):.4f} | ROC-AUC={auc:.4f} | PR-AUC={pr_auc:.4f}")

        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {
                "model": model.state_dict(),
                "mu": mu,
                "sd": sd,
                "best_auc": best_auc,
                "F": F,
                "T": T,
            }
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    # final eval
    model.load_state_dict(best_state["model"])
    model.eval()

    probs = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())

    y_prob = np.concatenate(probs)

    final_auc = roc_auc_score(y_te, y_prob)
    final_pr = average_precision_score(y_te, y_prob)

    thr, f1, p, r = find_best_threshold_f1(y_te, y_prob)

    print("\nBest/Final ROC-AUC:", final_auc)
    print("Best/Final PR-AUC :", final_pr)
    print("\nBest threshold (max F1):", thr)
    print("Best F1:", f1, "| Precision:", p, "| Recall:", r)

    y_pred = (y_prob >= thr).astype(int)
    print("\nClassification report @ best threshold:\n")
    print(classification_report(y_te, y_pred))

    torch.save(best_state, "cnn1d_seq24h_model.pt")
    print("\nSaved: cnn1d_seq24h_model.pt")


if __name__ == "__main__":
    main()
