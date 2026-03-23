import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


CSV_PATH = "train_multicase_3sig.csv"
LABEL_COL = "label_next_30min"
GROUP_COL = "caseid"


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def main():
    df = pd.read_csv(CSV_PATH)

    feat_cols = [c for c in df.columns if c not in [LABEL_COL, GROUP_COL, "sec"]]

    X = df[feat_cols].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int64)
    groups = df[GROUP_COL].values.astype(np.int64)

    print("Rows:", len(df))
    print("Features:", len(feat_cols))
    print("Label counts:", np.bincount(y))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # Standardization
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # Torch tensors
    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).float()

    X_te_t = torch.from_numpy(X_te).float()
    y_te_t = torch.from_numpy(y_te).float()

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    test_ds = TensorDataset(X_te_t, y_te_t)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = MLP(in_dim=X.shape[1]).to(device)

    # class imbalance
    neg = (y_tr == 0).sum()
    pos = (y_tr == 1).sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_auc = -1
    best_state = None
    patience = 10
    patience_left = patience

    for epoch in range(1, 51):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Eval AUC
        model.eval()
        probs_all = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs_all.append(torch.sigmoid(logits).cpu().numpy())

        y_prob = np.concatenate(probs_all)
        auc = roc_auc_score(y_te, y_prob)

        print(f"Epoch {epoch:02d} | loss={np.mean(losses):.4f} | test_auc={auc:.4f}")

        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = model.state_dict()
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    # Final eval with best
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(X_te_t.to(device))
        y_prob = torch.sigmoid(logits).cpu().numpy()

    auc = roc_auc_score(y_te, y_prob)
    print("\nBest/Final Test ROC-AUC:", auc)

    # Best threshold
    prec, rec, thr = precision_recall_curve(y_te, y_prob)
    f1 = (2 * prec * rec) / (prec + rec + 1e-8)
    best_i = np.argmax(f1)
    best_thr = thr[best_i] if best_i < len(thr) else 0.5

    print("\nBest threshold:", best_thr)
    print("Best F1:", f1[best_i])
    print("Precision:", prec[best_i], "Recall:", rec[best_i])

    y_pred = (y_prob >= best_thr).astype(int)
    print("\nClassification report:\n", classification_report(y_te, y_pred))

    torch.save({"model": model.state_dict(), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_},
               "mlp_features_3sig.pt")
    print("\nSaved: mlp_features_3sig.pt")


if __name__ == "__main__":
    main()
