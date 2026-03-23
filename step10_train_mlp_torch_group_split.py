import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


DATA_PATH = "train_multicase_3sig.csv"
LABEL_COL = "label_next_30min"
GROUP_COL = "caseid"


class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),  # logits
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def main():
    df = pd.read_csv(DATA_PATH)

    feature_cols = [c for c in df.columns if c not in [LABEL_COL, GROUP_COL, "sec"]]
    X = df[feature_cols].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)  # for BCEWithLogitsLoss
    groups = df[GROUP_COL].values

    print("Rows:", len(df))
    print("Features:", feature_cols)
    print("Label counts:", np.bincount(df[LABEL_COL].values.astype(int)))

    # Group split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Normalize (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # Torch setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)

    model = MLP(in_dim=X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    patience = 8
    patience_left = patience

    for epoch in range(1, 51):
        # Train
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        # Eval AUC
        model.eval()
        all_probs = []
        all_y = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_y.append(yb.numpy())

        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_y).astype(int)
        auc = roc_auc_score(y_true, y_prob)

        print(f"Epoch {epoch:02d} | loss={np.mean(train_losses):.4f} | test_auc={auc:.4f}")

        # Early stopping on AUC
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {
                "model": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "feature_cols": feature_cols,
            }
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    # Restore best + final report
    model.load_state_dict(best_state["model"])
    model.eval()

    with torch.no_grad():
        xb = torch.from_numpy(X_test).to(device)
        probs = torch.sigmoid(model(xb)).cpu().numpy()

    final_auc = roc_auc_score(y_true, probs)
    print("\nBest/Final Test ROC-AUC:", final_auc)

    y_pred = (probs >= 0.5).astype(int)
    print("\nClassification report:\n", classification_report(y_true, y_pred))

    # Save
    torch.save(best_state, "mlp_multisignal_features_torch.pt")
    print("\nSaved: mlp_multisignal_features_torch.pt")


if __name__ == "__main__":
    main()
