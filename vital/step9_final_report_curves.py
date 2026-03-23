import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

import torch
import torch.nn as nn

# optional joblib for XGB
try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False


# =========================
# PATHS
# =========================
X_PATH = "X_seq_24h.npy"
Y_PATH = "y_seq_24h.npy"
G_PATH = "patient_seq_24h.npy"

GRU_CKPT = "gru_24h_model.pt"
CNN_CKPT = "cnn1d_seq24h_model.pt"

# this is the XGB you trained before on tabular hourly_panel
XGB_MODEL = "xgb_deterioration_model.joblib"

OUT_ROC_PNG = "final_roc_curves.png"
OUT_PR_PNG = "final_pr_curves.png"


# =========================
# Utils
# =========================
def best_f1_threshold(y_true, y_prob):
    thr_grid = np.linspace(0.01, 0.99, 99)
    best_thr, best_f1, best_p, best_r = 0.5, -1.0, 0.0, 0.0
    for t in thr_grid:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1:
            best_f1, best_thr, best_p, best_r = f1, t, p, r
    return best_thr, best_f1, best_p, best_r


def plot_roc(curves, out_path):
    plt.figure(figsize=(7, 6))
    for name, y_true, y_prob in curves:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Test)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_pr(curves, out_path):
    plt.figure(figsize=(7, 6))
    for name, y_true, y_prob in curves:
        p, r, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(r, p, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curves (Test)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# =========================
# XGB eval on sequences: use ONLY last hour features
# =========================
def eval_xgb_last_hour(X_te, y_te, model_path):
    """
    X_te: (N, 24, F) sequence features.
    We evaluate XGB on the LAST hour only: X_last = X_te[:, -1, :]
    """
    model = joblib.load(model_path)
    X_last = X_te[:, -1, :]  # (N, F)

    # predict_proba for binary classifier -> [:,1]
    y_prob = model.predict_proba(X_last)[:, 1]
    roc = roc_auc_score(y_te, y_prob)
    pr = average_precision_score(y_te, y_prob)
    thr, f1, p, r = best_f1_threshold(y_te, y_prob)
    return y_prob, roc, pr, thr, f1, p, r


# =========================
# GRU Model (matches step7)
# =========================
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)      # (B,T,H)
        last = out[:, -1, :]      # (B,H)
        return self.head(last).squeeze(1)


def eval_gru(X_te, y_te, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = GRUModel(input_dim=X_te.shape[2]).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    X = torch.from_numpy(X_te).float()  # (N,T,F)
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = X[i:i+256].to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())

    y_prob = np.concatenate(probs)
    roc = roc_auc_score(y_te, y_prob)
    pr = average_precision_score(y_te, y_prob)
    thr, f1, p, r = best_f1_threshold(y_te, y_prob)
    return y_prob, roc, pr, thr, f1, p, r


# =========================
# CNN Model (must match checkpoint)
# =========================
class CNN1DSeq(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            # MUST be kernel=5 pad=2 to match net.0.weight shape [64,33,5]
            nn.Conv1d(in_ch, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),          # 24 -> 12

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),          # 12 -> 6

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)   # (B,256,1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return self.head(x).squeeze(1)


def eval_cnn(X_te, y_te, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = CNN1DSeq(in_ch=X_te.shape[2]).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # X_te: (N,T,F) -> (N,C,T)
    X = torch.from_numpy(X_te.transpose(0, 2, 1)).float()

    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = X[i:i+256].to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())

    y_prob = np.concatenate(probs)
    roc = roc_auc_score(y_te, y_prob)
    pr = average_precision_score(y_te, y_prob)
    thr, f1, p, r = best_f1_threshold(y_te, y_prob)
    return y_prob, roc, pr, thr, f1, p, r


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    X = np.load(X_PATH).astype(np.float32)  # (N,24,F)
    y = np.load(Y_PATH).astype(np.int64)
    g = np.load(G_PATH).astype(np.int64)

    print("Loaded X:", X.shape, "y:", y.shape)
    print("Label counts:", np.bincount(y))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=g))

    X_te = X[te_idx]
    y_te = y[te_idx]
    print("Test label counts:", np.bincount(y_te))

    curves = []

    # GRU
    if os.path.exists(GRU_CKPT):
        prob, roc, pr, thr, f1, p, r = eval_gru(X_te, y_te, GRU_CKPT, device)
        print("\n=== GRU 24h ===")
        print("ROC-AUC:", roc)
        print("PR-AUC :", pr)
        print("Best threshold:", thr)
        print("Best F1:", f1, "| Precision:", p, "| Recall:", r)
        print("\nClassification report @ best threshold:\n")
        print(classification_report(y_te, (prob >= thr).astype(int)))
        curves.append(("GRU", y_te, prob))
    else:
        print("\n[WARN] Missing:", GRU_CKPT)

    # CNN
    if os.path.exists(CNN_CKPT):
        prob, roc, pr, thr, f1, p, r = eval_cnn(X_te, y_te, CNN_CKPT, device)
        print("\n=== CNN1D 24h ===")
        print("ROC-AUC:", roc)
        print("PR-AUC :", pr)
        print("Best threshold:", thr)
        print("Best F1:", f1, "| Precision:", p, "| Recall:", r)
        print("\nClassification report @ best threshold:\n")
        print(classification_report(y_te, (prob >= thr).astype(int)))
        curves.append(("CNN1D", y_te, prob))
    else:
        print("\n[WARN] Missing:", CNN_CKPT)

    # XGB (last hour baseline)
    if HAS_JOBLIB and os.path.exists(XGB_MODEL):
        try:
            prob, roc, pr, thr, f1, p, r = eval_xgb_last_hour(X_te, y_te, XGB_MODEL)
            print("\n=== XGBoost (last hour baseline) ===")
            print("ROC-AUC:", roc)
            print("PR-AUC :", pr)
            print("Best threshold:", thr)
            print("Best F1:", f1, "| Precision:", p, "| Recall:", r)
            print("\nClassification report @ best threshold:\n")
            print(classification_report(y_te, (prob >= thr).astype(int)))
            curves.append(("XGB(last-hour)", y_te, prob))
        except Exception as e:
            print("\n[WARN] XGB eval failed:", e)
    else:
        if not HAS_JOBLIB:
            print("\n[WARN] joblib not installed, skipping XGB")
        elif not os.path.exists(XGB_MODEL):
            print("\n[WARN] Missing:", XGB_MODEL)

    if len(curves) == 0:
        print("\nNo models evaluated. Nothing to plot.")
        return

    plot_roc(curves, OUT_ROC_PNG)
    plot_pr(curves, OUT_PR_PNG)
    print("\nSaved:")
    print(" -", OUT_ROC_PNG)
    print(" -", OUT_PR_PNG)


if __name__ == "__main__":
    main()

