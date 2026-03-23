import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# =========================
# PATHS
# =========================
X_PATH = "X_seq_24h.npy"
Y_PATH = "y_seq_24h.npy"
G_PATH = "patient_seq_24h.npy"
H_PATH = "hour_seq_24h.npy"

ALERTS_CSV = "dashboard_alerts_only.csv"          # input (patients with alert)
CKPT_PATH = "gru_24h_model.pt"                   # GRU checkpoint
OUT_DIR = "timelines_alert_patients"             # output folder
THRESHOLD_DEFAULT = 0.94


# =========================
# GRU Model (must match training)
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
        out, _ = self.gru(x)     # (B,T,H)
        last = out[:, -1, :]     # (B,H)
        return self.head(last).squeeze(1)


def load_gru(input_dim: int, device: str):
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = GRUModel(input_dim=input_dim).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def predict_probs(model, X_seq, device: str, batch_size=512):
    """
    X_seq: (N, 24, F) float32
    returns probs: (N,)
    """
    X_t = torch.from_numpy(X_seq).float()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i+batch_size].to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def main(threshold=THRESHOLD_DEFAULT):
    print("SCRIPT STARTED ✅")
    print("PYTHON:", sys.executable)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # --- load arrays ---
    X = np.load(X_PATH).astype(np.float32)   # (N,24,F)
    y = np.load(Y_PATH).astype(np.int64)     # (N,)
    g = np.load(G_PATH).astype(np.int64)     # (N,)
    h = np.load(H_PATH).astype(np.int64)     # (N,)

    print("Loaded arrays:")
    print("X:", X.shape, "y:", y.shape, "g:", g.shape, "h:", h.shape)
    print("Label counts:", np.bincount(y))

    F = X.shape[2]
    model = load_gru(input_dim=F, device=device)
    print("Loaded GRU. Feature dim F:", F)

    # --- read alerts list ---
    if not os.path.exists(ALERTS_CSV):
        raise FileNotFoundError(f"Missing file: {ALERTS_CSV} (generate it first)")

    alerts = pd.read_csv(ALERTS_CSV)
    if "patient_id" not in alerts.columns:
        raise ValueError("alerts file must contain 'patient_id' column")

    alert_pids = alerts["patient_id"].dropna().astype(int).unique().tolist()
    print("Alert patients:", len(alert_pids))

    os.makedirs(OUT_DIR, exist_ok=True)

    saved = 0
    for pid in alert_pids:
        idx = np.where(g == pid)[0]
        if len(idx) == 0:
            continue

        # sort by hour
        idx = idx[np.argsort(h[idx])]

        Xp = X[idx]    # (W,24,F)
        yp = y[idx]    # (W,)
        hp = h[idx]    # (W,)

        probs = predict_probs(model, Xp, device=device, batch_size=512)

        df_out = pd.DataFrame({
            "patient_id": pid,
            "hour_from_admission": hp,
            "true_label": yp,
            "risk_score": probs,
            "threshold": float(threshold),
            "alert": (probs >= threshold).astype(int),
        })

        # add risk level for dashboard UI
        def risk_level(p):
            if p >= threshold:
                return "HIGH"
            elif p >= max(0.0, threshold * 0.7):
                return "MED"
            else:
                return "LOW"

        df_out["risk_level"] = df_out["risk_score"].apply(risk_level)

        out_path = os.path.join(OUT_DIR, f"patient_{pid}_timeline.csv")
        df_out.to_csv(out_path, index=False)
        saved += 1

    print("\nDONE ✅")
    print("Saved timelines:", saved)
    print("Output folder:", OUT_DIR)


if __name__ == "__main__":
    # إthreshold:
    # main(threshold=0.94)
    main(threshold=THRESHOLD_DEFAULT)
