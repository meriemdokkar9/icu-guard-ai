print("SCRIPT STARTED ✅")
import sys
print("PYTHON:", sys.executable)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =========================
# FILES
# =========================
X_PATH = "X_seq_24h.npy"
Y_PATH = "y_seq_24h.npy"
G_PATH = "patient_seq_24h.npy"
H_PATH = "hour_seq_24h.npy"

CKPT_PATH = "gru_24h_model.pt"
OUT_CSV = "dashboard_predictions_gru.csv"

# default threshold (you can change)
DEFAULT_THRESHOLD = 0.94  # from your best F1 threshold in training


# =========================
# MODEL (must match training step7)
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


def load_model(input_dim, device="cpu"):
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = GRUModel(input_dim=input_dim).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def main(threshold=DEFAULT_THRESHOLD):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # load sequences built in step6
    X = np.load(X_PATH).astype(np.float32)     # (N,24,F)
    y = np.load(Y_PATH).astype(np.int64)       # (N,)
    g = np.load(G_PATH).astype(np.int64)       # (N,) patient_id
    h = np.load(H_PATH).astype(np.int64)       # (N,) hour_from_admission

    print("Loaded:", X.shape, y.shape, g.shape, h.shape)
    print("Labels:", np.bincount(y))

    F = X.shape[2]
    model = load_model(F, device=device)

    # predict in batches
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.from_numpy(X[i:i+512]).float().to(device)  # (B,24,F)
            logit = model(xb)
            prob = torch.sigmoid(logit).cpu().numpy()
            probs.append(prob)

    prob_all = np.concatenate(probs).reshape(-1)

    df_pred = pd.DataFrame({
        "patient_id": g,
        "hour_from_admission": h,
        "true_label": y,
        "risk_score": prob_all,
    })

    # choose only LAST window per patient (latest hour)
    df_last = df_pred.sort_values(["patient_id", "hour_from_admission"]).groupby("patient_id").tail(1).copy()

    df_last["threshold"] = float(threshold)
    df_last["alert"] = (df_last["risk_score"] >= threshold).astype(int)

    # Optional: risk level
    def risk_level(p):
        if p >= threshold:
            return "HIGH"
        elif p >= 0.5 * threshold:
            return "MEDIUM"
        else:
            return "LOW"

    df_last["risk_level"] = df_last["risk_score"].apply(risk_level)

    df_last = df_last.sort_values("risk_score", ascending=False).reset_index(drop=True)

    df_last.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print("Rows (patients):", len(df_last))
    print("Alerts:", int(df_last["alert"].sum()))
    print(df_last.head(10))


if __name__ == "__main__":
    main()
