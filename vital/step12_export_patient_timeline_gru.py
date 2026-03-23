import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ===== PATHS =====
X_PATH = "X_seq_24h.npy"
Y_PATH = "y_seq_24h.npy"
G_PATH = "patient_seq_24h.npy"
H_PATH = "hour_seq_24h.npy"
CKPT_PATH = "gru_24h_model.pt"

# ===== MODEL (must match training) =====
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

@torch.no_grad()
def predict_probs(model, X_np, device="cpu", batch=512):
    # X_np: (N,24,F)
    X_t = torch.from_numpy(X_np).float().to(device)
    probs = []
    for i in range(0, len(X_t), batch):
        xb = X_t[i:i+batch]
        logits = model(xb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient_id", type=int, default=None, help="patient id to export")
    ap.add_argument("--threshold", type=float, default=0.94, help="alert threshold")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    X = np.load(X_PATH).astype(np.float32)
    y = np.load(Y_PATH).astype(np.int64)
    g = np.load(G_PATH).astype(np.int64)
    h = np.load(H_PATH).astype(np.int64)

    print("Loaded:", X.shape, y.shape, g.shape, h.shape)
    F = X.shape[2]
    model = load_model(F, device=device)

    # auto pick a patient with most windows if not provided
    if args.patient_id is None:
        uniq, cnt = np.unique(g, return_counts=True)
        pid = int(uniq[np.argmax(cnt)])
        print("Auto-selected patient_id:", pid, "windows:", int(cnt.max()))
    else:
        pid = int(args.patient_id)

    idx = np.where(g == pid)[0]
    if len(idx) == 0:
        print("No windows for patient:", pid)
        return

    Xp = X[idx]
    yp = y[idx]
    hp = h[idx]

    # sort by hour
    order = np.argsort(hp)
    Xp, yp, hp = Xp[order], yp[order], hp[order]

    prob = predict_probs(model, Xp, device=device, batch=512)

    df = pd.DataFrame({
        "patient_id": pid,
        "hour_from_admission": hp,
        "true_label": yp,
        "risk_score": prob,
    })
    df["threshold"] = float(args.threshold)
    df["alert"] = (df["risk_score"] >= df["threshold"]).astype(int)

    out = f"patient_{pid}_timeline_full.csv"
    df.to_csv(out, index=False)

    print("\nSaved:", out, "rows:", len(df))
    print("Max risk:", float(df['risk_score'].max()))
    print("Total alerts:", int(df['alert'].sum()))
    print(df.tail(12))

if __name__ == "__main__":
    main()
