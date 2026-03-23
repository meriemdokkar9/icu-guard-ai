import numpy as np
import torch
import torch.nn as nn

X_PATH = "X_seq_24h.npy"
Y_PATH = "y_seq_24h.npy"
G_PATH = "patient_seq_24h.npy"
H_PATH = "hour_seq_24h.npy"

CKPT_PATH = "gru_24h_model.pt"

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(1)

def load_gru(input_dim, device="cpu"):
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = GRUModel(input_dim=input_dim).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def predict_patient_timeline(patient_id=6, threshold=0.94, show_rows=40):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    X = np.load(X_PATH).astype(np.float32)
    y = np.load(Y_PATH).astype(np.int64)
    g = np.load(G_PATH).astype(np.int64)
    h = np.load(H_PATH).astype(np.int64)

    idx = np.where(g == patient_id)[0]
    if len(idx) == 0:
        print("No windows for patient:", patient_id)
        return

    idx = idx[np.argsort(h[idx])]
    Xp, yp, hp = X[idx], y[idx], h[idx]

    model = load_gru(input_dim=Xp.shape[2], device=device)

    probs = []
    with torch.no_grad():
        for i in range(0, len(Xp), 256):
            xb = torch.from_numpy(Xp[i:i+256]).float().to(device)
            prob = torch.sigmoid(model(xb)).cpu().numpy()
            probs.append(prob)
    probs = np.concatenate(probs)

    alerts = (probs >= threshold).astype(int)

    print("\n=== Dashboard Simulation (Per-hour) ===")
    print(f"patient_id={patient_id} | windows={len(Xp)} | threshold={threshold}")
    print("Format: hour | risk | alert | true_label")

    # طباعة آخر show_rows صفوف (أهم شيء لعرض الطبيب)
    start = max(0, len(Xp) - show_rows)
    for i in range(start, len(Xp)):
        print(f"{int(hp[i]):>4} | {float(probs[i]):.4f} | {int(alerts[i])} | {int(yp[i])}")

    # ملخص
    print("\nSummary:")
    print("Max risk:", float(probs.max()))
    print("Total alerts:", int(alerts.sum()))
    first_alert = np.where(alerts == 1)[0]
    if len(first_alert) > 0:
        fa = first_alert[0]
        print("First alert at hour:", int(hp[fa]), "| risk:", float(probs[fa]))
    else:
        print("No alerts for this patient under this threshold.")

if __name__ == "__main__":
    predict_patient_timeline(patient_id=6, threshold=0.94, show_rows=40)
