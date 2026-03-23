import numpy as np
import torch
import torch.nn as nn

# =========================
# PATHS
# =========================
X_PATH = "X_seq_24h.npy"
Y_PATH = "y_seq_24h.npy"
G_PATH = "patient_seq_24h.npy"
H_PATH = "hour_seq_24h.npy"     # optional, if موجود
CKPT_PATH = "gru_24h_model.pt"

# =========================
# MODEL (must match training)
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
        return self.head(last).squeeze(1)  # (B,)

def load_model(input_dim, device="cpu"):
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = GRUModel(input_dim=input_dim).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_one_sequence(model, X_24h, device="cpu"):
    # X_24h: (24, F)
    x = torch.from_numpy(X_24h).float().unsqueeze(0).to(device)  # (1,24,F)
    with torch.no_grad():
        logit = model(x).item()
        prob = sigmoid(logit)
    return prob

def main(threshold=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load sequences (the SAME used in training)
    X = np.load(X_PATH).astype(np.float32)   # (N,24,F)
    y = np.load(Y_PATH).astype(np.int64)     # (N,)
    g = np.load(G_PATH).astype(np.int64)     # (N,)

    # hour file is optional
    try:
        h = np.load(H_PATH).astype(np.int64)
        has_h = True
    except Exception:
        h = None
        has_h = False

    print("Loaded X:", X.shape, "y:", y.shape, "groups:", g.shape)
    print("Feature dim F:", X.shape[2], "| label counts:", np.bincount(y))

    # pick a patient with many windows
    uniq, counts = np.unique(g, return_counts=True)
    pid_ok = int(uniq[np.argmax(counts)])
    print("Auto-selected patient_id:", pid_ok, "windows:", int(counts.max()))

    # take last window for that patient
    idx = np.where(g == pid_ok)[0]
    idx_last = int(idx[-1])
    X_24h = X[idx_last]      # (24,F)
    y_true = int(y[idx_last])
    hour_info = int(h[idx_last]) if has_h else None

    # load model
    model = load_model(input_dim=X.shape[2], device=device)

    prob = predict_one_sequence(model, X_24h, device=device)
    alert = int(prob >= threshold)

    print("\n=== GRU One Patient (from saved sequences) ===")
    print("Patient:", pid_ok)
    if hour_info is not None:
        print("Hour:", hour_info)
    print("Risk score:", prob)
    print("Alert:", alert, "| threshold:", threshold)
    print("True label:", y_true)

if __name__ == "__main__":
    main(threshold=0.5)

