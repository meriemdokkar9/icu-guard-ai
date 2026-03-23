import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# =========================
# PATHS
# =========================
X_PATH = "X_seq_24h.npy"
Y_PATH = "y_seq_24h.npy"
G_PATH = "patient_seq_24h.npy"
H_PATH = "hour_seq_24h.npy"

CKPT_PATH = "gru_24h_model.pt"
OUT_PNG = "patient6_risk_curve.png"

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
        out, _ = self.gru(x)      # (B,T,H)
        last = out[:, -1, :]      # (B,H)
        return self.head(last).squeeze(1)


def load_gru(input_dim, device="cpu"):
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = GRUModel(input_dim=input_dim).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def auto_pick_patient_with_most_windows(g):
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(counts)])


def main(patient_id=None, threshold=0.94):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    X = np.load(X_PATH).astype(np.float32)  # (N,24,F)
    y = np.load(Y_PATH).astype(np.int64)
    g = np.load(G_PATH).astype(np.int64)
    h = np.load(H_PATH).astype(np.int64)

    print("Loaded:", X.shape, y.shape, g.shape, h.shape)
    print("Label counts:", np.bincount(y))
    print("Feature dim F:", X.shape[2])

    if patient_id is None:
        patient_id = auto_pick_patient_with_most_windows(g)

    idx = np.where(g == patient_id)[0]
    if len(idx) == 0:
        print("No windows found for patient:", patient_id)
        return

    # sort windows by hour
    idx = idx[np.argsort(h[idx])]
    Xp = X[idx]   # (W,24,F)
    yp = y[idx]
    hp = h[idx]

    print("Patient:", patient_id, "| windows:", len(idx), "| last hour:", int(hp[-1]))
    print("True label counts (patient):", np.bincount(yp) if len(np.unique(yp)) > 1 else {int(yp[0]): len(yp)})

    model = load_gru(input_dim=Xp.shape[2], device=device)

    # predict risk per window
    probs = []
    with torch.no_grad():
        for i in range(0, len(Xp), 256):
            xb = torch.from_numpy(Xp[i:i+256]).float().to(device)   # (B,24,F)
            logit = model(xb)
            prob = torch.sigmoid(logit).cpu().numpy()
            probs.append(prob)
    probs = np.concatenate(probs)

    # Plot
    plt.figure(figsize=(9, 4.5))
    plt.plot(hp, probs, marker="o", linewidth=1.5)
    plt.axhline(threshold, linestyle="--")
    plt.title(f"GRU Risk over Time | patient_id={patient_id}")
    plt.xlabel("hour_from_admission")
    plt.ylabel("risk score (prob deterioration_next_12h)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    plt.close()

    # quick summary
    alerts = (probs >= threshold).astype(int)
    print("\nSaved plot:", OUT_PNG)
    print("Max risk:", float(probs.max()), "| #alerts:", int(alerts.sum()), "| threshold:", threshold)

    # show top 5 risky hours
    topk = np.argsort(-probs)[:5]
    print("\nTop 5 risk points:")
    for k in topk:
        print(f"  hour={int(hp[k])}  risk={float(probs[k]):.4f}  true_label={int(yp[k])}  alert={int(alerts[k])}")


if __name__ == "__main__":
    # إذا بدك نفس المريض اللي طلع عندك (6):
    main(patient_id=6, threshold=0.94)

    # أو خليها تختار تلقائيًا المريض الأطول:
    # main(patient_id=None, threshold=0.94)
