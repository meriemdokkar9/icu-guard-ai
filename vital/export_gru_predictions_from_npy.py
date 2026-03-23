import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class GRUSeq(nn.Module):
    def __init__(self, input_dim=33, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # training used "head.1" instead of "fc" -> we match it
        self.head = nn.Sequential(
            nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        logits = self.head(last).squeeze(-1)
        return logits


def standardize(X, mu, sd):
    mu = np.array(mu, dtype=np.float32).reshape(1, 1, -1)
    sd = np.array(sd, dtype=np.float32).reshape(1, 1, -1)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="gru_24h_model.pt")
    ap.add_argument("--X", default="X_seq_24h.npy")
    ap.add_argument("--y", default="y_seq_24h.npy")
    ap.add_argument("--pid", default="patient_seq_24h.npy")
    ap.add_argument("--hour", default="hour_seq_24h.npy")
    ap.add_argument("--out_csv", default="gru_hourly_predictions_from_npy.csv")
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args()

    print("Loading checkpoint:", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    print("Loading NPY files...")
    X = np.load(args.X).astype(np.float32)        # (N, 24, 33)
    y = np.load(args.y).astype(int).reshape(-1)   # (N,)
    pid = np.load(args.pid).reshape(-1)           # (N,)
    hour = np.load(args.hour).reshape(-1)         # (N,)

    print("Shapes:", "X", X.shape, "y", y.shape, "pid", pid.shape, "hour", hour.shape)

    # Apply the SAME normalization used during training (mu/sd in checkpoint)
    mu = ckpt.get("mu", None)
    sd = ckpt.get("sd", None)
    if mu is not None and sd is not None:
        print("Applying normalization from checkpoint (mu/sd).")
        X = standardize(X, mu, sd)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Build model matching checkpoint architecture
    model = GRUSeq(
        input_dim=X.shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Load with strict=False to tolerate harmless naming differences if any
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model = model.to(device)
    model.eval()

    probs = np.zeros((X.shape[0],), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, X.shape[0], args.batch_size):
            xb = torch.from_numpy(X[i:i+args.batch_size]).to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            probs[i:i+args.batch_size] = p

    out = pd.DataFrame({
        "patient_id": pid.astype(int),
        "hour_from_admission": hour.astype(int),
        "true_label": y.astype(int),
        "risk_score": probs
    })

    out.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)
    print(out.head())


if __name__ == "__main__":
    main()