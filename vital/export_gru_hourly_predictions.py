import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class GRU24h(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        logits = self.fc(last).squeeze(-1)
        return logits


def build_windows(df, patient_col, time_col, label_col, feature_cols, window=24):
    df = df.sort_values([patient_col, time_col]).reset_index(drop=True)

    X_list = []
    meta_rows = []

    for pid, g in df.groupby(patient_col, sort=False):
        g = g.sort_values(time_col)
        if len(g) < window:
            continue

        vals = g[feature_cols].to_numpy(dtype=np.float32)
        y = g[label_col].astype(int).to_numpy()
        t = g[time_col].to_numpy()

        for i in range(window - 1, len(g)):
            X_list.append(vals[i - window + 1 : i + 1])
            meta_rows.append((pid, t[i], int(y[i])))

    X = (
        np.stack(X_list, axis=0)
        if len(X_list) > 0
        else np.empty((0, window, len(feature_cols)), dtype=np.float32)
    )
    meta = pd.DataFrame(meta_rows, columns=[patient_col, time_col, "true_label"])
    return X, meta


def load_checkpoint_model(checkpoint, model: nn.Module):
    """
    Handles common checkpoint formats:
      - state_dict directly
      - dict with key 'model' holding state_dict
      - dict with key 'state_dict' holding state_dict
    """
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            model.load_state_dict(checkpoint["model"])
            return checkpoint
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            model.load_state_dict(checkpoint["state_dict"])
            return checkpoint
        # Sometimes keys are prefixed (e.g. "module.")
        if any(k.startswith("gru.") or k.startswith("fc.") for k in checkpoint.keys()):
            model.load_state_dict(checkpoint)
            return checkpoint

        raise RuntimeError(
            f"Checkpoint dict does not contain a usable state_dict. Keys: {list(checkpoint.keys())}"
        )

    # If entire model object saved
    if isinstance(checkpoint, nn.Module):
        return checkpoint

    raise RuntimeError(f"Unsupported checkpoint type: {type(checkpoint)}")


def apply_standardization(X: np.ndarray, mu, sd) -> np.ndarray:
    """
    Apply per-feature standardization if mu/sd exist in checkpoint.
    Supports mu/sd as lists, numpy arrays, or torch tensors.
    Expected shape: (F,)
    """
    if mu is None or sd is None:
        return X

    mu = np.array(mu, dtype=np.float32).reshape(1, 1, -1)
    sd = np.array(sd, dtype=np.float32).reshape(1, 1, -1)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_csv", default="gru_hourly_predictions.csv")
    ap.add_argument("--patient_col", default="patient_id")
    ap.add_argument("--time_col", default="hour_from_admission")
    ap.add_argument("--label_col", default="deterioration_next_12h")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--hidden_dim", type=int, default=32)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args()

    print("Loading data:", args.csv)
    df = pd.read_csv(args.csv)

    feature_cols = [
        "heart_rate",
        "respiratory_rate",
        "spo2_pct",
        "temperature_c",
        "systolic_bp",
        "diastolic_bp",
    ]
    for c in [args.patient_col, args.time_col, args.label_col] + feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0)

    print("Building 24h windows...")
    X, meta = build_windows(
        df,
        args.patient_col,
        args.time_col,
        args.label_col,
        feature_cols,
        window=args.window,
    )
    print("Windows:", X.shape, "Meta rows:", len(meta))
    if len(meta) == 0:
        raise ValueError("No windows built.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = GRU24h(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)

    # Load weights (handles ckpt["model"])
    loaded = load_checkpoint_model(ckpt, model)
    if isinstance(loaded, nn.Module):
        model = loaded

    # Optional: apply checkpoint standardization if available
    mu = None
    sd = None
    if isinstance(ckpt, dict):
        mu = ckpt.get("mu", None)
        sd = ckpt.get("sd", None)

    if mu is not None and sd is not None:
        print("Applying checkpoint standardization using mu/sd.")
        X = apply_standardization(X, mu, sd)

    model = model.to(device)
    model.eval()

    probs = np.zeros((X.shape[0],), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, X.shape[0], args.batch_size):
            xb = torch.from_numpy(X[i : i + args.batch_size]).to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            probs[i : i + args.batch_size] = p

    out = meta.copy()
    out["risk_score"] = probs
    out.to_csv(args.out_csv, index=False)

    print("Saved:", args.out_csv)
    print(out.head())


if __name__ == "__main__":
    main()