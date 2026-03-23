import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

HOURS_PER_DAY = 24


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def try_parse_datetime(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().mean() > 0.9:
            return parsed
    return s


def to_hours_diff(a, b) -> float:
    diff = a - b
    if hasattr(diff, "total_seconds"):
        return float(diff.total_seconds() / 3600.0)
    return float(diff)


def prepare_sorted(df: pd.DataFrame, patient_col: str, time_col: str) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = try_parse_datetime(df[time_col])
    df = df.sort_values([patient_col, time_col]).reset_index(drop=True)
    return df


def pick_vitals_feature_cols(
    df: pd.DataFrame,
    patient_col: str,
    time_col: str,
    label_col: str,
    exclude_extra: set[str],
) -> list[str]:
    exclude = {patient_col, time_col, label_col} | set(exclude_extra)
    cols = [c for c in df.columns if c not in exclude]

    keywords = [
        "heart", "hr", "pulse",
        "resp", "rr", "respiratory",
        "spo2", "o2", "sao2", "o2sat",
        "temp", "temperature",
        "sbp", "dbp", "map", "bp", "blood_pressure",
    ]

    vitals = []
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in keywords) and pd.api.types.is_numeric_dtype(df[c]):
            vitals.append(c)

    if len(vitals) >= 3:
        return vitals

    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols


def build_windows(
    df: pd.DataFrame,
    patient_col: str,
    time_col: str,
    label_col: str,
    feature_cols: list[str],
    window_hours: int = 24,
    stride: int = 1,
):
    """
    Past-only sliding windows:
      window ending at index i uses [i-window_hours+1 .. i]
      label taken at i (predicting next 12h as per dataset)
    Returns:
      X: (n_windows, window_hours, n_features)
      meta: DataFrame with patient_id, t_end, y
    """
    df = prepare_sorted(df, patient_col, time_col)
    X_list = []
    meta_rows = []

    for pid, g in df.groupby(patient_col, sort=False):
        g = g.sort_values(time_col)

        if len(g) < window_hours:
            continue

        vals = g[feature_cols].to_numpy(dtype=float, copy=False)
        y = g[label_col].to_numpy(dtype=int, copy=False)
        t = g[time_col].to_numpy(copy=False)

        for i in range(window_hours - 1, len(g), stride):
            window = vals[i - window_hours + 1 : i + 1]
            X_list.append(window)
            meta_rows.append((pid, t[i], int(y[i])))

    if len(X_list) == 0:
        X = np.empty((0, window_hours, len(feature_cols)), dtype=float)
        meta = pd.DataFrame(columns=[patient_col, "t_end", "y"])
    else:
        X = np.stack(X_list, axis=0).astype(float)
        meta = pd.DataFrame(meta_rows, columns=[patient_col, "t_end", "y"])

    return X, meta


def flatten_last_hour_window(X: np.ndarray) -> np.ndarray:
    return X[:, -1, :]


def compute_first_event_times(meta: pd.DataFrame, patient_col: str) -> pd.Series:
    return meta.loc[meta["y"] == 1].groupby(patient_col)["t_end"].min()


def apply_suppression(
    df_in: pd.DataFrame,
    patient_col: str,
    theta: float,
    cooldown_hours: int = 6,
    consecutive_k: int = 3,
) -> pd.DataFrame:
    """
    Suppression logic for real ICU monitoring:
      - Raise alert only after k consecutive raw alerts.
      - After an alert, block new alerts for cooldown_hours.
    """
    df = df_in.copy()
    df["raw_alert"] = (df["p"] >= theta).astype(int)
    df["alert"] = 0

    for pid, g in df.groupby(patient_col, sort=False):
        g = g.sort_values("t_end")
        last_alert_time = None
        consec = 0

        for ridx, row in g.iterrows():
            if row["raw_alert"] == 1:
                consec += 1
            else:
                consec = 0

            if consec >= consecutive_k:
                if last_alert_time is None:
                    df.loc[ridx, "alert"] = 1
                    last_alert_time = row["t_end"]
                    consec = 0
                else:
                    dt = row["t_end"] - last_alert_time
                    if hasattr(dt, "total_seconds"):
                        hours_since = dt.total_seconds() / 3600.0
                    else:
                        hours_since = float(dt)

                    if hours_since >= cooldown_hours:
                        df.loc[ridx, "alert"] = 1
                        last_alert_time = row["t_end"]
                        consec = 0

    return df


def evaluate_streaming(
    meta: pd.DataFrame,
    p: np.ndarray,
    theta: float,
    patient_col: str,
    suppression: bool,
    cooldown_hours: int,
    consecutive_k: int,
):
    eval_df = meta.copy()
    eval_df["p"] = p

    if suppression:
        eval_df = apply_suppression(
            eval_df,
            patient_col=patient_col,
            theta=theta,
            cooldown_hours=cooldown_hours,
            consecutive_k=consecutive_k,
        )
    else:
        eval_df["alert"] = (eval_df["p"] >= theta).astype(int)

    total_alerts = int(eval_df["alert"].sum())
    total_steps = int(len(eval_df))
    alerts_per_patient_day = (total_alerts / total_steps) * HOURS_PER_DAY if total_steps else 0.0

    flagged_patients = eval_df.groupby(patient_col)["alert"].max()
    pct_patients_flagged = float(100.0 * flagged_patients.mean()) if len(flagged_patients) else 0.0

    first_event = compute_first_event_times(eval_df, patient_col)
    patients_with_event = list(first_event.index)

    detected = 0
    missed = 0
    lead_times = []

    for pid in patients_with_event:
        t_event = first_event.loc[pid]
        g = eval_df.loc[eval_df[patient_col] == pid].sort_values("t_end")

        alerts_before = g.loc[(g["t_end"] < t_event) & (g["alert"] == 1)]
        if len(alerts_before) == 0:
            missed += 1
            continue

        detected += 1
        t_first_alert = alerts_before["t_end"].iloc[0]
        lead_times.append(to_hours_diff(t_event, t_first_alert))

    n_events = len(patients_with_event)
    event_recall = (detected / n_events) if n_events else float("nan")

    lead_arr = np.array(lead_times, dtype=float)
    median_lead = float(np.median(lead_arr)) if len(lead_arr) else float("nan")
    iqr_lead = float(np.percentile(lead_arr, 75) - np.percentile(lead_arr, 25)) if len(lead_arr) else float("nan")

    return {
        "theta": float(theta),
        "alerts_per_patient_day": float(alerts_per_patient_day),
        "pct_patients_flagged": float(pct_patients_flagged),
        "n_events": int(n_events),
        "event_recall": float(event_recall),
        "missed_events": int(missed),
        "median_lead_hours": float(median_lead),
        "iqr_lead_hours": float(iqr_lead),
    }


def save_table_png(df: pd.DataFrame, path: str, title: str):
    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)

    df_show = df.copy()
    # nice formatting
    for c in ["alerts_per_patient_day", "pct_patients_flagged", "event_recall", "median_lead_hours", "iqr_lead_hours"]:
        if c in df_show.columns:
            df_show[c] = df_show[c].map(lambda x: f"{x:.3f}" if isinstance(x, (float, np.floating)) and not np.isnan(x) else str(x))

    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_recall_vs_alerts(df_raw: pd.DataFrame, df_sup: pd.DataFrame, out_path: str):
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()

    ax.plot(df_raw["alerts_per_patient_day"], df_raw["event_recall"], marker="o", label="Raw alerts")
    ax.plot(df_sup["alerts_per_patient_day"], df_sup["event_recall"], marker="o", label="Suppressed alerts")

    for _, r in df_sup.iterrows():
        ax.annotate(f"θ={r['theta']:.1f}", (r["alerts_per_patient_day"], r["event_recall"]), fontsize=8)

    ax.set_xlabel("Alerts per patient-day")
    ax.set_ylabel("Event recall (first deterioration)")
    ax.set_title("Real-time: Event recall vs Alert burden")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_leadtime_vs_alerts(df_raw: pd.DataFrame, df_sup: pd.DataFrame, out_path: str):
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()

    ax.plot(df_raw["alerts_per_patient_day"], df_raw["median_lead_hours"], marker="o", label="Raw alerts")
    ax.plot(df_sup["alerts_per_patient_day"], df_sup["median_lead_hours"], marker="o", label="Suppressed alerts")

    ax.set_xlabel("Alerts per patient-day")
    ax.set_ylabel("Median lead time (hours)")
    ax.set_title("Real-time: Lead time vs Alert burden")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--patient_col", type=str, default="patient_id")
    ap.add_argument("--time_col", type=str, default="hour_from_admission")
    ap.add_argument("--label_col", type=str, default="deterioration_next_12h")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--thresholds", type=str, default="0.2,0.3,0.4,0.5,0.6")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--exclude_cols", type=str, default="deterioration_hour",
                    help="Comma-separated columns to exclude from features (e.g., deterioration_hour,los_hours)")
    ap.add_argument("--cooldown", type=int, default=6, help="Suppression cooldown in hours")
    ap.add_argument("--consecutive", type=int, default=3, help="k consecutive hours above threshold to trigger alert")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    print("Loading CSV:", args.csv)
    df = pd.read_csv(args.csv)

    for c in [args.patient_col, args.time_col, args.label_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # numeric label
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce").fillna(0).astype(int)

    exclude_extra = set()
    if args.exclude_cols.strip():
        exclude_extra = {c.strip() for c in args.exclude_cols.split(",") if c.strip()}

    feature_cols = pick_vitals_feature_cols(df, args.patient_col, args.time_col, args.label_col, exclude_extra)
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found after exclusion.")

    print(f"Selected {len(feature_cols)} vital/numeric feature columns.")
    print("Sample features:", feature_cols[:12])

    # numeric + missing handling (pandas >=2)
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].ffill()
    df[feature_cols] = df[feature_cols].bfill()
    df[feature_cols] = df[feature_cols].fillna(0)

    # Group split by patient
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    idx = np.arange(len(df))
    groups = df[args.patient_col].to_numpy()
    train_idx, test_idx = next(gss.split(idx, groups=groups))
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    print("Train patients:", df_train[args.patient_col].nunique(),
          "| Test patients:", df_test[args.patient_col].nunique())

    # Baseline LR on per-hour rows
    X_train = df_train[feature_cols].to_numpy(dtype=float, copy=False)
    y_train = df_train[args.label_col].to_numpy(dtype=int, copy=False)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, class_weight="balanced"))
    ])

    print("Training baseline LogisticRegression (vitals only)...")
    pipe.fit(X_train, y_train)

    # sanity on test rows
    X_test_rows = df_test[feature_cols].to_numpy(dtype=float, copy=False)
    y_test_rows = df_test[args.label_col].to_numpy(dtype=int, copy=False)
    p_test_rows = pipe.predict_proba(X_test_rows)[:, 1]
    if int(y_test_rows.sum()) > 0:
        auroc = roc_auc_score(y_test_rows, p_test_rows)
        auprc = average_precision_score(y_test_rows, p_test_rows)
        print(f"[Row sanity] AUROC={auroc:.4f} | AUPRC={auprc:.4f}")
    else:
        print("[Row sanity] WARNING: no positives in test rows (rare split).")

    # Real-time windows on TEST
    Xw, meta = build_windows(
        df_test,
        patient_col=args.patient_col,
        time_col=args.time_col,
        label_col=args.label_col,
        feature_cols=feature_cols,
        window_hours=args.window,
        stride=args.stride
    )
    if len(meta) == 0:
        raise ValueError("No windows built. Check window length/time column sorting.")

    X_last = flatten_last_hour_window(Xw)
    p_stream = pipe.predict_proba(X_last)[:, 1]

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    # RAW metrics
    raw_rows = []
    for th in thresholds:
        raw_rows.append(
            evaluate_streaming(
                meta, p_stream, th, args.patient_col,
                suppression=False,
                cooldown_hours=args.cooldown,
                consecutive_k=args.consecutive
            )
        )
    df_raw = pd.DataFrame(raw_rows)
    raw_csv = os.path.join(args.outdir, "realtime_results_raw.csv")
    df_raw.to_csv(raw_csv, index=False)
    print("\nSaved:", raw_csv)
    print(df_raw)

    # SUPPRESSED metrics
    sup_rows = []
    for th in thresholds:
        sup_rows.append(
            evaluate_streaming(
                meta, p_stream, th, args.patient_col,
                suppression=True,
                cooldown_hours=args.cooldown,
                consecutive_k=args.consecutive
            )
        )
    df_sup = pd.DataFrame(sup_rows)
    sup_csv = os.path.join(args.outdir, "realtime_results_suppressed.csv")
    df_sup.to_csv(sup_csv, index=False)
    print("\nSaved:", sup_csv)
    print(df_sup)

    # PNG outputs
    plot_recall_vs_alerts(
        df_raw, df_sup,
        os.path.join(args.outdir, "realtime_recall_vs_alerts.png")
    )
    plot_leadtime_vs_alerts(
        df_raw, df_sup,
        os.path.join(args.outdir, "realtime_leadtime_vs_alerts.png")
    )
    save_table_png(
        df_sup.round(4),
        os.path.join(args.outdir, "realtime_table_suppressed.png"),
        title=f"Suppressed alerts (cooldown={args.cooldown}h, consecutive={args.consecutive})"
    )

    print("\nPNG saved in outputs/:")
    print("- realtime_recall_vs_alerts.png")
    print("- realtime_leadtime_vs_alerts.png")
    print("- realtime_table_suppressed.png")
    print("\nDONE ✅")


if __name__ == "__main__":
    main()