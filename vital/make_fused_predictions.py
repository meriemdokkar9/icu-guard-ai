import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gru_csv", default="gru_hourly_predictions_from_npy.csv")
    ap.add_argument("--xgb_csv", default="xgb_hourly_predictions_from_npy.csv")
    ap.add_argument("--alpha", type=float, default=0.6, help="fusion weight for GRU")
    ap.add_argument("--out_csv", default="fused_predictions.csv")
    args = ap.parse_args()

    g = pd.read_csv(args.gru_csv)
    x = pd.read_csv(args.xgb_csv)

    key = ["patient_id", "hour_from_admission", "true_label"]
    g = g[key + ["risk_score"]].rename(columns={"risk_score": "risk_gru"})
    x = x[key + ["risk_score"]].rename(columns={"risk_score": "risk_xgb"})

    m = g.merge(x, on=key, how="inner")
    if len(m) == 0:
        raise ValueError("Merge produced 0 rows. Check that both files use same patient/hour/label.")

    a = args.alpha
    m["risk_score"] = a * m["risk_gru"] + (1 - a) * m["risk_xgb"]

    out = m[key + ["risk_score"]].copy()
    out.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)
    print(out.head())
    print("Rows:", len(out))

if __name__ == "__main__":
    main()