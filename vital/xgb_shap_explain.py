import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb

def main():
    print("Loading data...")
    df = pd.read_csv("hospital_deterioration_ml_ready.csv")

    label_col = "deterioration_next_12h"
    time_col = "hour_from_admission"

    # If patient_id exists use it for proper group split
    group_col = "patient_id" if "patient_id" in df.columns else None

    # Feature columns (exclude label + time + patient_id if exists)
    drop_cols = [label_col, time_col]
    if group_col:
        drop_cols.append(group_col)

    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[label_col].astype(int)

    # One-hot encode object/categorical columns
    cat_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
    print("Categorical columns:", cat_cols if cat_cols else "None")

    X = pd.get_dummies(X_raw, columns=cat_cols, dummy_na=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Groups for split
    if group_col:
        groups = df[group_col].values
        print("Using patient-level group split:", group_col)
    else:
        groups = np.arange(len(df))
        print("[WARN] patient_id not found; using row groups.")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = float(neg / max(pos, 1))
    print(f"Train positives={pos} negatives={neg} scale_pos_weight={spw:.2f}")

    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=1.0,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print("Test AUROC:", round(auc, 4))

    # ===== Feature Importance PNG =====
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    imp.to_csv("outputs/xgb_feature_importance_onehot.csv", index=False)

    top = imp.head(20).copy()
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"][::-1], top["importance"][::-1])
    plt.title("Top 20 XGBoost Feature Importances (One-Hot)")
    plt.tight_layout()
    plt.savefig("outputs/xgb_feature_importance_onehot.png", dpi=300)
    plt.close()
    print("Saved: outputs/xgb_feature_importance_onehot.png")

    # ===== SHAP Summary PNG =====
    print("Computing SHAP values...")
    # sample for speed and stability
    X_shap = X_test.sample(n=min(2000, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig("outputs/xgb_shap_summary_onehot.png", dpi=300)
    plt.close()
    print("Saved: outputs/xgb_shap_summary_onehot.png")

    # Also save mean absolute SHAP values table
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_rank = pd.DataFrame({"feature": X_shap.columns, "mean_abs_shap": mean_abs})
    shap_rank = shap_rank.sort_values("mean_abs_shap", ascending=False)
    shap_rank.to_csv("outputs/xgb_shap_mean_abs.csv", index=False)
    print("Saved: outputs/xgb_shap_mean_abs.csv")

if __name__ == "__main__":
    main()