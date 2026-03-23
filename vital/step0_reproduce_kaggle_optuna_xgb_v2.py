import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import xgboost as xgb
import optuna

# ============================
# Load Data
# ============================
df = pd.read_csv("hospital_deterioration_hourly_panel.csv")
target_col = "deterioration_next_12h"

# Drop non-features columns
drop_cols = [target_col]
if "patient_id" in df.columns:
    drop_cols.append("patient_id")
if "hour_from_admission" in df.columns:
    drop_cols.append("hour_from_admission")

X = df.drop(columns=drop_cols)
y = df[target_col].astype(int)

print("Loaded dataset:", df.shape)
print("Features shape:", X.shape)
print("Label counts:", np.bincount(y))
print("Num features:", X.shape[1])

# ============================
# Convert object columns to categorical
# ============================
obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
print("\nObject columns:", obj_cols)

for c in obj_cols:
    X[c] = X[c].astype("category")

# ============================
# Stratified Random Split (Kaggle Style)
# ============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n=== Kaggle-style Split ===")
print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

# ============================
# Patient overlap check (proof)
# ============================
if "patient_id" in df.columns:
    train_patients = set(df.loc[X_train.index, "patient_id"])
    val_patients = set(df.loc[X_val.index, "patient_id"])
    overlap = len(train_patients & val_patients)
    print("Patient overlap (train vs val):", overlap)

# ============================
# Compute scale_pos_weight
# ============================
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print("\nscale_pos_weight:", scale_pos_weight)

# Convert to DMatrix (enable categorical)
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

# ============================
# Optuna Objective
# ============================
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "enable_categorical": True,

        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),

        "scale_pos_weight": scale_pos_weight,
        "seed": 42
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, "validation")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    preds = model.predict(dval)
    auc = roc_auc_score(y_val, preds)
    return auc

# ============================
# Run Optuna
# ============================
print("\n=== Running Optuna Hyperparameter Tuning (Kaggle style) ===")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("\nBest Trial ROC-AUC:", study.best_value)
print("Best Params:", study.best_params)

# ============================
# Train Final Model using best params
# ============================
best_params = study.best_params
best_params.update({
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "enable_categorical": True,
    "scale_pos_weight": scale_pos_weight,
    "seed": 42
})

final_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=3000,
    evals=[(dval, "validation")],
    early_stopping_rounds=50,
    verbose_eval=50
)

# ============================
# Evaluation
# ============================
val_preds = final_model.predict(dval)

roc_auc = roc_auc_score(y_val, val_preds)
pr_auc = average_precision_score(y_val, val_preds)

print("\n=== FINAL Kaggle-style XGBoost Results ===")
print("ROC-AUC:", roc_auc)
print("PR-AUC :", pr_auc)

y_pred_class = (val_preds >= 0.5).astype(int)
print("\nClassification Report (thr=0.5):\n")
print(classification_report(y_val, y_pred_class))

# Save model
final_model.save_model("xgb_kaggle_style_optuna.json")
print("\nSaved model: xgb_kaggle_style_optuna.json")
