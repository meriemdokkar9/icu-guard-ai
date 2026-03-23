import os
import joblib
import pandas as pd
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "xgb_deterioration_model.joblib")
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "vital", "hospital_deterioration_hourly_panel.csv"))

DROP_COLS = [
    "patient_id",
    "deterioration_next_12h",
    "deterioration_event",
    "deterioration_hour",
    "deterioration_within_12h_from_admission",
]

print("=" * 80)
print("MODEL PATH:", MODEL_PATH)
print("DATA PATH :", DATA_PATH)
print("=" * 80)

model = joblib.load(MODEL_PATH)

print("\n[1] MODEL TYPE")
print(type(model))

print("\n[2] MODEL ATTRIBUTES")
for attr in [
    "feature_names_in_",
    "n_features_in_",
    "get_booster",
    "named_steps",
    "steps",
]:
    has_attr = hasattr(model, attr)
    print(f"{attr}: {has_attr}")

if hasattr(model, "named_steps"):
    print("\n[3] PIPELINE STEPS")
    for k, v in model.named_steps.items():
        print(f"- {k}: {type(v)}")

df = pd.read_csv(DATA_PATH)
print("\n[4] DATAFRAME SHAPE")
print(df.shape)

print("\n[5] ALL COLUMNS")
print(df.columns.tolist())

X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

print("\n[6] X SHAPE AFTER DROP")
print(X.shape)

print("\n[7] X COLUMNS AFTER DROP")
print(X.columns.tolist())

print("\n[8] FIRST ROW SAMPLE")
print(X.iloc[0].to_dict())

print("\n[9] TRY PREDICT_PROBA ON FIRST ROW")
try:
    one_row = X.iloc[[0]].copy()
    pred = model.predict_proba(one_row)
    print("SUCCESS: predict_proba worked")
    print(pred)
except Exception as e:
    print("FAILED on raw row")
    print("ERROR:", repr(e))
    traceback.print_exc()

print("\n[10] TRY NUMERIC-ONLY ROW")
try:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    one_row_num = X[numeric_cols].iloc[[0]].copy()
    print("Numeric cols count:", len(numeric_cols))
    pred = model.predict_proba(one_row_num)
    print("SUCCESS: numeric-only predict_proba worked")
    print(pred)
except Exception as e:
    print("FAILED on numeric-only row")
    print("ERROR:", repr(e))
    traceback.print_exc()

print("\n[11] TRY TO INSPECT PIPELINE PREPROCESS IF EXISTS")
if hasattr(model, "named_steps") and "prep" in model.named_steps:
    prep = model.named_steps["prep"]
    print("prep type:", type(prep))
    try:
        Xt = prep.transform(X.iloc[[0]])
        print("prep.transform success")
        print("Transformed shape:", Xt.shape)
    except Exception as e:
        print("prep.transform failed")
        print("ERROR:", repr(e))
        traceback.print_exc()

print("\nDONE")
print("=" * 80)