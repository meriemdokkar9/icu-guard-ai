print("STEP5 started...")
import pandas as pd
import numpy as np

CSV_FILE = "track_b50ea1e4.csv"

# إعدادات النوافذ
FS = 1                 # نعمل resample إلى 1Hz (مرة كل ثانية)
WINDOW_MIN = 5         # نافذة features: آخر 5 دقائق
HORIZON_MIN = 30       # التنبؤ: خلال 30 دقيقة القادمة
THRESH = 90            # deterioration إذا SpO2 < 90

# --- تحميل وتنظيف ---
df = pd.read_csv(CSV_FILE)
time_col = df.columns[0]
val_col = df.columns[1]

df = df[[time_col, val_col]].dropna()
df.columns = ["t", "spo2"]
df = df.sort_values("t")

# نحول الزمن لثواني صحيحة ونعمل grid 1Hz
t0 = df["t"].min()
t1 = df["t"].max()
t_grid = np.arange(np.floor(t0), np.ceil(t1) + 1, 1.0 / FS)

# نعيد فهرسة على الشبكة ونملأ القيم بطريقة forward-fill (آخر قيمة)
s = pd.Series(df["spo2"].values, index=df["t"].values)
s = s[~s.index.duplicated(keep="last")]
s_grid = s.reindex(t_grid, method="ffill")

data = pd.DataFrame({"t": t_grid, "spo2": s_grid.values}).dropna()

# --- تعريف حدث التدهور ---
data["is_low"] = (data["spo2"] < THRESH).astype(int)

# --- بناء label: هل سيحدث low خلال 30 دقيقة القادمة؟ ---
horizon_sec = HORIZON_MIN * 60
# rolling max على المستقبل: نستخدم shift(-horizon) مع rolling
# طريقة سهلة: نحسب max في [t, t+horizon]
future_max = (
    data["is_low"][::-1]
    .rolling(window=horizon_sec, min_periods=1)
    .max()[::-1]
)
data["label_next_30min"] = future_max.astype(int)

# --- features من آخر 5 دقائق ---
win_sec = WINDOW_MIN * 60

# rolling mean/min/std على الماضي
data["feat_mean_5m"] = data["spo2"].rolling(win_sec, min_periods=win_sec).mean()
data["feat_min_5m"] = data["spo2"].rolling(win_sec, min_periods=win_sec).min()
data["feat_std_5m"] = data["spo2"].rolling(win_sec, min_periods=win_sec).std()

# slope تقريبي: الفرق بين الآن وقبل 5 دقائق / الزمن
data["feat_slope_5m"] = (data["spo2"] - data["spo2"].shift(win_sec)) / win_sec

# نرمي الصفوف اللي ما عندها نافذة كاملة
train = data.dropna(subset=["feat_mean_5m", "feat_min_5m", "feat_std_5m", "feat_slope_5m"]).copy()

# نحفظ جدول التدريب
cols = ["t", "feat_mean_5m", "feat_min_5m", "feat_std_5m", "feat_slope_5m", "label_next_30min"]
train[cols].to_csv("train_table_one_track.csv", index=False)

print("Saved: train_table_one_track.csv")
print("Rows:", len(train))
print("Label distribution (0/1):")
print(train["label_next_30min"].value_counts())

