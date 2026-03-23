import pandas as pd

# ضعي هنا اول tid من ملف csv
TID = "b50ea1e4216b5c88f1b8d53c5f2c4eff2993edb6"

url = f"https://api.vitaldb.net/{TID}"

# VitalDB يرجع CSV مضغوط gzip — pandas يفتحه تلقائياً
df = pd.read_csv(url, compression="gzip")

print("Downloaded rows, cols:", df.shape)
print(df.head(10))

# حفظه كملف csv عادي
out = f"track_{TID[:8]}.csv"
df.to_csv(out, index=False)
print("Saved:", out)

