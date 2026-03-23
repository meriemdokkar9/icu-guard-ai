import torch

ckpt = torch.load("gru_24h_model.pt", map_location="cpu", weights_only=False)

print("=== CHECKPOINT KEYS ===")
print(list(ckpt.keys()))

print("\n=== BASIC INFO ===")
print("F =", ckpt.get("F"))
print("T =", ckpt.get("T"))

print("\n=== SEARCHING FOR FEATURE NAMES ===")
possible_keys = [
    "feat_cols",
    "feature_cols",
    "feature_names",
    "columns",
    "colnames",
    "X_cols"
]

found = False
for k in possible_keys:
    if k in ckpt:
        print("FOUND:", k)
        print("Number of features:", len(ckpt[k]))
        print("First 10 names:", ckpt[k][:10])
        found = True

if not found:
    print("No explicit feature names found in checkpoint.")