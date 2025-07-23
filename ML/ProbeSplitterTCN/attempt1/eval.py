import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# === Load data ===
df = pd.read_csv("out.csv")  # must have 'labels' and 'pred' columns

# === Load label map ===
with open("label_map.json", "r") as f:
    label_map = json.load(f)
    label_map = {k.upper(): int(v) for k, v in label_map.items()}

# === Map labels to 0/1 ===
true = df["labels"].str.upper().map(label_map)
pred_col = df["pred"]

# Try converting predictions to binary if needed
if pred_col.dtype == object:
    try:
        pred = pred_col.map({"NP": 0, "P": 1})
    except:
        raise ValueError("Couldn't map prediction labels to binary. Make sure they are 0/1 or 'P'/'NP'.")
else:
    pred = pred_col.astype(int)

# === Drop any missing values ===
mask = true.notna() & pred.notna()
true = true[mask].reset_index(drop=True)
pred = pred[mask].reset_index(drop=True)

# === Evaluate ===
acc = accuracy_score(true, pred)
f1 = f1_score(true, pred)

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")

# # === Print mismatches ===
# inv_label_map = {0: "NP", 1: "P"}
# print("\nMismatches (index, true, predicted):")
# for i, (t, p) in enumerate(zip(true, pred)):
#     if t != p:
#         print(f"{i}: true={inv_label_map[t]}, pred={inv_label_map[p]}")
