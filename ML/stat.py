import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# === File paths ===
ground_truth_path = r"D:\USDA-Auburn\CS-Repository\Data\Sharpshooter Data - HPR 2017\sharpshooter_labeled\sharpshooter_b11_labeled.csv"
ml_output_path = r"D:\USDA-Auburn\CS-Repository\ML\out.csv"

# === Load the data ===
df_truth = pd.read_csv(ground_truth_path)
df_pred = pd.read_csv(ml_output_path)

# === Make sure lengths match ===
assert len(df_truth) == len(df_pred), "Mismatch in number of rows between CSVs."

# === Extract labels (adjust column name if needed) ===
y_true = df_truth["labels"]
y_pred = df_pred["labels"]

# === Compute statistics ===
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
conf_mat = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)

# === Print results ===
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_mat)
print("\nClassification Report:")
print(report)
