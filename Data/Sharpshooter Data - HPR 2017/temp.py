import pandas as pd

file_path = r"D:\USDA-Auburn\CS-Repository\Data\Sharpshooter Data - HPR 2017\sharpshooter_labeled\sharpshooter_d07_labeled.csv"

# Read the CSV
df = pd.read_csv(file_path)

# Replace 2 with "F2" in the 'labels' column (works if it's int or string)
if "labels" in df.columns:
    df["labels"] = df["labels"].replace("2", "F2")

# Save back (overwrite or change path)
df.to_csv(file_path, index=False)


# import pandas as pd

# df = pd.read_csv(r"D:\USDA-Auburn\CS-Repository\Data\Sharpshooter Data - HPR 2017\sharpshooter_labeled\sharpshooter_d07_labeled.csv")

# # Replace 'column_name' with the actual column you're checking
# matching_rows = df[df["labels"] == "2"]

# if not matching_rows.empty:
#     # Get row indices (zero-based)
#     print("Matching row indices:", matching_rows.index.tolist())
# else:
#     print("No rows found with value 2 in 'column_name'")
