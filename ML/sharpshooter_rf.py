import numpy as np
import pandas as pd
import os
import sys
from rf import Model
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

excluded = {
    "a01", "a02", "a03", "a10", "a15",
    "b01", "b02", "b04", "b07", "b12", "b188", "b202", "b206", "b208",
    "c046", "c07", "c09", "c10",
    "d01", "d03", "d056", "d058", "d12",
    "b11", # TEST FILE
}


def read_file(file_path):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, engine="pyarrow")
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path, columns=["time", "pre_rect", "labels"], engine="pyarrow")
    else:
        return None
    return df.rename(columns={"pre_rect": "voltage"})



if __name__ == "__main__":
    data_dir = r"D:\USDA-Auburn\CS-Repository\Data\Sharpshooter Data - HPR 2017\sharpshooter_parquet"
    probes = []

    # Collect valid file paths
    file_paths = [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if not any(bug_id in filename for bug_id in excluded) and filename.endswith((".csv", ".parquet"))
    ]
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(read_file, path) for path in file_paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading files"):
            df = future.result()
            if df is not None:
                probes.append(df)


    # for filename in os.listdir(data_dir):
    #     if any(bug_id in filename for bug_id in excluded):
    #         continue
    #     if filename.endswith(".csv"):
    #         print(f"Reading {filename}...")
    #         file_path = os.path.join(data_dir, filename)
    #         df = pd.read_csv(file_path, engine="pyarrow")
    #         df = df.rename(columns={"pre_rect": "voltage"})
    #         probes.append(df)
    #     elif filename.endswith(".parquet"):
    #         print(f"Reading {filename}...")
    #         file_path = os.path.join(data_dir, filename)
    #         df = pd.read_parquet(file_path, columns=["time", "pre_rect", "labels"], engine="pyarrow")
    #         df = df.rename(columns={"pre_rect": "voltage"})
    #         probes.append(df)

    rf_model = Model()
    rf_model.train(probes)


    print("Running Model")
    test_df = pd.read_csv(r"D:\USDA-Auburn\CS-Repository\Data\Sharpshooter Data - HPR 2017\sharpshooter_labeled\sharpshooter_b11_labeled.csv")
    test_df.rename(columns={"pre_rect": "voltage"}, inplace = True)
    predictions = rf_model.predict([test_df])[0]
    print("Model run.")
    print()
    print(predictions)

    print("Saving output...")
    rf_model.save()
    test_df["labels"] = predictions
    test_df.to_csv("out.csv")
    print("Output saved.")













