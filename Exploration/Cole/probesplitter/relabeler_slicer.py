import numpy as np
import pandas as pd
import os
import sys
from rf import Model
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def read_file(file_path, desired = None, save_dir=None):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, engine="pyarrow")
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path, columns=["time", "pre_rect", "labels"], engine="pyarrow")
    else:
        return None
    
    df.rename(columns={"pre_rect": "voltage"})
    df_copy = df.copy()

    if "labels" in df_copy.columns and desired is not None:
        if desired == "p":
            df_copy["labels"] = df_copy["labels"].replace({"N": "NP", "Z": "NP"})
        if desired == "np":
            df_copy.loc[~df_copy["labels"].isin(["N", "Z"]), "labels"] = "P"
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.basename(file_path)
        save_path = os.path.join(save_dir, base_name)
        if file_path.endswith(".csv"):
            df_copy.to_csv(save_path, index=True) #these are true now because idk which index will be helpful
        elif file_path.endswith(".parquet"):
            df_copy.to_parquet(save_path, index=True)
    return df_copy

if __name__ == "__main__":
    data_dir = r"/Users/cathy/coding/bugs2025/USDA-Auburn-Su25/Exploration/Cole/probesplitter/gooddata"
    save_dir_1 = r"/Users/cathy/coding/bugs2025/USDA-Auburn-Su25/Exploration/Cole/probesplitter/gooddata_edited_p"
    save_dir_2 = r"/Users/cathy/coding/bugs2025/USDA-Auburn-Su25/Exploration/Cole/probesplitter/gooddata_edited_np"
    probes = []

    # Collect valid file paths
    file_paths = [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith((".csv", ".parquet"))
    ]
    
    with ThreadPoolExecutor() as executor:
        futures1 = [executor.submit(read_file, path, "p", save_dir_1) for path in file_paths]
        for future in tqdm(as_completed(futures1), total=len(futures1), desc="Reading files"):
            df = future.result()
            if df is not None:
                probes.append(df)
        futures2 = [executor.submit(read_file, path, "np", save_dir_2) for path in file_paths]
        for future in tqdm(as_completed(futures2), total=len(futures2), desc="Reading files"):
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

    # rf_model = Model()
    # rf_model.train(probes)


    # print("Running Model")
    # test_df = pd.read_csv(r"/Users/cathy/coding/bugs2025/USDA-Auburn-Su25/Exploration/Cole/probesplitter/gooddata/sharpshooter_b11_labeled.csv", index_col=0)
    # test_df.rename(columns={"pre_rect": "voltage"}, inplace = True)
    # predictions = rf_model.predict([test_df])[0]
    # print("Model run.")
    # print()
    # print(predictions)

    # print("Saving output...")
    # rf_model.save()
    # test_df["labels"] = predictions
    # test_df.to_csv("out.csv")
    # print("Output saved.")















