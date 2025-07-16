import os
import pandas as pd

def convert_csvs_to_parquet(input_dir, output_dir, recursive=True, index=False, verbose=True):
    """
    Converts all .csv files in `input_dir` into .parquet files in `output_dir`,
    preserving subdirectory structure if recursive.

    Args:
        input_dir (str): Root folder containing .csv files.
        output_dir (str): Root folder to write .parquet files.
        recursive (bool): Whether to walk through subdirectories.
        index (bool): Whether to preserve index in output Parquet.
        verbose (bool): Whether to print progress messages.
    """
    for root, _, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(output_path, exist_ok=True)

        for filename in files:
            if filename.endswith(".csv"):
                input_file = os.path.join(root, filename)
                output_file = os.path.join(output_path, filename.replace(".csv", ".parquet"))

                try:
                    df = pd.read_csv(input_file, low_memory=False)
                    df.to_parquet(output_file, index=index)
                    if verbose:
                        print(f"[✓] Converted: {input_file} → {output_file}")
                except Exception as e:
                    print(f"[!] Failed to convert {input_file}: {e}")

        if not recursive:
            break


if __name__ == "__main__":
    INPUT_DIR  = r"D:\USDA-Auburn\CS-Repository\Data\Sharpshooter Data - HPR 2017\sharpshooter_labeled"
    OUTPUT_DIR = r"D:\USDA-Auburn\CS-Repository\Data\Sharpshooter Data - HPR 2017\sharpshooter_parquet"

    convert_csvs_to_parquet(INPUT_DIR, OUTPUT_DIR)
