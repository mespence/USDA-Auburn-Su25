import sys
import os
import pandas as pd

class Filter:
    def filter_data_by_time_ranges(input_filepath, time_ranges, time_column_name="time"):
        """
        Filters data from an input file based on specified time ranges and returns
        a list of pandas DataFrame objects, one for each range.

        The script assumes:
        1. The input file is a CSV with a header.
        2. The 'time' value is located in the column specified by 'time_column_name'.
        3. The 'time' value can be converted to a numeric type.

        Args:
            input_filepath (str): Path to the input data file (e.g., 'data.csv').
            time_ranges (list of tuples): A list of (start_time, end_time) tuples.
                                        Data points whose time falls within these ranges
                                        (inclusive: start_time <= current_time <= end_time)
                                        will be included in the respective DataFrame.
            time_column_name (str): The name of the column that contains the time values.
                                    Defaults to "Time".

        Returns:
            list of pandas.DataFrame: A list where each element is a DataFrame
                                    containing data filtered for one of the provided time ranges.
                                    Returns an empty list if no data is found or errors occur.
        """
        print(f"Attempting to filter data from '{input_filepath}'...")
        print(f"Filtering for time ranges: {time_ranges}")
        print(f"Looking for time data in column: '{time_column_name}'")

        filtered_dfs = []

        try:
            # Read the entire CSV file into a pandas DataFrame
            df = pd.read_csv(input_filepath)

            # Check if the time column exists
            if time_column_name not in df.columns:
                print(f"Error: Time column '{time_column_name}' not found in the input file's header. "
                    f"Available columns: {df.columns.tolist()}")
                return []

            # Convert the time column to numeric, coercing errors to NaN
            df[time_column_name] = pd.to_numeric(df[time_column_name], errors='coerce')

            # Drop rows where the time column could not be converted to a number (NaNs)
            df.dropna(subset=[time_column_name], inplace=True)

            if df.empty:
                print("Warning: No valid numeric time data found after parsing. Returning empty list.")
                return []

            # Iterate through each time range and filter the DataFrame
            for start_time, end_time in time_ranges:
                # Filter rows where the time is within the current range (inclusive)
                df_range = df[(df[time_column_name] >= start_time) & (df[time_column_name] <= end_time)].copy()
                filtered_dfs.append(df_range)
                print(f"  Range ({start_time}, {end_time}): Found {len(df_range)} rows.")

            print(f"Filtering complete. Generated {len(filtered_dfs)} DataFrame objects.")
            return filtered_dfs

        except FileNotFoundError:
            print(f"Error: Input file not found at '{input_filepath}'. Please ensure the path is correct.")
            return []
        except pd.errors.EmptyDataError:
            print(f"Error: Input file '{input_filepath}' is empty.")
            return []
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return []


if __name__ == "__main__":
    time_ranges = [(0.02,0.05), (0.07,0.08)]
    print(Filter.filter_data_by_time_ranges("/Users/ashleykim/Desktop/USDA/USDA-Auburn-Su25/Data/Sharpshooter Data - HPR 2017/sharpshooter_a01_labeled.csv", time_ranges))