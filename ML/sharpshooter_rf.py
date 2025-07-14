import numpy as np
import pandas as pd
import pyarrow
import os
import sys
from rf import Model
from probe_filter import Filter 

valid_probes = {
    # TRAIN
    "data/sharpshooter_labeled/sharpshooter_a01_labeled.csv": [
        (3388.49, 3400.0),
        (3414.73, 3437.75),
        (3445.45, 3503.68),
        (7325.77, 7520.3),
        (7485.8, 9789.1),
        (9808.4, 11286.6),
        (11287.0, 11531.6),
        (11752.0, 11800.4),
        (12463, 12491.15),
        (12507, 12522.5),
        (12525.5, 12573.7),
        (12680, 12720),
        (12850,12950),
        (14368.4, 14371.3),
        (14374,14414.2),
        (14420,14942),
        (15025, 15082),
        (15090, 15660),
        (15790, 15808),
        (15865, 16059),
        (16066, 16082),
        (22485, 22534),
        (22548, 22674.2),
        (22741,22758.1),
        (22840, 22954),
        (23094, 23097),
        (23124, 23965),
    ],
    "data/sharpshooter_labeled/sharpshooter_c01_labeled.csv": [
        (433.25, 461.2),
        (483.2, 620.6),
        (725.2, 887.0),
        (1934, 2117.6)
    ],
    "data/sharpshooter_labeled/sharpshooter_a10_labeled.csv": [
        (2249.4, 3064),
        (5086, 6458),
        (11162, 12385),
        (15259, 16633),
        (21795, 24624)
    ],
    
}
test = {
    "data/sharpshooter_labeled/sharpshooter_a15_labeled.csv": [
        (7532, 9921),
        (9973, 10812),
        (11368, 11553),
    ],     
}


probes = []

# for filepath, probe_timestamps in valid_probes.items():
#     file_probes = Filter.filter_data_by_time_ranges(filepath, probe_timestamps)
#     probes.extend(file_probes)

data_dir = r"C:\EPG-Project\Summer\CS-Repository\Data\Sharpshooter Data - HPR 2017\sharpshooter_labeled"
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        print(f"Reading {filename}...")
        file_path = os.path.join(data_dir, filename)
        df = pd.read_csv(file_path, engine="pyarrow")
        df.rename(columns={"pre_rect": "voltage"}, inplace = True)
        probes.append(df)


#print(probes)

rf_model = Model()
print("Training model...")
rf_model.train(probes)
print("Model trained.")

print("Running Model")
test_df = pd.read_csv(r"C:\EPG-Project\Summer\CS-Repository\ML\data\test_a02.csv")
predictions = rf_model.predict([test_df])[0]
print("Model run.")
print()
print(predictions)

print("Saving output...")

test_df["labels"] = predictions
test_df.to_csv("out.csv")
print("Output saved.")













