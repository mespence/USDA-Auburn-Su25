import numpy as np
import pandas as pd
import os
import sys
from rf import Model
from probe_filter import Filter 



valid_probes = {
    # TRAIN
    "data/sharpshooter_labeled/sharpshooter_a01_labeled.csv": [
        (),
        (),
        (),
    ],
    "data/sharpshooter_labeled/sharpshooter_b01_labeled.csv": [
        (),
        (),
        (),
    ],
    "data/sharpshooter_labeled/sharpshooter_c01_labeled.csv": [
        (),
        (),
        (),
    ],
    "data/sharpshooter_labeled/sharpshooter_d01_labeled.csv": [
        (),
        (),
        (),
    ],
    # TEST
    "data/sharpshooter_labeled/sharpshooter_a02_labeled.csv": [
        (),
        (),
        (),
    ],     
}

probes = []

for filepath, probe_timestamps in valid_probes.items():
    file_probes = Filter.filter_data_by_time_ranges(filepath, probe_timestamps)
    probes.extend(file_probes)

rf_model = Model()
rf_model.train(probes)













