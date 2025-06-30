import pandas as pd
import windaq as wdq
import os
from pathlib import Path
import re

#################################################################################################
# This parser pairs time-voltage data from .WDQ files with the corresponding labels from 
# the label .CSV based on the ID of the bug (see structure below). It is written specifically 
# for the HPR 2017 sharpshooter dataset (06/2017-7/2017, .WDQs and .CSV) provided by Dr. Backus, though may
# be expandable in the future if all of their data is structured according to these bug IDs.
#################################################################################################

# Bug ID Structure:
# letter: treatment
#       a: clean-grape
#       b: infec-grape
#       c: clean-wild
#       d: infec-wild
# first two numbers: rep #
#       e.g., 144 -> rep14
# third number: channel # (1-indexed w/ odd being pre-rect, even being post-rect)
#       e.g., 144 -> rep14 channel 4
#
# IDs need to be two numbers in CSV, so for three digit IDs:
#       0XX - drop the 0 (e.g., a082 -> a82 in csv)
#       1XX - drop the 1 (e.g., a144 -> a44 in csv)
#       20X - drop the 0 (e.g., c204 -> c24 in csv)

# hardcoded IDs from legend
BUG_IDS = [
    "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a09", "a10", "a11",
    "a12", "a13", "a15", "a16", "a17", "a082", "a088", "a144", "a148", "a192",
    "b01", "b02", "b03", "b04", "b05", "b06", "b07", "b09", "b10", "b11",
    "b12", "b13", "b15", "b16", "b17", "b184", "b188", "b196", "b202", "b206", "b208", 
    "c01", "c02", "c03", "c06", "c07", "c08", "c09", "c10", "c11", "c12",
    "c13", "c14", "c15", "c16", "c17", "c046", "c048", "c182", "c186", "c204",
    "d01", "d02", "d03", "d06", "d07", "d08", "d09", "d10", "d11", "d12",
    "d13", "d14", "d15", "d16", "d17", "d056", "d058", "d194", "d196", "d198",
]
TREATMENT_MAP = {
    "a": "clean-grape",
    "b": "infec-grape",
    "c": "clean-wild",
    "d": "infec-wild",
}


DATA_DIR = os.path.dirname(os.path.realpath(__file__)) # change dir as needed
OUTPUT_DIR = DATA_DIR + r"\parsed_output"
CSV_PATH = DATA_DIR + r"\Backus BGSS 2017 HPR data - for Mudd CS.csv"

label_df = pd.read_csv(CSV_PATH)

for bug_id in BUG_IDS:
    print(f"Parsing {bug_id}...")

    # Get rep number, csv ID, and maybe channel ID
    channel_id = None
    if len(bug_id[1:]) == 2:
        rep = "rep" + str(int(bug_id[1:])) # drop leading zero by converting to int
        csv_id = bug_id
    elif len(bug_id[1:]) == 3:
        if bug_id[1] == "0":
            rep = "rep" + bug_id[2]
            csv_id = bug_id[0] + bug_id[2:]
        elif bug_id[1] == "1":
            rep = "rep" + bug_id[1:3]
            csv_id = bug_id[0] + bug_id[2:]
        elif bug_id[1] == "2":
            rep = "rep" + bug_id[1:3]
            csv_id = bug_id[0] + bug_id[1] + bug_id[3]
        channel_id = int(bug_id[3])
   
    # Only works if there is are no duplicate reps. Additional duplicate reps are ignored
    pattern = re.compile(rf"(?:^|_)({re.escape(rep)})(?:_|$)")
    filepath = [
        str(path) for path in Path(DATA_DIR).rglob(f"*{rep}*")
        if pattern.search(path.stem)
    ][0]


    daq = wdq.windaq(filepath)
    
    # Find channel ID if it's not in bug ID
    if channel_id is None:
        treatment = TREATMENT_MAP.get(bug_id[0])
        if treatment is None:
            print(f"Unknown treatement type '{bug_id[0]}', skipping ID.")
            continue
        # Normalize for their treatment typos (- vs _, infec vs infect)
        treatment_regex = treatment.replace("-", "[-_]").replace("infec", "infect?")
        treatment_pattern = re.compile(treatment_regex, re.IGNORECASE)
        channel_annotations = [daq.chAnnotation(x+1) for x in range(8)]

        # Normalize for their post typos (post vs pos)
        post_pattern = re.compile(r"\bpos(?:t)?\b", re.IGNORECASE)

        # Find channel ID
        channel_id = next(
            (
                i + 1
                for i, annot in enumerate(channel_annotations)
                if treatment_pattern.search(annot) and post_pattern.search(annot)
            ),
            None
        )

        if channel_id is None:
            print("No matching channel found in .WDQ file. Skipping ID.")
            continue
    
    df = pd.DataFrame(columns = ["time", "pre_rect", "post_rect", "labels"])
    df["time"] = daq.time().astype("float32").round(3)
    df["pre_rect"] = daq.data(channel_id - 1).astype("float32")
    df["post_rect"] = daq.data(channel_id).astype("float32")

    bug_id_mask = [label_df["insectno"] == csv_id]

    # Get label data for this bug ID
    bug_labels = label_df[label_df["insectno"] == csv_id]

    if bug_labels.empty:
        print(f"No label data found for {csv_id}, skipping.")
        continue

    # Get end times and waveform labels
    label_end_times = bug_labels["tbf"].to_numpy()
    label_names = bug_labels["waveform"].to_numpy()

    # Prepend 0
    label_bins = [0] + list(label_end_times)

    # Assign label to each timestamp in the waveform
    df["labels"] = pd.cut(
        df["time"],
        bins=label_bins,
        labels=label_names,
        include_lowest=True,
        right=True,
        ordered=False
    )
    print(f"{bug_id} parsed.")
    
    csv_name = rf"{OUTPUT_DIR}\sharphooter_{bug_id}.csv"
    print(f"Writing {bug_id}...")
    df.to_csv(csv_name)
    print(f"{bug_id} written.")






    
            



