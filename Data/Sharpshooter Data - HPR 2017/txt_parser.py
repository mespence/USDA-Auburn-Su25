import re
import pandas as pd
import windaq as wdq
from pathlib import Path

DATA_DIR = r"Data\Sharpshooter Data - HPR 2017\windaq_files"
TXT_DIR = r"Data\Sharpshooter Data - HPR 2017\Notepad text files - corrected"
OUTPUT_PATH = r"Data\Sharpshooter Data - HPR 2017\sharpshooter_txt_parsed"

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

for bug_id in BUG_IDS:
    if len(bug_id) == 4 and bug_id[1] == 0:
        txt_id = bug_id[0] + bug_id[2:]
    else:
        txt_id = bug_id
    
    print(f"Parsing {bug_id}...")

    # Get rep number, txt ID, and maybe channel ID
    channel_id = None
    if len(bug_id[1:]) == 2:
        rep = "rep" + str(int(bug_id[1:])) # drop leading zero by converting to int
        txt_id = bug_id
    elif len(bug_id[1:]) == 3:
        if bug_id[1] == "0":
            rep = "rep" + bug_id[2]
            txt_id = bug_id[0] + bug_id[2:]
        elif bug_id[1] == "1":
            rep = "rep" + bug_id[1:3]
            txt_id = bug_id
        elif bug_id[1] == "2":
            rep = "rep" + bug_id[1:3]
            txt_id = bug_id

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



    # Construct label data from .txt file
    pattern = re.compile(rf"^{re.escape(txt_id)}(?!\d)")
    txt_path = [
        path for path in Path(TXT_DIR).rglob("*")
        if path.is_file() and pattern.match(path.stem)
    ][0]

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        # Prepare result list
        time_label_pairs = []

        # Iterate over every 2 lines (label + data)
        for i in range(0, len(lines)-1, 2):
            label_line = lines[i].strip().strip('"')
            data_line = lines[i+1]

            # Extract label (removes padding spaces)
            label = label_line.strip()

            # Extract the first float from the data line
            match = re.search(r"[-+]?\d*\.\d+|\d+", data_line)
            if match:
                time = float(match.group())
                if time != 0.0:
                    time_label_pairs.append((time, label.upper()))

        # Create DataFrame
        label_df = pd.DataFrame(time_label_pairs, columns=["time", "label"])
    
    df = pd.DataFrame(columns = ["time", "pre_rect", "post_rect", "labels"])
    df["time"] = daq.time().astype("float32").round(3)
    df["pre_rect"] = daq.data(channel_id - 1).astype("float32")
    df["post_rect"] = daq.data(channel_id).astype("float32")

    label_end_times = label_df["time"].to_numpy()
    label_names = label_df["label"].to_numpy()

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

    df.to_csv(rf"{OUTPUT_PATH}\sharpshooter_{bug_id}_txt.csv")
    print(f"{bug_id} parsed.")
    print(df.head())  




