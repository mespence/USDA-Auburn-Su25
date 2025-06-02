from pandas import read_csv, DataFrame
import numpy as np
import pandas as pd
import re
import windaq
import os

class EPGData():
        """
        A class for interacting with and saving our waveform recordings
        """
        
        def __init__(self):
                self.dfs = {}  # A dictionary of filename : pandas dataframe objects
                self.label_column = 'labels'
                self.probe_column = 'probes'
                self.prepost_suffix = "_rect"
                self.current_file = "test_recording.csv"
                self.dir_path = os.path.dirname(os.path.realpath(__file__))

        def load_data(self, file):
                """
                load_data takes in either a Windaq or CSV file, converts it
                into a pandas dataframe, and makes it available for use
                Inputs:
                        files: a filenames (either .DAQ or .csv) as a string
                Returns:
                        True if successful, False otherwise
                """

                if re.search(r'\.(WDQ|DAQ)$', file, re.IGNORECASE):
                        windaq_file = windaq.windaq(file)
                        # TODO: don't hardcode channel count and names
                        # TODO: add event markers as column
                        df = DataFrame({
                                "time":      windaq_file.time(),
                                "pre_rect":  windaq_file.data(1),
                                "post_rect": windaq_file.data(2),
                        })
                        # This will overwrite if there are multiple
                        # markers pointing to the same index
                        comments = [None for i in range(len(df))]
                        for event in windaq_file.eventmarkers:
                            if 'comment' in event:
                                comments[event['index']] = event['comment']
                            elif 'timestamp' in event:
                                comments[event['index']] = event['timestamp']
                            else:
                                comments[event['index']] = ''
                        df['comments'] = comments
                        self.dfs[file] = df
                elif re.search(r'\.csv$', file, re.IGNORECASE):
                        full_path = os.path.join(self.dir_path,file)
                        try:
                                self.dfs[file] = read_csv(full_path)
                        except FileNotFoundError:
                                print(f"Could not find {full_path}")
                                return False
                else:
                        # unknown file extension
                        return False

                self.current_file = file
                if not self.label_column in self.dfs[file]:
                        self.dfs[file][self.label_column] = np.nan
                if not self.probe_column in self.dfs[file]:
                        self.dfs[file][self.probe_column] = np.nan
                return True

        def export_csv(self, file, destination):
                """
                export_csv saves a CSV of loaded EPG data to disk.
                Inputs:
                        file: string containing the key of the recording in
                              self.dfs
                        destination: where the file should be saved
                Returns:
                        True if successful, False otherwise
                """
                
                try:
                        self.dfs[file].to_csv(destination)
                except:
                        return False
                return True

        def export_txt(self, file, destination):
                """
                export_txt saves a plaintext file to disk with timestamps
                for the end of each label in the loaded EPG data.
                Inputs:
                        file: string containing the key of the recording in
                              self.dfs
                        destination: where the file should be saved
                Returns:
                        True if successful, False otherwise
                """
                df = self.dfs[file]
                df1 = df[:-1].reset_index()['labels']
                df2 = df[1:].reset_index()['labels']
                where = np.where(~((df1 == df2) | (df1.isna() & df2.isna())))[0]
                where = np.append(where, [len(df)-1])
                with open(destination, 'w') as f:
                        for i, row in df.iloc[where].iterrows():
                                f.write(f'"{row["labels"]}"\n    {row["time"]:.02f}\n')

        def get_recording(self, file, prepost):
                """
                get_recording returns a pandas dataframe containing time
                and voltage data only.
                Inputs:
                        file: string containing the key of the recording
                        prepost: string containing either "pre" or "post"
                                 for either pre or post rectification data
                Outputs:
                        A pandas dataframe containing the desired data if
                        request is valid.
                """
                
                if not prepost in ["pre", "post"]:
                        raise Exception(f"{prepost} is not either 'pre' or 'post'")

                elif not file in self.dfs:
                        raise Exception(f"{file} is not a key in self.dfs")

                else:
                        return self.dfs[file][['time', f'{prepost}{self.prepost_suffix}']]

        def set_labels(self, file, labels):
                """
                set_labels sets the labels of file to be those given in the
                input labels
                Inputs:
                        file: string containing the key of the recording
                        labels: a numpy array containing label strings of
                                the same length as the number of columns
                                in the recording dataframe
                Returns:
                        None
                """
                if not file in self.dfs.keys():
                        raise Exception(f"{file} is not a key in self.dfs")
                elif not self.dfs[file].shape[0] == len(labels):
                        raise Exception(f"Input labels is of size {len(labels)}"
                                        f"but dataframe has length {self.dfs[file].shape[0]}")
                else:
                        self.dfs[file][self.label_column] = labels

        def set_transitions(self, file, transitions, section_type):
                """
                set_transitions takes in transitions in the format of get_transitions
                and writes them to the labels for files. Think of this as an alternative
                to set_labels that is sometime more convenient. Assumes file already has labels.
                Inputs:
                        file: string containing the key of the recording
                        transitions: a list of (time, label) tuples
                Returns:
                        Nothing
                """
                if not file in self.dfs:
                        raise Exception(f"{file} is not a key in self.dfs")

                df = self.dfs[file].copy()
                df['time'] = df['time'].apply(lambda x : round(x, 2))
                df = df.set_index('time')

                section_to_column = {
                    "labels" : self.label_column,
                    "probes" : self.probe_column
                }
                col = section_to_column[section_type]
                df[col] = df[col].astype(str)
                df[col] = pd.NA
                for time, label in transitions:
                        rounded_time = round(time, 2)
                        df.loc[rounded_time, col] = label
                new_labels = df.reset_index()[col].ffill()
                self.dfs[file][col] = new_labels
                                
        def get_transitions(self, file, section_type):
                """
                get_transitions looks at the labels from file and returns the times
                where transitions occur and the state that follows them. Always has
                a transition at time 0 for convenience.
                Inputs:
                        file: string containing the key of the recording
                Outputs:
                        a list of (time, label) tuples. If no labels, then just has
                        one element at time 0.
                """
                if not file in self.dfs:
                        raise Exception(f"{file} is not a key in self.dfs")
                
                df = self.dfs[file]
                times = df['time'].values
                transitions = []
                
                if section_type == 'labels':
                        labels = df[self.label_column].values
                        transitions = [(0, labels[0])]

                        for i in range(1, len(labels)):
                                if labels[i] != labels[i - 1]:
                                        transitions.append((times[i], labels[i]))

                if section_type == 'probes':
                        probes = df[self.probe_column].values
                        transitions = [(0, probes[0])]

                        for i in range(1, len(probes)):
                                if probes[i] != probes[i - 1]:
                                        transitions.append((times[i], probes[i]))

                return transitions
