import numpy as np
import pandas as pd
import importlib.util
"""
import sys
sys.path.insert(1, '../ML/')
"""
#from postprocessing import PostProcessor
from itertools import groupby
from ProbeSplitter import ProbeSplitter
from PyQt6.QtCore import pyqtSignal, QObject

from models import rf, tcn, unet, transformer

class Labeler(QObject):
    start_labeling_progress = pyqtSignal(int, int)
    stopped_labeling = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.stop_flag = False
        self.model = None

    def load_model(self, model_name):
        name_to_class = {
            'Random Forests (CSVs only)' : rf.Model, 
            'UNet (Block)' : unet.Model,
            'UNet (Attention)' : unet.Model, 
            'SegTransformer' : transformer.Model,
            'TCN' : tcn.Model
        }
        name_to_path = {
            'Random Forests (CSVs only)' : "models/rf_pickle", 
            'UNet (Block)' : "models/unet_weights_block",
            'UNet (Attention)' : "models/unet_weights_attention", 
            'SegTransformer' : "models/transformer_weights",
            'TCN' : "models/tcn_weights"
        }
        # Build UNet kwargs
        kwargs = {}
        # UNet settings
        if "UNet" in model_name:
            if "Block" in model_name:
                kwargs['bottleneck_type'] = 'windowed_attention'
                kwargs = kwargs | {'epochs': 64, 'lr': 0.0005, 'dropout_rate': 1e-05, 'weight_decay': 1e-06, 'num_layers': 8, 'features': 32, 'transformer_window_size': 150, 'transformer_layers': 2}
                heads_per_channel = 32
                kwargs['transformer_nhead'] = max(kwargs['features'] // heads_per_channel, 1)
                kwargs['embed_dim'] = kwargs['features']
            else:
                kwargs['bottleneck_type'] = 'block'
                kwargs = kwargs | {'epochs': 64, 'lr': 0.0005, 'dropout_rate': 0.1, 'weight_decay': 1e-06, 'num_layers': 8, 'features': 32}

        self.model = name_to_class[model_name](**kwargs)
        self.model.load(path = name_to_path[model_name])

    def leak_probe_finder(self, labels):
        non_np_indices = np.where(labels != 'NP')[0]
        probes = []
        start = 0
        end = 0
        for i in range(len(non_np_indices)):
            if i == 0:
                start = non_np_indices[i]
            elif i == len(non_np_indices) - 1:
                end = non_np_indices[i]
            elif abs(non_np_indices[i] - non_np_indices[i - 1]) > 1:
                start = non_np_indices[i]
            elif abs(non_np_indices[i] - non_np_indices[i + 1]) > 1:
                end = non_np_indices[i]
            if start > 0 and end > 0:
                probes.append((start, end))
                start = 0
                end = 0
        return probes

    def start_probe_splitting(self, epgdata, datawindow):
        data = epgdata.dfs[epgdata.current_file]
        pre_rect = data["pre_rect"].values
        self.start_labeling_progress.emit(25, 100)
        probes = ProbeSplitter.simple_probe_finder(pre_rect)
        self.start_labeling_progress.emit(50, 100)
        data['probes'] = 'NP'
        for i, (start, end) in enumerate(probes, start=1):
            data.loc[start:end, 'probes'] = 'P'
        datawindow.transition_mode = 'probes'
        datawindow.plot_recording(epgdata.current_file)
        datawindow.plot_transitions(epgdata.current_file)
        self.start_labeling_progress.emit(100, 100)

    def stop_labeling(self):
        self.stop_flag = True


    def start_labeling(self, epgdata, datawindow):
        if not self.model:
            print("No model loaded!")
            return
        current_file = epgdata.dfs[epgdata.current_file]
        current_file["file"] = "placeholder file"
        # We need to split based on the probe labels
        probe_indices = self.leak_probe_finder(current_file["probes"].values)
        probes = [current_file.iloc[start:end + 1].reset_index(drop=True).copy() 
                    for start, end in probe_indices]
        #probe_labels, logits = self.model.predict(probes, return_logits = True)

        """
        if self.stop_flag:
            print("Labeling stopped before prediction.")
            return
        self.start_labeling_progress.emit(25, 100)
        """
        probe_labels = self.model.predict(probes)
        """
        import time
        time.sleep(3)
        self.start_labeling_progress.emit(50, 100)

        if self.stop_flag:
            print("Labeling stopped during prediction.")
            self.stopped_labeling.emit()
            return
        #pp = PostProcessor(inv_label_map = self.model.inv_label_map)
        #probe_labels = [pp.postprocess_smooth(pl.squeeze(0).detach().numpy()) for pl in logits]
        #probe_labels = [self.dumb_barcode_fix(p) for p in probe_labels]
        """
        # Fill in only in probing regions
        labels = np.repeat("NP", current_file.shape[0])
        for i, probe in enumerate(probe_indices):
            start, end = probe
            labels[start:end + 1] = probe_labels[i]

        # Save and write to screen
        epgdata.set_labels(epgdata.current_file, labels)
        datawindow.transition_mode = 'labels'
        datawindow.plot_recording(epgdata.current_file)
        datawindow.plot_transitions(epgdata.current_file)
        self.start_labeling_progress.emit(100, 100)
