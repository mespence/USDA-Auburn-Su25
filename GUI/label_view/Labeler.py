import numpy as np
import pandas as pd
import importlib
from sklearn.metrics import accuracy_score, classification_report

"""
import sys
sys.path.insert(1, '../ML/')
"""
#from postprocessing import PostProcessor
#from itertools import groupby
from models.unet_probesplitter import UNetProbeSplitter
from models.ProbeSplitterMosquito import SimpleProbeSplitter
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QCursor

from models import unet_mosquito, unet_sharpshooter

class Labeler(QObject):
    start_labeling_progress = pyqtSignal(int, int)
    stopped_labeling = pyqtSignal()

    def __init__(self, parent = None):
        super().__init__(parent=parent)
        self.stop_flag = False
        self.model = None
    
    def load_model(self, model_name):
        model_chooser = self.parent().modelChooser
        model_chooser.blockSignals(True)
        model_chooser.setEditable(True)
        model_chooser.lineEdit().setReadOnly(True)
        model_chooser.setCurrentText(f"Loading {model_name}...")
        model_chooser.lineEdit().setStyleSheet("color: gray;")
        model_chooser.setEnabled(False)

        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.BusyCursor))
        QApplication.processEvents()

        name_to_module = {
            'Mosquito UNet (Block)': 'models.unet_mosquito',
            'Mosquito UNet (Attention)': 'models.unet_mosquito',
            'Sharpshooter UNet (Block)': 'models.unet_sharpshooter',
        }
        name_to_path = {
            'Mosquito UNet (Block)' : "models/unet_block_mosquito_weights",
            'Mosquito UNet (Attention)' : "models/unet_attention_mosquito_weights", 
            'Sharpshooter UNet (Block)' : "models/unet_block_sharpshooter_weights",
        }

        # Build UNet kwargs
        kwargs = {}
        # UNet settings
        if "Mosquito" in model_name:
            if "Attention" in model_name:
                kwargs['bottleneck_type'] = 'windowed_attention'
                kwargs = kwargs | {'epochs': 64, 'lr': 0.0005, 'dropout_rate': 1e-05, 'weight_decay': 1e-06, 'num_layers': 8, 'features': 32, 'transformer_window_size': 150, 'transformer_layers': 2}
                heads_per_channel = 32
                kwargs['transformer_nhead'] = max(kwargs['features'] // heads_per_channel, 1)
                kwargs['embed_dim'] = kwargs['features']
            else:
                kwargs['bottleneck_type'] = 'block'
                kwargs = kwargs | {'epochs': 64, 'lr': 0.0005, 'dropout_rate': 0.1, 'weight_decay': 1e-06, 'num_layers': 8, 'features': 32}
        elif "Sharpshooter" in model_name:
            kwargs['bottleneck_type'] = 'block'
            kwargs = kwargs | {'epochs': 64, 'lr': 0.0005, 'dropout_rate': 0.1, 'weight_decay': 1e-06, 'num_layers': 6, 'features': 64} # TODO update params

        print(f'Importing {model_name}')
        module = importlib.import_module(name_to_module[model_name])
        print(f'{model_name} imported')
        ModelClass = getattr(module, "Model")
        self.model = ModelClass(**kwargs)
        self.model.load(path = name_to_path[model_name])

        model_chooser.setEnabled(True)
        model_chooser.lineEdit().setStyleSheet("")
        model_chooser.setEditable(False)
        model_chooser.setCurrentText(model_name)
        model_chooser.blockSignals(False)

        QApplication.restoreOverrideCursor()

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

    def start_sharpshooter_probe_splitting(self, epgdata, datawindow):
        probe_splitter = UNetProbeSplitter()
        probe_splitter.load(r".\models\unet_probesplitter_weights")

        data = epgdata.dfs[epgdata.current_file]
        true_str = data["labels"].astype(str).str.upper()
        true_binary = (~true_str.isin(["N", "Z"])).astype(int).to_numpy()

        self.start_labeling_progress.emit(25, 100)

        predicted_str = probe_splitter.predict([data])[0]
        predicted_np = np.array(predicted_str, dtype="object")
        predicted_binary = np.array([0 if label == "NP" else 1 for label in predicted_str])

        # TODO need to uncomment this eventually when post prcoessing for less barcoding doen
        # data["probes"] = predicted_np
    
        # datawindow.transition_mode = 'probes'
        # datawindow.plot_recording(epgdata.current_file)
        self.start_labeling_progress.emit(100, 100)

        # Ensure lengths match for evaluation
        assert len(true_binary) == len(predicted_binary)


        # TODO need to comment this out, this is for evaluation
        print("\n=== Probe Splitting Evaluation Report ===")
        print("Sample predicted:", predicted_str[:20])
        print("Sample ground truth:", np.where(true_binary, "P", "NP")[:20])
        print(f"Accuracy: {accuracy_score(true_binary, predicted_binary):.4f}")
        print(classification_report(true_binary, predicted_binary, target_names=["NP", "P"]))

    def start_mosquito_probe_splitting(self, epgdata, datawindow):
        data = epgdata.dfs[epgdata.current_file]
        pre_rect = data["voltage"].values
        self.start_labeling_progress.emit(25, 100)
        probes = SimpleProbeSplitter.simple_probe_finder(pre_rect)
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
