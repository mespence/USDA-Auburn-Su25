import numpy as np
import pandas as pd
import importlib
from sklearn.metrics import accuracy_score, classification_report # Ensure these are imported at the top of your file

"""
import sys
sys.path.insert(1, '../ML/')
"""
#from postprocessing import PostProcessor
#from itertools import groupby
from label_view.ProbeSplitter import ProbeSplitter
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QCursor

#from models import rf, tcn, unet, transformer

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
            'Random Forests (CSVs only)': 'models.rf',
            'UNet (Block)': 'models.unet',
            'UNet (Attention)': 'models.unet',
            'SegTransformer': 'models.transformer',
            'TCN': 'models.tcn'
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

    def start_probe_splitting(self, epgdata, datawindow):
        data = epgdata.dfs[epgdata.current_file]

        # --- Define Best Parameters from Optuna Optimization ---
        # IMPORTANT: Replace these with the actual best parameters from your Optuna run.
        # These are placeholders from your last output.

        BEST_SPLITTER_PARAMS = {
            "window": 887,
            "threshold": 0.28,
            "min_probe_length": 839,
            "np_pad": 420,
        }
        BEST_G_REFINEMENT_PARAMS = {
            "g_window_seconds": 20,
            "g_std_threshold": 0.017,
            "g_ptp_threshold": 0.144,
            "g_dilation_seconds": 10,
            "min_g_length_seconds": 10,
        }
        sample_rate = 100 # Ensure this matches your data's actual sample rate

        # 1. Prepare ground truth labels
        ground_truth_labels = data["labels"].astype(str).str.lower()
        # True for probe (P), False for NP ('n' or 'z'). 'g' and other labels are 'P'.
        true_is_probe = ~ground_truth_labels.isin(["n", "z"])

        pre_rect = data["voltage"].values # Use 'voltage' column
        self.start_labeling_progress.emit(25, 100)

        # 2. Stage 1: Run simple_probe_finder
        initial_probes_tuples = ProbeSplitter.simple_probe_finder(
            pre_rect,
            window=BEST_SPLITTER_PARAMS["window"],
            threshold=BEST_SPLITTER_PARAMS["threshold"],
            min_probe_length=BEST_SPLITTER_PARAMS["min_probe_length"],
            np_pad=BEST_SPLITTER_PARAMS["np_pad"]
        )

        # Convert initial probe tuples to a boolean array for the next stage
        initial_predicted_is_probe = np.zeros_like(pre_rect, dtype=bool)
        for start, end in initial_probes_tuples:
            start_bound = max(0, start)
            end_bound = min(len(initial_predicted_is_probe), end)
            initial_predicted_is_probe[start_bound:end_bound] = True

        self.start_labeling_progress.emit(50, 100)

        # 3. Stage 2: Refine predictions using G-signal detector
        final_predicted_is_probe = ProbeSplitter.refine_predictions_for_g(
            initial_predicted_is_probe,
            pre_rect,
            sample_rate,
            BEST_G_REFINEMENT_PARAMS["g_window_seconds"],
            BEST_G_REFINEMENT_PARAMS["g_std_threshold"],
            BEST_G_REFINEMENT_PARAMS["g_ptp_threshold"],
            BEST_G_REFINEMENT_PARAMS["g_dilation_seconds"],
            BEST_G_REFINEMENT_PARAMS["min_g_length_seconds"],
            initial_probes_tuples
        )

        # 4. Apply final predictions to the DataFrame for GUI display
        # Convert boolean array back to 'P'/'NP' strings for the 'probes' column
        data['probes'] = np.where(final_predicted_is_probe, 'P', 'NP')

        # 5. Update GUI (as per your original code)
        datawindow.transition_mode = 'probes'
        datawindow.plot_recording(epgdata.current_file)
        self.start_labeling_progress.emit(100, 100)

        # 6. Evaluate and print results to terminal
        # Ensure lengths match for evaluation
        min_len = min(len(true_is_probe), len(final_predicted_is_probe))
        true_labels_clipped = true_is_probe[:min_len]
        predicted_labels_clipped = final_predicted_is_probe[:min_len]

        # Convert boolean arrays to 'P'/'NP' strings for classification_report
        true_labels_str = np.where(true_labels_clipped, 'P', 'NP')
        predicted_labels_str = np.where(predicted_labels_clipped, 'P', 'NP')

        print("\n=== Two-Stage Probe Splitting Evaluation ===")
        print(f"Accuracy: {accuracy_score(true_labels_str, predicted_labels_str):.4f}")
        print(classification_report(true_labels_str, predicted_labels_str, target_names=["NP", "P"]))


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
