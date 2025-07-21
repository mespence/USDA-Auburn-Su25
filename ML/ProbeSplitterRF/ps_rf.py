import os
import numpy as np
import pandas as pd

import pickle
from tqdm import tqdm
from optuna.trial import Trial
from transform_worker import transform_single_probe
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier


class Model:
    def __init__(self, save_path: str = None, trial: Trial = None) -> None:
        self.chunk_seconds = 3
        self.num_estimators = 128
        self.num_freqs = 7
        self.max_depth = 16
        self.sample_rate = 100
        self.chunk_size = self.chunk_seconds * self.sample_rate
        self.random_state = 42
        self.model = None
        self.save_path = save_path
        self.model_path = "./ML/rf_pickle"
        
        if trial:
            self.chunk_seconds = trial.suggest_int('chunk_seconds', 1, 3)
            self.num_freqs = trial.suggest_int('num_freqs', 1, 10)
            self.num_estimators = trial.suggest_categorical('num_estimators', [8, 16, 32, 64, 128])
            self.max_depth = trial.suggest_categorical('max_depth', [8, 16, 32, 64, 128])

    def extract_features(self, dfs, training = True) -> list[pd.DataFrame]:
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    transform_single_probe,
                    probe,
                    self.chunk_size,
                    self.sample_rate,
                    self.num_freqs,
                    training
                ) for probe in dfs
            ]

            transformed_dfs = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing probes"):
                result = future.result()
                transformed_dfs.append(result)

        return transformed_dfs


    def train(self, dfs):
        features_dfs = self.extract_features(dfs)
        train = pd.concat(features_dfs)
        X_train = train.drop(["label"], axis=1)
        Y_train = train["label"]
        rf = RandomForestClassifier(self.num_estimators, class_weight="balanced", max_depth = self.max_depth, verbose=1, n_jobs=-1)
        self.model = rf.fit(X_train, Y_train)

    def predict(self, dfs) -> list[pd.DataFrame]:
        features_dfs = self.extract_features(dfs, training = False)
        predictions = []
        for raw_df, test_df in zip(dfs, features_dfs):
            pred = self.model.predict(test_df)

            # We need to expand the prediction based on the sample rate
            pred = np.repeat(pred, self.chunk_seconds * self.sample_rate)
            # Expand until the end since probe is never exactly divisible by window size
            pred = np.pad(pred, (0, len(raw_df) - len(pred)), 'edge')
            predictions.append(pred)
        return predictions

    def save(self) -> None:
        with open(self.model_path, 'ab') as model_save:
            pickle.dump(self.model, model_save)
    def load(self, path = None) -> None:
        with open(path, 'rb') as model_save:
            self.model = pickle.load(model_save)


if __name__ == "__main__":
    pass