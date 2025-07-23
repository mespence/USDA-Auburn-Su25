import pandas as pd
import numpy as np
import random
#import torch
#from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import stft
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader import import_data
from label_mapper import load_label_map, build_label_map


class ProbeSplitterTCN:
    def __init__(self, epochs = 4, trial = None):
        random.seed(42)



