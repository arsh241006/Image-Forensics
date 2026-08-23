import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from preprocessing import load_and_preprocess
from freq_branch.dct_transform import extract_dct_features

class FrequencyDataset(Dataset):
    def __init__(self, manifest_path='data/manifest.csv', split='train'):
        df = pd.read_csv(manifest_path)
        self.data = df[df['split'] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = load_and_preprocess(row['filepath'])
        dct_map = extract_dct_features(img)
        # add a channel dimension: (224,224) -> (1,224,224) since our small
        # CNN expects a channel dimension even though this is grayscale-like
        dct_tensor = torch.tensor(dct_map, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(row['label'], dtype=torch.long)
        return dct_tensor, label