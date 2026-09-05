import pandas as pd
import torch
from torch.utils.data import Dataset
from preprocessing import load_and_preprocess
from spatial_branch.ela_transform import compute_ela
from freq_branch.dct_transform import extract_dct_features
 
class FusionDataset(Dataset):
    def __init__(self, manifest_path='data/manifest.csv', split='train'):
        df = pd.read_csv(manifest_path)
        self.data = df[df['split'] == split].reset_index(drop=True)
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
 
        img = load_and_preprocess(row['filepath'])
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
 
        # Arshpreet's part — ELA input for the spatial branch
        ela = compute_ela(row['filepath'])
        ela_tensor = torch.tensor(ela, dtype=torch.float32).permute(2, 0, 1)
 
        # Anshika's part — DCT input for the frequency branch
        dct = extract_dct_features(img)
        dct_tensor = torch.tensor(dct, dtype=torch.float32).unsqueeze(0)
 
        label = torch.tensor(row['label'], dtype=torch.long)
 
        # return order: rgb image, ela input, dct input, label
        return img_tensor, ela_tensor, dct_tensor, label
