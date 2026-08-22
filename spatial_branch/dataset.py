import pandas as pd
import torch
from torch.utils.data import Dataset
from preprocessing import load_and_preprocess

class TamperedImageDataset(Dataset):
    def __init__(self, manifest_path='data/manifest.csv', split='train'):
        # read the whole manifest, then keep only rows matching our split
        df = pd.read_csv(manifest_path)
        self.data = df[df['split'] == split].reset_index(drop=True)

    def __len__(self):
        # PyTorch needs to know how many examples exist
        return len(self.data)

    def __getitem__(self, idx):
        # PyTorch calls this one index at a time to grab a single example
        row = self.data.iloc[idx]
        img = load_and_preprocess(row['filepath'])  # returns a (224,224,3) numpy array

        # PyTorch expects images as (channels, height, width) not (height, width, channels)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(row['label'], dtype=torch.long)

        return img, label


if __name__ == '__main__':
    # this block only runs when the file is executed directly,
    # not when another file imports TamperedImageDataset from it
    ds = TamperedImageDataset(split='train')
    print("Number of training images:", len(ds))

    img, label = ds[0]
    print("Image shape:", img.shape)   # expect torch.Size([3, 224, 224])
    print("Label:", label)              # expect tensor(0) or tensor(1)