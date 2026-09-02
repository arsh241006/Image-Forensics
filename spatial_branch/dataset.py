import pandas as pd
import torch
from torch.utils.data import Dataset
from preprocessing import load_and_preprocess
from spatial_branch.ela_transform import compute_ela
from spatial_branch.srm_transform import compute_srm

class TamperedImageDataset(Dataset):
    def __init__(self, manifest_path='data/manifest.csv', split='train', variant='raw'):
        # read the whole manifest, then keep only rows matching our split
        df = pd.read_csv(manifest_path)
        self.data = df[df['split'] == split].reset_index(drop=True)
        self.variant = variant

    def __len__(self):
        # PyTorch needs to know how many examples exist
        return len(self.data)

    def __getitem__(self, idx):
        # PyTorch calls this one index at a time to grab a single example
        row = self.data.iloc[idx]
        img = load_and_preprocess(row['filepath'])  # returns a (224,224,3) numpy array

        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(row['label'], dtype=torch.long)

        if self.variant == 'ela':
            ela = compute_ela(row['filepath'])
            aux_tensor = torch.tensor(ela, dtype=torch.float32).permute(2, 0, 1)
            return img_tensor, aux_tensor, label

        if self.variant == 'srm':
            srm = compute_srm(img)
            aux_tensor = torch.tensor(srm, dtype=torch.float32).permute(2, 0, 1)
            return img_tensor, aux_tensor, label

        # 'raw' variant — return a dummy zero-tensor instead of None,
        # since PyTorch's default batching cannot collate None values.
        # SpatialCNN's forward() never actually reads x_aux when
        # variant='raw', so this placeholder has zero effect on results.
        aux_tensor = torch.zeros_like(img_tensor)
        return img_tensor, aux_tensor, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    ds = TamperedImageDataset(split='train', variant='raw')
    print("Number of training images:", len(ds))

    img, aux, label = ds[0]
    print("Image shape:", img.shape)
    print("Aux shape:", aux.shape)
    print("Label:", label)

    train_loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
    images, auxs, labels = next(iter(train_loader))
    print("Batch of images shape:", images.shape)
    print("Batch of aux shape:", auxs.shape)
    print("Batch of labels shape:", labels.shape)