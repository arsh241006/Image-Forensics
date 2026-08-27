import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch


class CoverageDataset(Dataset):
    """
    Loads COVERAGE image/mask pairs for localization evaluation.

    Filename pattern found in this dataset copy:
        image/  -> N.tif   (authentic), Nt.tif  (tampered)
        mask/   -> Ncopy.tif, Nforged.tif, Npaste.tif  (three masks per image)

    We use '{N}forged.tif' as the ground-truth tampering mask, since 'forged'
    is the standard COVERAGE convention for the evaluation mask.
    VERIFY this with the Day 3-5 visual sanity check (Step 4) before trusting it -
    if the white region doesn't line up with the visibly tampered area, try
    '{N}paste.tif' instead.
    """

    def __init__(self, image_dir='data/COVERAGE/image', mask_dir='data/COVERAGE/mask', size=224):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size

        # find all tampered images - filenames like '10t.tif'
        all_files = os.listdir(image_dir)
        self.tampered_files = [f for f in all_files if f.lower().endswith('t.tif')]

    def __len__(self):
        return len(self.tampered_files)

    def __getitem__(self, idx):
        img_filename = self.tampered_files[idx]
        img_path = os.path.join(self.image_dir, img_filename)

        # '10t.tif' -> base_id '10'
        base_id = img_filename[:-len('t.tif')]
        mask_filename = f"{base_id}forged.tif"
        mask_path = os.path.join(self.mask_dir, mask_filename)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"No mask found for {img_filename} at {mask_path}")
        mask = cv2.resize(mask, (self.size, self.size))
        mask = (mask > 127).astype(np.float32)  # binarize: white=1 (tampered), black=0

        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        return img_tensor, mask_tensor, img_filename
