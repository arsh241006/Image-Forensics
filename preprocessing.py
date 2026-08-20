import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def load_and_preprocess(filepath, size=224, normalize=True):
    """
    Shared preprocessing function — used by BOTH spatial_branch and freq_branch.
    Do not modify without team agreement.
    """
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {filepath}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0

    if normalize:
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

    return img