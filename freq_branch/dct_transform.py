import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def _denormalize(img_rgb):
    """
    Reverses the ImageNet normalization applied by load_and_preprocess(),
    bringing pixel values back to a proper 0-1 range before DCT/FFT,
    since compression artifacts are tied to real pixel intensity,
    not a normalized version of it.
    """
    img = img_rgb * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)  # normalization + float error can push slightly outside 0-1
    return img

def extract_dct_features(img_rgb, block_size=8):
    img_denorm = _denormalize(img_rgb)
    img_gray = cv2.cvtColor((img_denorm * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    h, w = img_gray.shape
    h = h - (h % block_size)
    w = w - (w % block_size)
    img_gray = img_gray[:h, :w]

    dct_map = np.zeros_like(img_gray, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img_gray[i:i+block_size, j:j+block_size].astype(np.float32)
            dct_map[i:i+block_size, j:j+block_size] = cv2.dct(block)

    dct_map = np.sign(dct_map) * np.log1p(np.abs(dct_map))
    return dct_map

def extract_fft_features(img_rgb):
    img_denorm = _denormalize(img_rgb)
    img_gray = cv2.cvtColor((img_denorm * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    return magnitude.astype(np.float32)