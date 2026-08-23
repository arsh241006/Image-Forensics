import cv2
import numpy as np

def extract_dct_features(img_rgb, block_size=8):
    """
    Takes a preprocessed RGB image (already resized to 224x224 by
    load_and_preprocess) and returns its block-DCT representation.
    """
    # convert to grayscale for DCT — luminance is where compression artifacts live
    img_gray = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    h, w = img_gray.shape
    h = h - (h % block_size)
    w = w - (w % block_size)
    img_gray = img_gray[:h, :w]

    dct_map = np.zeros_like(img_gray, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img_gray[i:i+block_size, j:j+block_size].astype(np.float32)
            dct_map[i:i+block_size, j:j+block_size] = cv2.dct(block)
    return dct_map  # shape: (224, 224) roughly, same spatial size as input

def extract_fft_features(img_rgb):
    """
    Takes a preprocessed RGB image and returns its FFT magnitude spectrum.
    """
    img_gray = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    return magnitude.astype(np.float32)