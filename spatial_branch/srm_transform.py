import cv2
import numpy as np

SRM_KERNEL = np.array([
    [-1, 2, -1],
    [ 2, -4, 2],
    [-1, 2, -1]
], dtype=np.float32) / 4.0

def compute_srm(img_rgb):
    """
    Shared SRM extraction function — takes an already-preprocessed RGB
    image (from load_and_preprocess, ImageNet-normalized) and returns
    a (224, 224, 3) SRM-filtered array, ready to be used as an
    auxiliary input to the spatial CNN.
    """
    # IMPORTANT: reverse ImageNet normalization first, same fix that
    # solved the DCT plateau bug — SRM needs real pixel intensities,
    # not normalized ones, or the filter output becomes meaningless
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    img_denorm = np.clip(img_rgb * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
    img_uint8 = (img_denorm * 255).astype(np.uint8)

    # apply the SRM filter to each color channel separately, then
    # recombine — this keeps the auxiliary input 3-channel, matching
    # what the aux_branch conv layer in SpatialCNN expects
    channels = []
    for c in range(3):
        filtered = cv2.filter2D(img_uint8[:, :, c].astype(np.float32), -1, SRM_KERNEL)
        channels.append(filtered)
    srm_image = np.stack(channels, axis=-1)  # (224, 224, 3)

    # normalize to roughly 0-1 range for stable training, same
    # principle as the log-scaling fix you used for DCT
    srm_image = (srm_image - srm_image.mean()) / (srm_image.std() + 1e-8)
    srm_image = np.clip((srm_image + 3) / 6, 0, 1)  # rough rescale to 0-1

    return srm_image.astype(np.float32)