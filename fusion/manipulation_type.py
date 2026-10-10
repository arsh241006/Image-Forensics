import cv2
import numpy as np


# ============================================================
# COPY-MOVE DETECTION — Arshpreet (ORB self-matching)
# ============================================================
def _check_copy_move(image_path, match_threshold=8, max_features=2000, match_ratio=0.75):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(img, None)

    if descriptors is None or len(keypoints) < 2:
        return False, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors, descriptors, k=3)

    good_matches = []
    for group in matches:
        if len(group) < 3:
            continue
        m0, m1, m2 = group
        if m0.queryIdx != m0.trainIdx:
            continue
        if m1.distance < match_ratio * m2.distance:
            if m1.queryIdx != m1.trainIdx:
                good_matches.append(m1)

    match_count = len(good_matches)
    return match_count >= match_threshold, match_count


# ============================================================
# SPLICING DETECTION — Anshika (DCT block inconsistency)
# ============================================================
def detect_splicing_region(image_path, block_size=8, std_threshold=2.0):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = img.shape
    h = h - (h % block_size)
    w = w - (w % block_size)
    img = img[:h, :w]

    energies = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size].astype(np.float32)
            dct_block = cv2.dct(block)
            ac_energy = np.sum(np.abs(dct_block)) - np.abs(dct_block[0, 0])
            energies.append(ac_energy)

    energies = np.array(energies)
    deviations = np.abs(energies - energies.mean()) / (energies.std() + 1e-8)
    max_deviation = deviations.max()

    return bool(max_deviation > std_threshold), float(max_deviation)


# ============================================================
# COMBINED THREE-WAY ESTIMATE
# ============================================================
def estimate_manipulation_type(image_path, match_threshold=8, std_threshold=2.0,
                               use_filename_hint=False):
    """
    use_filename_hint: CASIA-only shortcut that reads the label from the
    filename. Keep False for any real evaluation.
    """
    if use_filename_hint:
        filename = image_path.replace("\\", "/").split("/")[-1]
        if filename.startswith("Tp_S_"):
            return "copy-move", {"source": "filename_hint"}
        if filename.startswith("Tp_D_"):
            return "splicing", {"source": "filename_hint"}

    is_copy_move, match_count = _check_copy_move(image_path, match_threshold)
    has_splice_region, deviation = detect_splicing_region(image_path, std_threshold=std_threshold)

    if is_copy_move:
        return "copy-move", {"match_count": match_count}
    elif has_splice_region:
        return "splicing", {"deviation": deviation}
    else:
        return "uncertain", {"match_count": match_count, "deviation": deviation}