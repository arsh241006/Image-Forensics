import cv2
import numpy as np
from collections import Counter


# ============================================================
# COPY-MOVE DETECTION: ORB self-matching with shift consistency
# ============================================================
def _check_copy_move(image_path, match_threshold=5, max_features=5000,
                     match_ratio=0.85, min_distance=30, shift_bin=8):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(img, None)

    if descriptors is None or len(keypoints) < 3:
        return False, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors, descriptors, k=3)

    shifts = []
    for group in matches:
        if len(group) < 3:
            continue
        m0, m1, m2 = group
        if m0.queryIdx != m0.trainIdx:
            continue
        if m1.distance < match_ratio * m2.distance:
            p1 = np.array(keypoints[m1.queryIdx].pt)
            p2 = np.array(keypoints[m1.trainIdx].pt)
            # ignore matches between nearby keypoints (local texture)
            if np.linalg.norm(p1 - p2) < min_distance:
                continue
            shift = p2 - p1
            # A->B and B->A describe the same copy; normalize direction
            if shift[0] < 0 or (shift[0] == 0 and shift[1] < 0):
                shift = -shift
            shifts.append(shift)

    if not shifts:
        return False, 0

    # a real copy-move: many matches agree on ONE shift vector
    bins = Counter(tuple(np.round(s / shift_bin).astype(int)) for s in shifts)
    consistent_matches = max(bins.values())
    return consistent_matches >= match_threshold, consistent_matches


# ============================================================
# SPLICING DETECTION: DCT block inconsistency (cluster-based)
# ============================================================
def detect_splicing_region(image_path, block_size=8, z_threshold=3.5, min_blocks=12):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = img.shape
    h = h - (h % block_size)
    w = w - (w % block_size)
    img = img[:h, :w]

    rows, cols = h // block_size, w // block_size
    energy_map = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            block = img[i*block_size:(i+1)*block_size,
                        j*block_size:(j+1)*block_size].astype(np.float32)
            dct_block = cv2.dct(block)
            energy_map[i, j] = np.sum(
                np.abs(dct_block)) - np.abs(dct_block[0, 0])

    # robust z-score: median/MAD instead of mean/std
    median = np.median(energy_map)
    mad = np.median(np.abs(energy_map - median)) + 1e-8
    robust_z = 0.6745 * (energy_map - median) / mad

    flagged = (np.abs(robust_z) > z_threshold).astype(np.uint8)

    # a spliced region is a connected PATCH of unusual blocks,
    # not one stray block
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        flagged, connectivity=8)
    largest_cluster = int(
        stats[1:, cv2.CC_STAT_AREA].max()) if num_labels > 1 else 0

    return largest_cluster >= min_blocks, float(largest_cluster)


# ============================================================
# COMBINED THREE-WAY ESTIMATE
# ============================================================
def estimate_manipulation_type(image_path, match_threshold=10,
                               z_threshold=3.5, min_blocks=12,
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

    is_copy_move, consistent_matches = _check_copy_move(
        image_path, match_threshold)
    has_splice_region, cluster_blocks = detect_splicing_region(
        image_path, z_threshold=z_threshold, min_blocks=min_blocks
    )

    if is_copy_move:
        return "copy-move", {"consistent_matches": consistent_matches}
    elif has_splice_region:
        return "splicing", {"cluster_blocks": cluster_blocks}
    else:
        return "uncertain", {"consistent_matches": consistent_matches,
                             "cluster_blocks": cluster_blocks}
