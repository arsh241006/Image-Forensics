import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from fusion.pipeline import ForensicsPipeline
from fusion.robustness import load_subset

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
RESULTS_PATH = 'fusion/cross_dataset_results.csv'
SEED = 42


def list_images(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(IMG_EXTS))


def columbia_items():
    # EDIT: replace these two folder names with the real ones from Step 2
    authentic = list_images('data/Columbia Uncompressed Image Splicing Detection/4cam_auth/4cam_auth')
    spliced = list_images('data/Columbia Uncompressed Image Splicing Detection/4cam_splc/4cam_splc')
    return [(p, 0) for p in authentic] + [(p, 1) for p in spliced]


def coverage_items():
    # EDIT if needed: in COVERAGE, forged images are named '<n>t.tif' and
    # originals '<n>.tif'. Confirm against the real filenames from Step 2.
    items = []
    for p in list_images('data/COVERAGE/image'):
        stem = os.path.splitext(os.path.basename(p))[0]
        items.append((p, 1 if stem.endswith('t') else 0))
    return items


def casia_reference_items():
    # same balanced CASIA test subset Arshpreet uses for robustness
    subset = load_subset()
    return list(zip(subset['filepath'], subset['label'].astype(int)))


def sample_balanced(items, n_per_class, seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    for lab in (0, 1):
        group = [it for it in items if it[1] == lab]
        idx = rng.permutation(len(group))[:n_per_class]
        out += [group[i] for i in idx]
    return out


def safe_predict(pipe, path):
    try:
        return pipe.predict(path)
    except Exception as e:
        print('SKIPPED', path, type(e).__name__, e)
        return None


def evaluate(pipe, name, items):
    preds, probs, labels, failed = [], [], [], 0
    for path, label in items:
        result = safe_predict(pipe, path)
        if result is None:
            failed += 1
            continue
        pred, prob = result
        preds.append(pred)
        probs.append(prob)
        labels.append(label)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float('nan')

    return {
        'dataset': name,
        'n': len(labels),
        'n_failed': failed,
        'n_authentic': labels.count(0),
        'n_tampered': labels.count(1),
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'roc_auc': auc,
        'pct_predicted_tampered': float(np.mean(preds)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


DATASETS = {
    'COVERAGE': coverage_items,
    'Columbia': lambda: sample_balanced(columbia_items(), 200),
    'CASIA_reference': casia_reference_items,
}

if __name__ == '__main__':
    pipe = ForensicsPipeline()

    # resume support: skip datasets already saved
    done = set()
    if os.path.exists(RESULTS_PATH):
        done = set(pd.read_csv(RESULTS_PATH)['dataset'])

    for name, get_items in DATASETS.items():
        if name in done:
            continue
        t0 = time.time()
        row = evaluate(pipe, name, get_items())
        write_header = not os.path.exists(RESULTS_PATH)
        pd.DataFrame([row]).to_csv(RESULTS_PATH, mode='a',
                                   header=write_header, index=False)
        print(f"{name:16s} n={row['n']} acc={row['accuracy']:.3f} "
              f"auc={row['roc_auc']:.3f} recall={row['recall']:.3f} "
              f"pred_tampered={row['pct_predicted_tampered']:.2f} "
              f"({(time.time() - t0) / 60:.1f} min)")