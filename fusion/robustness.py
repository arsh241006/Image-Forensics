import os
import time
import tempfile
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fusion.pipeline import ForensicsPipeline
 
SEED = 42
N_PER_CLASS = 200
RESULTS_PATH = 'fusion/robustness_results.csv'
 
 
def load_subset():
    df = pd.read_csv('data/manifest.csv')
    test = df[df['split'] == 'test']
    parts = [test[test['label'] == lab].sample(n=N_PER_CLASS, random_state=SEED)
             for lab in (0, 1)]
    return pd.concat(parts).reset_index(drop=True)
 
 
def add_noise(im, sigma, seed):
    arr = np.asarray(im).astype(np.float32)
    rng = np.random.default_rng(seed)
    arr = np.clip(arr + rng.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)
 
 
# each condition returns (perturbed_image, jpeg_quality_to_save_at)
# 'original' is None: the untouched file is used as-is
CONDITIONS = {
    'original':       None,
    'control_q95':    lambda im, s: (im, 95),   # re-save only: isolates the cost of re-saving
    'jpeg_q90':       lambda im, s: (im, 90),
    'jpeg_q70':       lambda im, s: (im, 70),
    'jpeg_q50':       lambda im, s: (im, 50),
    'resize_0.5':     lambda im, s: (im.resize((max(1, im.width // 2), max(1, im.height // 2))), 95),
    'resize_1.5':     lambda im, s: (im.resize((int(im.width * 1.5), int(im.height * 1.5))), 95),
    'noise_5':        lambda im, s: (add_noise(im, 5, s), 95),
    'noise_15':       lambda im, s: (add_noise(im, 15, s), 95),
    'brightness_+20': lambda im, s: (ImageEnhance.Brightness(im).enhance(1.2), 95),
    'brightness_-20': lambda im, s: (ImageEnhance.Brightness(im).enhance(0.8), 95),
}
 
 
def run_condition(pipe, subset, name, fn):
    preds, labels = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, row in subset.iterrows():
            path = row['filepath']
            if fn is not None:
                im = Image.open(path).convert('RGB')
                im2, quality = fn(im, i)
                path = os.path.join(tmp, 'perturbed.jpg')
                im2.save(path, 'JPEG', quality=quality)
            pred, prob = pipe.predict(path)
            preds.append(pred)
            labels.append(int(row['label']))
    return {
        'condition': name,
        'n': len(labels),
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'pct_predicted_tampered': float(np.mean(preds)),
    }
 
 
if __name__ == '__main__':
    subset = load_subset()
    pipe = ForensicsPipeline()
 
    # resume support: skip conditions already saved, so a crash loses little
    done = set()
    if os.path.exists(RESULTS_PATH):
        done = set(pd.read_csv(RESULTS_PATH)['condition'])
 
    for name, fn in CONDITIONS.items():
        if name in done:
            continue
        t0 = time.time()
        row = run_condition(pipe, subset, name, fn)
        write_header = not os.path.exists(RESULTS_PATH)
        pd.DataFrame([row]).to_csv(RESULTS_PATH, mode='a', header=write_header, index=False)
        print(f"{name:16s} acc={row['accuracy']:.3f} f1={row['f1']:.3f} "
              f"pred_tampered={row['pct_predicted_tampered']:.2f} "
              f"({(time.time() - t0) / 60:.1f} min)")
