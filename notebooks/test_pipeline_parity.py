import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fusion.pipeline import ForensicsPipeline
from backend.pipeline_utils import prepare_inputs, predict_from_inputs
import pandas as pd

pipeline = ForensicsPipeline()
df = pd.read_csv('data/manifest.csv')
test_rows = df[df['split'] == 'test'].head(20)

mismatches = 0
for _, row in test_rows.iterrows():
    orig_label, orig_prob = pipeline.predict(row['filepath'])

    inputs = prepare_inputs(row['filepath'])
    split_label, split_prob = predict_from_inputs(inputs)

    if orig_label != split_label or abs(orig_prob - split_prob) > 1e-6:
        mismatches += 1
        print(f"MISMATCH on {row['filepath']}: orig=({orig_label},{orig_prob}) split=({split_label},{split_prob})")

print(f"\nMismatches: {mismatches} of {len(test_rows)}")