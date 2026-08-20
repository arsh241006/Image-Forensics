import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# test_setup.py (run once, don't need to commit)
from preprocessing import load_and_preprocess
import pandas as pd
import os

# check 1: preprocessing works and gives expected shape/range
df = pd.read_csv('data/manifest.csv')
sample_path = df.iloc[0]['filepath']
img = load_and_preprocess(sample_path)
print("Shape:", img.shape)          # expect (224, 224, 3)
print("Min/Max:", img.min(), img.max())  # expect roughly -2 to 2

# check 2: all manifest files exist locally
missing = [f for f in df['filepath'] if not os.path.exists(f)]
print(f"Missing files: {len(missing)} out of {len(df)}")