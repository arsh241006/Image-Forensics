import os
import csv
import random

random.seed(42)  # fixed seed — same split for everyone, every time


def collect_images(folder, label):
    images = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
            images.append(
                (os.path.join(folder, fname).replace('\\', '/'), label))
    return images


# adjust these to match your actual CASIA2 folder names exactly
authentic = collect_images('data/CASIA2/Au', 0)
tampered = collect_images('data/CASIA2/Tp', 1)

all_images = authentic + tampered
random.shuffle(all_images)

n = len(all_images)
train_end = int(0.70 * n)
val_end = int(0.85 * n)  # 70% + 15%

rows = []
for i, (path, label) in enumerate(all_images):
    if i < train_end:
        split = 'train'
    elif i < val_end:
        split = 'val'
    else:
        split = 'test'
    rows.append((path, label, split))

with open('data/manifest.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filepath', 'label', 'split'])
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to data/manifest.csv")
print(f"Train: {train_end}, Val: {val_end - train_end}, Test: {n - val_end}")
