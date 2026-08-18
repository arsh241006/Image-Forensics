## Readme for dataset

# Dataset Information


## CASIA 2.0

**Source:** [Kaggle – CASIA 2.0 Image Tampering Detection Dataset](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset?resource=download)

### Folder Structure

```text
CASIA2/
├── Au/                     # Authentic images
├── Tp/                     # Tampered images
└── CASIA 2 Groundtruth/    # Ground-truth masks
```

### Dataset Details

| Folder                 | Description          | Images |
| ---------------------- | -------------------- | -----: |
| `Au/`                  | Authentic/original   |  7,491 |
| `Tp/`                  | Tampered/manipulated |  5,123 |
| `CASIA 2 Groundtruth/` | Tampering masks      |      — |

**Total images:** 12,614

### Notes

* `Au` → authentic images (**label 0**)
* `Tp` → tampered images (**label 1**)
* Groundtruth masks identify manipulated regions.
* Image sizes and formats are not uniform.
* Tampered filenames contain information about manipulation/source images.
* Groundtruth is useful for **localization/segmentation**, but not required for basic binary classification.

> The dataset is downloaded from Kaggle; the complete dataset is not included in this repository.








# COVERAGE Dataset

**COVERAGE (Copy-Move Forgery Database with Similar but Genuine Objects)** is a dataset for **copy-move forgery detection and localization**.

### Dataset

* **100 original + 100 forged image pairs = 200 images**
* Image format: `.tif`
* `image/` — Original and forged images
* `mask/` — Copy, paste/SGO, and forged-region masks
* `label/` — Tampering type and other annotations
* `readme.txt` — Original dataset documentation

### Tampering Types

1. Rotation
2. Scaling
3. Translation
4. Illumination
5. Free-form
6. Combination

### Key Feature

Contains **Similar but Genuine Objects (SGOs)**, making the dataset useful for testing false positives caused by natural self-similarity.

### Source / Download

**Official GitHub:**
https://github.com/wenbihan/coverage

The actual dataset is hosted through the download link provided in the repository.

### Citation

B. Wen et al., **"COVERAGE - A Novel Database for Copy-Move Forgery Detection," ICIP, 2016.**

### Usage

For **non-commercial research only**. Do not upload the dataset itself to our GitHub repository.





