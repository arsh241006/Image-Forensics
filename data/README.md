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





## Columbia Uncompressed Image Splicing Detection Dataset

The **Columbia Uncompressed Image Splicing Detection Dataset** is a benchmark dataset developed by Columbia University's DVMM Laboratory for **image splicing detection and digital image forensics**.

### Dataset Statistics

| Property         | Details                                            |
| ---------------- | -------------------------------------------------- |
| Total Images     | 363                                                |
| Authentic Images | 183                                                |
| Spliced Images   | 180                                                |
| Image Type       | Uncompressed color images                          |
| Forgery Type     | Image Splicing                                     |
| Cameras          | Canon G3, Nikon D70, Canon EOS 350D, Kodak DCS 330 |

The dataset contains:

* `4cam_auth` – 183 authentic images
* `4cam_splc` – 180 spliced images

### Role in This Project

**CASIA** is used for model training, while **Columbia** is used as an **external testing dataset**. This evaluates whether the trained model can generalize to images from a different dataset.

* **Class 0:** Authentic
* **Class 1:** Spliced/Tampered

Performance is evaluated using **Accuracy, Precision, Recall, F1-Score, and Confusion Matrix**.

### Limitations

The dataset is relatively small and mainly contains indoor scenes. It also lacks modern post-processing effects such as JPEG compression, resizing, and social-media transformations.

### Source

**Kaggle:** [Columbia Dataset – Kaggle](https://www.kaggle.com/datasets/shriya0/columbia?resource=download&utm_source=chatgpt.com)

**Original:** Columbia University DVMM Laboratory.


## COVERAGE

**Source:** [COVERAGE — A Novel Database for Copy-Move Forgery Detection](https://github.com/wenbihan/coveragedataset)

### Folder Structure

```text
COVERAGE/
├── image/    # N.tif = authentic, Nt.tif = tampered
├── mask/     # Ncopy.tif, Nforged.tif, Npaste.tif (3 masks per image)
└── label/    # .mat metadata files (FEA, fPSNR, TFlabel, Tlevel) — not used by loader
```

### Dataset Details

| Folder | Description | Images |
|---|---|---|
| `image/` | Authentic + tampered (copy-move) | 100 pairs |
| `mask/` | Ground-truth tampering masks | 100 |

### Notes

* Mask suffix used for ground truth: **`Nforged.tif`** — verified visually
  against `notebooks/coverage_sanity_check.png`; white region correctly
  highlights the pasted/duplicated object.
* `copy.tif` and `paste.tif` are also present but not used by the current loader.
* No `readme.txt` shipped with this specific download — verify filenames
  with `Get-ChildItem`/`ls` before assuming this pattern if you re-download.