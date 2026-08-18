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
