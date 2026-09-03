## Preprocessing Configuration (Agreed Week 1)

* **Image size:** 224 × 224
* **Normalization:** ImageNet mean/std using `ResNet18_Weights.DEFAULT.transforms()`
* **Labels:** `0 = Authentic`, `1 = Tampered`
* **Training dataset:** CASIA v2
* **Data split:** 70% Train / 15% Validation / 15% Test
* **Random seed:** `42` for reproducibility
* **Columbia and COVERAGE:** Completely held out from training, validation, and test splits
* **Cross-dataset testing:** Columbia and COVERAGE will be used only in **Phase 5** to evaluate generalization to unseen datasets

### Important

Columbia and COVERAGE must **never be mixed into the CASIA v2 train/validation/test split**. This ensures that Phase 5 measures genuine cross-dataset generalization.

## Branch Feature Vector Convention (agreed Week 2)

- Person A (spatial_branch): SpatialCNN.forward(x, return_features=True) -> (batch, 128)
- Person B (freq_branch): FrequencyCNN.forward(x, return_features=True) -> (batch, 128)
- Fusion (Week 4) concatenates these into a (batch, 256) vector before classification.


## Week 3 — Spatial Branch Auxiliary Technique Comparison

Three variants of the spatial branch (ResNet18-based `SpatialCNN`) were trained and compared on the validation set:

| Variant       | Val Accuracy |
|---------------|--------------|
| Raw RGB only  | 69.6%        |
| RGB + ELA     | 75.8%        |
| RGB + SRM     | 69.2%        |

**Winning variant: RGB + ELA (75.8%)** — selected as the official spatial branch for Week 4 fusion, based on both highest accuracy and stability across training epochs.

- ELA (Error Level Analysis) exposes recompression-quality mismatches between a tampered region and its surroundings.
- SRM (Steganalysis Rich Model) targets a broader category of noise-residual inconsistencies; it performed close to the raw-RGB baseline, suggesting it added limited additional signal on top of raw RGB for this dataset, though it remains a legitimate, documented technique.
- Both auxiliary branches were built using a shared `variant='raw'|'ela'|'srm'` structure in `spatial_branch/model.py`, so all three share the same underlying architecture and 128-dim feature output for fair comparison.

**Model file for fusion:** `spatial_branch/baseline_model_ela.pt ` (feature_dim=128)