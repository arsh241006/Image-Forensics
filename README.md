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
