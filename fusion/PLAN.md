# Fusion Plan — Week 3 Wrap-up (Person C: Goldi)

## Ownership
- **Arshpreet + Anshika** — fusion data pipeline (`fusion/dataset.py`, `fusion/extract_features.py`)
- **Goldi + Aastha** — fusion training loop + evaluation

## Winning branch variants (confirmed Day 5)
| Branch | Variant | Val Accuracy | Checkpoint |
|---|---|---|---|
| Spatial | ELA | 75.8% | `spatial_branch/baseline_model_ela.pt` |
| Frequency | — | ~64.7% | `freq_branch/baseline_model.pt` |

## Confirmed interfaces
- Both branch models expose `return_features=True`, each returning a `(batch, 128)` vector.
- Spatial: `SpatialCNN.forward(x_rgb, x_aux, return_features=True)` — `x_aux` is the ELA image, required since ELA won.
- Frequency: `FrequencyCNN.forward(x_dct, return_features=True)` — `x_dct` is a `(1, 224, 224)` DCT map from `extract_dct_features()`.

## Fusion dataset
`fusion/dataset.py` (already written and verified by Aastha on a real batch) returns:
```
img_tensor   (3, 224, 224)   — raw RGB
ela_tensor   (3, 224, 224)   — from compute_ela(filepath)
dct_tensor   (1, 224, 224)   — from extract_dct_features(img)
label        scalar
```

## Fusion model
`fusion/fusion_model.py` (written by Aastha, reviewed Day 3):
```
concat([spatial_feat(128), freq_feat(128)]) -> Linear(256,64) -> ReLU -> Dropout(0.3) -> Linear(64,2)
```

## Branch loading — verified working (Day 5)
`notebooks/test_branch_loading.py` confirms:
- Spatial branch loads OK, feature shape (2, 128)
- Frequency branch loads OK, feature shape (2, 128)

## Open items for Week 4
- Confirm whether `fusion/extract_features.py` caches features to disk or computes them live during training — affects how `train_fusion.py` should be structured.
- Frequency checkpoint file is unusually small (132 KB) — plausible given it's a lightweight 3-layer CNN, but worth a sanity check on predictions once training starts.
- Write `fusion/train_fusion.py` (Goldi + Aastha) — frozen branches in eval mode, only FusionModel's params trainable, CrossEntropyLoss + Adam.
- Write evaluation script — accuracy, confusion matrix, compare against branch baselines (75.8% / 64.7%) to check fusion actually helps.