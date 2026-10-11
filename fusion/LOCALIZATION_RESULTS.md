bash

cat << 'EOF' > /home/claude/LOCALIZATION_RESULTS.md
# Grad-CAM Localization Results — Week 5 (Person C: Goldi)

## 1. How Grad-CAM works, and why it's spatial-branch-only

Grad-CAM (Gradient-weighted Class Activation Mapping) produces a visual heatmap showing which regions of an image most influenced a model's prediction. It works by taking the gradient of the predicted class score with respect to the activations in the model's last convolutional layer, using those gradients to weight each feature map by importance, and combining them into a single spatial heatmap.

Grad-CAM is applied only to the **spatial (ELA) branch**, not the fusion model. The spatial branch's ResNet18 backbone retains real spatial structure (a 7×7 grid of features) right up until its final convolutional block. The fusion model, by contrast, only ever receives two already-flattened 128-dimensional feature vectors — there is no spatial grid left to trace a prediction back to. This is a structural limitation of the fusion architecture, not an oversight.

## 2. Methodology

- **Model:** frozen spatial branch (`spatial_branch/baseline_model_ela.pt`), ELA variant, 75.8% standalone validation accuracy.
- **Target layer:** `model.backbone[-2][-1]` — the last BasicBlock of ResNet18's layer4, confirmed by inspecting the backbone's printed structure.
- **Validation set:** all 100 image/mask pairs in the COVERAGE dataset (copy-move forgery, ground-truth tampering masks).
- **Metric:** Intersection over Union (IoU) between the resized, thresholded Grad-CAM heatmap and the ground-truth binary mask. Threshold = 0.5 (a reasonable default, not separately tuned — IoU at other thresholds was not tested).

## 3. Headline result

**Mean IoU across 100 COVERAGE images: 0.140**

| Result | Filename | IoU |
|---|---|---|
| Strongest | `69t.tif` | 0.492 |
| Strongest | `20t.tif` | 0.491 |
| Strongest | `37t.tif` | 0.445 |
| Weakest | `13t.tif` | 0.000 |
| Weakest | `15t.tif` | 0.000 |
| Weakest | `1t.tif` | 0.000 |

## 4. Investigating the low score: a domain-mismatch finding

A mean IoU of 0.140 is low, but not uniformly so — strong individual results (up to 0.49) show the heatmap-to-mask comparison pipeline itself works correctly when there is real overlap to find. This pointed toward a dataset-level cause rather than a code bug.

**Direct test:** we compared the Grad-CAM heatmap for a tampered image, `13t.tif` (IoU 0.000), against its authentic counterpart, `13.tif` — the same scene, with no tampering present.

Both images produced a near-identical heatmap, with the strongest activation (value 1.0) in the same location in both cases, despite one image containing no tampering at all.

**Conclusion:** on COVERAGE, the spatial model's heatmap is driven by visual content in the image (e.g. a bright, high-contrast object) rather than any genuine tampering signal. This is consistent with a known limitation of ELA: it detects artifacts from JPEG recompression, and COVERAGE's images are uncompressed TIFF files with no such recompression history to detect. The model, trained entirely on CASIA2's JPEG-compressed, splicing-heavy data, has no real signal to key off in this format — and falls back to attending to visually prominent image content instead.

## 5. Illustrated examples

- `notebooks/debug_iou_13t.png` — original, heatmap, and ground-truth mask for the weak case (`13t.tif`, IoU 0.000). Heatmap concentrates on a visually prominent fairy figure, not the true tampered region (left side of frame).
- `notebooks/debug_authentic_13.png` — same scene, authentic version (`13.tif`), near-identical heatmap despite no tampering present — the key evidence for the domain-mismatch explanation.
- `notebooks/gradcam_test.png` — a strong, in-domain result on a CASIA2 JPEG splicing example, for comparison: a concentrated, specific hotspot along a real tampering boundary.

## 6. Scope and limitations

This result is specific to **cross-domain evaluation**: an ELA-trained model (JPEG, splicing) tested on COVERAGE (TIFF, copy-move). It should not be read as a general statement about Grad-CAM or ELA's localization ability. A natural follow-up — not yet tested — would be to run the same IoU validation on CASIA2's own tampered images (matching the model's training domain), where stronger localization would be expected given ELA's dependence on JPEG artifacts present in that data.

The 0.5 heatmap threshold used throughout was not tuned; different thresholds would likely produce different mean IoU values and were not explored in this pass.
EOF
echo done
Output

done