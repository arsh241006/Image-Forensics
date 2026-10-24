# Backend API — `POST /predict`

## Request

**Endpoint:** `POST /predict`
**Content-Type:** `multipart/form-data`
**Field name:** `file` (required) — the uploaded image.

**Supported formats:** currently anything OpenCV/PIL can decode (JPEG, PNG, TIFF, etc.) — no explicit format allow-list yet. **Results are most reliable for JPEG images**; the model was trained on JPEG-compressed data (CASIA2), and cross-dataset testing (Columbia, COVERAGE — both uncompressed TIFF) showed accuracy dropping to ~51-54% on non-JPEG images, with the model biased toward predicting "tampered." Non-JPEG uploads return a warning in the response (see below).

**Size limit:** not yet enforced — **planned for Day 4** (error handling + size guard). Do not rely on a specific limit until confirmed.

## Response — `200 OK`

```json
{
  "filename": "example.jpg",
  "label": "tampered",
  "tampered_probability": 0.6918,
  "manipulation_type": "copy-move",
  "manipulation_details": {"consistent_matches": 12},
  "heatmap": "<base64 PNG string>",
  "frequency_view": "<base64 PNG string>",
  "warnings": []
}
```

### Field notes

- **`label`**: `"tampered"` or `"authentic"` — model's predicted class.
- **`tampered_probability`**: float 0-1, softmax probability for the "tampered" class.
- **`manipulation_type`**: `"copy-move"`, `"splicing_or_other"`, or `null` if `label` is `"authentic"` (not computed for authentic predictions).
- **`manipulation_details`**: supporting detail dict for the manipulation type call, or `null` alongside `manipulation_type`.
- **`heatmap`**: base64-encoded PNG, Grad-CAM overlay on the original image, highlighting the region that most influenced the "tampered" prediction. `null` if `label` is `"authentic"`.
- **`frequency_view`**: base64-encoded PNG, FFT magnitude spectrum of the uploaded image. Always present regardless of label.
- **`warnings`**: array of strings, human-readable caveats about this specific result. Can include:
  - Low-confidence warning, when `tampered_probability` is between 0.4 and 0.6.
  - Non-JPEG format warning, when the upload isn't a JPEG.
  - (More may be added as the backend is hardened on Day 4.)

## Error responses

**Not yet implemented** — currently a malformed or unreadable upload may cause an unhandled 500 error rather than a clean error response. **Planned for Day 4.** Do not document specific error codes until confirmed; this section will be updated once built.

## Important caveats for the results page (frontend)

- This is a prototype model (79.2% test accuracy on its own CASIA2-derived test set) — results should be presented as model predictions, not certain fact.
- Results are most reliable for JPEG photos. Non-JPEG uploads carry a backend-generated warning; the frontend should surface any string present in `warnings[]` directly to the user.
- Demo images for presentations should be JPEG only — COVERAGE and Columbia (TIFF) are both out-of-domain for this model and also can't be previewed directly in-browser.

## Status

Last updated: Week 6, Day 3. Size limit, format allow-list, and structured error responses are open items for Day 4 — this doc will be revised once those land.
