# Frontend Handover — Week 6

## Completed

- Connected the frontend to the real backend `/predict` endpoint.
- Added image upload, analyzing, cancellation and error states.
- Added a verdict card showing the model's tampering score.
- Added a display for the uploaded image and backend warnings.
- Added an academic-prototype disclaimer.
- Added a responsive results layout and checked it on a phone-sized screen.
- Checked the keyboard flow and text contrast.
- Added frontend setup instructions to the root README.
- Captured screenshots for the report.

## Visual polish for Week 7

Please review the spacing, typography and overall consistency of the upload,
loading, error and results screens. The main frontend styles are in
`frontend/src/index.css`.

## Open items

- The heatmap view currently shows “Heatmap is not available yet.”
- Heatmap, copy-move and frequency views depend on the backend providing
  the corresponding data. Confirm the response format with Goldi.
- Add a heatmap screenshot to the report when the feature is available.
- Check how backend warning values should be displayed if they are codes
  rather than readable sentences.