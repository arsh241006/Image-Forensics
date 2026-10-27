# Image Forensics — Frontend Design Brief

## Audience and goal

The application is intended for a project demonstration to professors
and classmates, as well as for non-technical users.

The goal is to help users understand the model's prediction quickly
and then explore the available supporting visualizations.

## Application states

The application has four main states:

1. **Upload:** The user selects or drops an image.
2. **Analyzing:** The application waits for the backend response and
   offers a cancel button.
3. **Result:** The prediction, model score, warnings and available
   visualizations are displayed.
4. **Error:** A readable message is shown with an option to try again.

## Layout

- On wide screens, show the image on the left and the prediction and
  evidence cards on the right.
- On narrow screens, use a single-column layout with the verdict first,
  followed by the image and evidence.
- Avoid horizontal scrolling.

## Visual direction

- Use a calm, light theme with a deep teal accent.
- Use a system font so the page does not depend on internet access.
- Use clear typography, generous spacing and consistent cards.
- Use verdict colours together with text and icons, not colour alone.
- Keep colours, spacing and other reusable styles in CSS variables.

## Components

- UploadZone
- AnalysisProgress
- ErrorBanner
- VerdictCard
- ImageCompare
- ManipulationCard
- WarningBanner
- FrequencyPanel
- Footer

## API integration

The frontend sends an image as the `file` field in a
`multipart/form-data` request to `POST /predict`.

It uses the response fields described in `backend/API.md`. The frontend
must handle nullable fields and display backend warning messages directly.

## Wording rules

- Describe the result as a **model prediction**, not a fact.
- Use **model score for tampering**, not certainty.
- Describe copy-move output as evidence or an estimate; do not claim
  that the system has proven a manipulation type.
- Explain that the heatmap is approximate and does not show exact edit
  boundaries.
- Display non-JPEG and other warnings returned by the backend.
- Include an academic-prototype disclaimer.

## Accessibility

- Make controls usable with a keyboard and provide visible focus rings.
- Add meaningful alternative text to images.
- Announce important results and errors to assistive technology.
- Respect reduced-motion preferences.
- Check text and background contrast.