# Hugging Face Fixed 3×3 Web App Design

## Goal

Update the Hugging Face Gradio app so its analysis matches the latest GitHub fixed-fixture workflow instead of the legacy variable-grid rectangular crop.

## User Experience

- The user uploads one photo of the complete fixed 3×3 fixture.
- The app validates the image and detects the fixture automatically.
- The app rectifies small rotation or perspective differences and measures only the nine innermost content squares.
- The app displays the source-image detection overlay, 3×3 heatmap, per-cell measurements, and a concise status message.
- The app provides one ZIP download containing nine cell crops, overlay, heatmap, CSV, and JSON.
- There are no row/column controls or manual rectangular ROI sliders because the supported fixture is always 3×3.

## Architecture

`grid_detector.py` remains the single source of truth for fixture detection. `app.py` calls `detect_inner_squares`, extracts each rectified inner square, calculates the existing ImageJ-equivalent inverted grayscale measurements, and builds web-friendly previews and downloadable artifacts. Shared calculation and rendering behavior should remain compatible with the desktop app.

## Failure Handling

- Missing uploads produce an actionable upload prompt.
- Images smaller than the detector's supported minimum are rejected with a retake message.
- Incomplete, ambiguous, or unreliable fixtures return the detector's actionable error and produce no measurement bundle.
- Temporary files are isolated per request so concurrent web users cannot overwrite one another.

## Output Contract

The table and CSV contain cell number, row, column, intensity statistics, area, confidence, recovery status, source bounding box, and source quadrilateral coordinates. JSON carries the same cell measurements and geometry. The ZIP also contains `cell_01.png` through `cell_09.png`, the overlay, and heatmap.

## Verification

- Reuse the existing detector unit tests from the latest GitHub branch.
- Add focused web-pipeline tests for successful nine-cell output, ZIP contents, and fail-safe invalid-image behavior.
- Run the full relevant test suite before deployment.

## Deployment

Deploy `app.py`, `grid_detector.py`, and the lightweight requirements to the existing Hugging Face Space at `Yilunaria/coagulation-quantification`. The Hugging Face account namespace remains part of the public URL.
