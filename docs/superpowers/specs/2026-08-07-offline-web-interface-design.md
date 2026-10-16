# Offline web interface for fixed nine-grid analysis

## Purpose

Provide a Windows application that opens as a local English-language website,
reuses the existing fixed 3x3 fixture detector, and works without an internet
connection or Python installation. The website must support both immediate
single-image review and unattended batch processing while keeping all images
and results on the user's computer.

This feature also changes two analysis outputs:

1. The heatmap uses a single blue color for measurements classified as no clot
   and a light-red-to-deep-red gradient for measurements classified as clot.
2. Every detected inner square is inset before measurement and export so that
   the black fixture border is excluded from the final crop.

## Confirmed product decisions

- Deployment: a local-only Windows website.
- Startup: double-click `StartWebsite.exe`; the default browser opens
  automatically.
- Network binding: loopback only (`127.0.0.1`), with no cloud service,
  telemetry, account, or image upload.
- Interface language: English.
- Modes: single-image analysis and batch processing.
- Layout: one-page workspace with separate `Single Image` and
  `Batch Processing` tabs.
- Framework: Gradio, backed by the existing Python and OpenCV analysis engine.
- Result storage: automatic local archival under `results/`, plus file and ZIP
  download controls in the browser.
- Batch failure policy: continue processing other images and report each
  failure explicitly.
- Inner inset: 5% by default, adjustable from 0% through 15%.
- Heatmap threshold: user-adjustable on the 0-255 measurement scale, with an
  initial value of 60.
- Heatmap palette: the approved publication palette: muted blue for no clot,
  then light red through deep red for clot.

## Architecture

The packaged application contains four bounded components.

### Windows launcher

`StartWebsite.exe` starts the local Gradio server, selects an available local
port, and opens the default browser. It binds only to `127.0.0.1`. The launcher
window remains open while the site is running; closing it stops the service.
Startup failures must be printed in the launcher and written to the audit log.

### Web interface

The Gradio interface owns file selection, settings, progress, previews, result
tables, and download links. It never implements image analysis. The interface
passes validated paths and settings to a service layer and renders the returned
structured results.

### Analysis service

The service layer adapts the existing `grid_detector` and measurement pipeline
for both desktop and web callers. It is responsible for input validation,
nine-grid detection, final inset geometry, measurement, heatmap generation,
metadata, atomic output publication, and per-image errors. It exposes the same
single-image operation to both UI tabs so the batch path cannot drift from the
single-image path.

### Result archive

The archive layer creates collision-safe result directories, writes all
artifacts atomically, builds downloadable ZIP files, and produces batch summary
and failure tables. Uploaded browser temporaries are removed after the durable
result has been published.

## User interface

### Shared header and controls

The page header reads `Coagulation Analysis` and visibly states
`Local and offline`. Both tabs expose the same two analysis settings:

- `Inner crop inset`, shown as a percentage slider from 0 through 15 with a
  default of 5.
- `No-clot threshold`, shown as a numeric slider from 0 through 255 with a
  default of 60.

Settings are captured at the start of each analysis. Changing a slider after a
run does not silently relabel stored results; the user must run the analysis
again.

### Single Image tab

The user drops or selects one complete fixture image and clicks
`Analyze Image`. A successful result shows:

- the nine final crops in row-major order;
- the source overlay with detected and final crop boundaries;
- the thresholded publication heatmap;
- the nine measurements and confidence values;
- the effective inset and threshold;
- buttons to open the local result folder, download CSV, and download the
  complete result ZIP.

The interface must not imply success until all artifacts have been published.
On failure it shows an actionable message and no partial result links.

### Batch Processing tab

The user selects multiple images and starts one batch with one shared inset and
threshold. The table shows filename, status, detected-cell count, failure
reason, and result location. Each image is processed independently. A failed
image does not prevent later images from running.

The batch completes with a summary table and a ZIP containing all successful
outputs plus the batch summary and failure report.

## Image and crop rules

Accepted input extensions are PNG, JPG, JPEG, BMP, and TIFF. Both image
dimensions must be at least 600 pixels. Existing fixture completeness,
structure, geometry, and confidence checks remain authoritative.

The detector first returns each cell's innermost square using its existing
half-open coordinate contract. The analysis service then computes the final
crop independently for each cell:

```text
inset_pixels = round(min(detected_width, detected_height) * inset_percent / 100)
final_box = detected_box inset by inset_pixels on all four sides
```

The final box remains half-open. The service rejects any inset that produces an
empty or analysis-inadequate crop. The same final box is used for the exported
cell image, measurement, heatmap, overlay, CSV, and JSON. This prevents the UI
preview and scientific data from describing different regions.

The overlay distinguishes the detected inner boundary from the final inset
boundary using different line styles or colors and labels each final crop 1-9.

## Heatmap rules

The scalar is the existing ImageJ-equivalent inverted 8-bit grayscale mean, in
the range 0-255. For threshold `T` and measurement `M`:

- `M <= T`: use the fixed no-clot blue `#3F78B5`.
- `M > T`: interpolate only within the approved red ramp from light red
  `#F6D2CF` through medium red `#D45F62` to deep red `#7E1024`.

The red interpolation domain is fixed from `T` to 255. It is not normalized to
the minimum and maximum of the current image or batch; therefore the same
measurement and threshold always produce the same color across experiments.
Green, yellow, and unrelated intermediate hues are prohibited.

Every heatmap cell displays its row-major index and numeric measurement. The
legend explicitly labels the blue category `No clot`, marks the selected
threshold, and labels the red direction `More clot`.

## Results and provenance

### Single-image directory

Each successful input receives a collision-safe directory under `results/`.
It contains:

- `cell_01.png` through `cell_09.png`;
- a source overlay showing detected and final boundaries;
- the publication heatmap;
- CSV and JSON results;
- a complete result ZIP;
- the per-image audit log or an audit-log reference.

CSV and JSON include the source filename, source dimensions, detector version,
overall and per-cell confidence, recovery flags, detected boxes, final boxes,
inset percentage, inset pixels per cell, threshold, palette version,
measurement method, and final measurements.

### Batch directory

Batch outputs use this structure:

```text
results/
└── batch_YYYYMMDD_HHMMSS_<short-id>/
    ├── sample-001_<filename-key>/
    ├── sample-002_<filename-key>/
    ├── failures.csv
    ├── batch-summary.csv
    ├── batch-metadata.json
    └── batch-results.zip
```

The short identifier and stable filename keys prevent timestamp and duplicate
name collisions. `failures.csv` records the source filename and actionable
failure reason. The batch ZIP includes both successes and failure reports.

## Failure handling

The website provides distinct, actionable messages for:

- unsupported or unreadable files;
- images below 600x600 pixels;
- incomplete or unreliable fixture detection;
- excessive perspective or missing outer edges;
- invalid inset or threshold values;
- final crops that are too small;
- unwritable result directories;
- ZIP creation failures;
- server startup or browser-open failures.

Single-image failure produces no final artifact directory. Batch failure is
isolated to that image. Atomic staging remains mandatory so a crash cannot
publish a directory that looks complete.

Unicode paths and filenames are supported throughout. Console messages must
fall back safely when the active Windows code page cannot represent a path;
UTF-8 logs retain the original characters.

## Security and privacy

- Listen only on loopback, never `0.0.0.0`.
- Disable public sharing and Gradio share links.
- Do not load fonts, scripts, analytics, models, or assets from a CDN.
- Do not make outbound network requests during startup or analysis.
- Treat browser uploads as untrusted input and validate size, extension, and
  successful image decoding.
- Do not expose arbitrary filesystem browsing or result paths outside the
  configured application directories.

## Packaging

The Windows package is a complete extracted folder containing
`StartWebsite.exe` and its runtime files. It requires neither Python nor an
internet connection. GitHub Actions builds the package on Windows after tests
and module compilation succeed, then publishes a versioned ZIP and updates the
documented latest-download link.

The desktop drag-and-drop executable remains available for compatibility. Both
desktop and web entry points call the shared analysis service and must generate
equivalent crops and measurements for identical settings.

## Testing and acceptance

### Unit tests

- Percentage inset rounding and half-open coordinate behavior.
- Rejection of empty or inadequate final crops.
- Threshold boundary: values at and below `T` are blue.
- Red-ramp endpoints and interpolation above `T`.
- Fixed cross-image color mapping.
- Result metadata includes inset, threshold, and palette version.

### Service and UI tests

- Single-image callback returns nine previews and complete download artifacts.
- Slider values reach the analysis service unchanged.
- Batch processing continues after an image failure.
- Batch summaries and ZIPs include success and failure records.
- Duplicate and Unicode filenames remain isolated.
- Failed analyses do not publish partial result directories.

### Packaging tests

- Windows build imports Gradio, the shared analysis service, and
  `grid_detector`.
- `StartWebsite.exe` starts on a clean Windows runner, binds to loopback, and
  returns the application page.
- The packaged application performs a synthetic nine-grid analysis without
  network access.

### Physical-sample acceptance

For the provided straight-on fixture photograph, the website must:

- detect exactly nine cells in row-major order;
- place every final crop strictly inside its detected inner boundary when the
  inset is greater than zero;
- exclude visible black fixture borders from the 5% default crops;
- show no-clot cells only in blue and clot cells only in the approved red ramp;
- generate complete CSV, JSON, overlay, heatmap, crops, audit data, and ZIP;
- produce the same crop geometry and measurements in single and batch modes.

## Out of scope

- Public hosting, domains, accounts, and Chinese regulatory filing.
- Multi-user or laboratory-network access.
- Cloud storage or synchronization.
- Mobile-specific layout.
- Automatic clinical diagnosis or replacement of the user-selected no-clot
  threshold.
- Rewriting the detector in JavaScript or WebAssembly.
