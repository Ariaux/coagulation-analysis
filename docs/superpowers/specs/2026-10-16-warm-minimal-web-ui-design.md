# Warm Minimal Web UI Design

## Goal

Redesign the existing coagulation-analysis interface so it feels calm, polished,
and easy to use on phones and computers. Preserve all analysis behavior, inputs,
outputs, hosting behavior, and local/offline behavior.

## Visual Direction

- Warm off-white page background with white and pale-sand surfaces.
- Dark brown primary text, muted taupe supporting text, and brick-red actions.
- Soft shadows, generous spacing, rounded cards, and restrained borders.
- Clear typography and compact labels; no decorative imagery or gradients that
  compete with scientific results.
- The product name is `Coagula`; hosted and local status appears as a small badge.

## Page Structure

1. A compact top bar with product name and environment badge.
2. A short hero that explains the single purpose of the tool.
3. A pill-style mode switch for single-image and batch workflows.
4. Single-image mode uses a responsive two-column workspace:
   - left: upload/camera, parameters, primary action, and status;
   - right: 3×3 crop gallery, boundary image, heatmap, and measurements.
5. Batch mode uses the same visual language, with batch upload and result table.
6. Download files sit in a distinct result-actions card after analysis.

On narrow screens, every section becomes one column. The primary action stays
large and easy to tap, tables scroll horizontally, and image results retain their
aspect ratio.

## Interaction and States

- Keep the existing gallery/file and camera inputs mutually exclusive.
- Keep current inset and threshold defaults and ranges.
- Keep all output components and download files unchanged.
- Use short instructional copy and visible empty states before analysis.
- Present status messages in a quiet tinted panel, with enough contrast for
  success and error text.
- Do not expose server-side file paths or folder-opening actions on hosted mode.

## Implementation Boundaries

- Modify only the Gradio composition and `web_styles.css` needed for presentation.
- Do not change grid detection, quantification, batch processing, persistence,
  cleanup, or download formats.
- Preserve the existing hosted cache-expiry policy and local LAN panel.
- Use stable element IDs/classes so styling is testable without brittle generated
  Gradio selectors where possible.

## Verification

- Add tests for new headings, labels, stable element IDs, and hosted/local badges.
- Run the full unit-test suite.
- Launch a local preview and inspect desktop and phone-width layouts.
- Deploy the same verified files to GitHub and the existing Hugging Face Space,
  then confirm the live homepage and Gradio config respond successfully.
