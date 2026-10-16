# Hugging Face Fixed 3×3 Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Hugging Face Gradio app use the latest GitHub fixed 3×3 inner-square detector and provide complete, safe downloadable results.

**Architecture:** Keep `grid_detector.py` as the detection authority. Replace the legacy variable-grid pipeline in `app.py` with a fixed-grid web adapter that converts Gradio RGB images to OpenCV BGR, detects and measures nine rectified inner squares, renders previews, and creates an isolated ZIP bundle. Test the web adapter independently with synthetic fixtures and detector failures.

**Tech Stack:** Python 3, Gradio, OpenCV, NumPy, standard-library CSV/JSON/ZIP/tempfile, unittest/pytest.

---

### Task 1: Synchronize the detector baseline

**Files:**
- Update from GitHub: `grid_detector.py`
- Update from GitHub: `app_standalone.py`
- Update from GitHub: `tests/test_grid_detector.py`
- Update from GitHub: `tests/test_standalone_pipeline.py`

- [ ] **Step 1: Merge the latest tracked GitHub branch**

Run: `git merge origin/main`

Expected: the 24 upstream commits merge without overwriting untracked handoff files.

- [ ] **Step 2: Run the detector and desktop tests**

Run: `python3 -m pytest tests/test_grid_detector.py tests/test_standalone_pipeline.py -q`

Expected: all selected tests pass.

- [ ] **Step 3: Commit only if the merge did not create a merge commit automatically**

Run: `git status --short --branch`

Expected: the branch contains the upstream detector files and the design commit.

### Task 2: Define the web pipeline with failing tests

**Files:**
- Create: `tests/test_web_pipeline.py`
- Modify: `app.py`

- [ ] **Step 1: Write a successful pipeline contract test**

Create a synthetic 900×900 fixture with nine framed inner squares, call `process_pipeline(image_rgb, "fixture.png")`, and assert that the result has an overlay, heatmap, exactly nine table rows, a real ZIP path, and a success message. Open the ZIP and assert its members are exactly nine `cell_XX.png` crops plus `grid_overlay.png`, `heatmap.png`, `results.csv`, and `results.json`.

- [ ] **Step 2: Write fail-safe contract tests**

Call the same function with `None` and with an incomplete fixture. Assert that both return empty visual/data outputs, no ZIP file, and an actionable error message rather than partial measurements.

- [ ] **Step 3: Run the new tests and confirm the old app fails the contract**

Run: `python3 -m pytest tests/test_web_pipeline.py -q`

Expected: FAIL because the legacy `process_pipeline` accepts row and column counts and does not create the required ZIP contract.

- [ ] **Step 4: Commit the failing contract tests**

Run: `git add tests/test_web_pipeline.py && git commit -m "test: define fixed-grid web pipeline contract"`

Expected: one commit containing only the new tests.

### Task 3: Implement the fixed-grid web adapter

**Files:**
- Modify: `app.py`
- Test: `tests/test_web_pipeline.py`

- [ ] **Step 1: Replace legacy grid detection**

Import `DetectionError` and `detect_inner_squares` from `grid_detector.py`. Convert Gradio RGB input to BGR, reject images whose width or height is below 600 pixels, and call `detect_inner_squares`.

- [ ] **Step 2: Measure the nine detected crops**

For each detected square, crop `detection.rectified[y1:y2, x1:x2]`, compute ImageJ-equivalent inverted 8-bit grayscale statistics, and append cell number, row, column, confidence, recovery flag, source bounding box, and source quadrilateral coordinates.

- [ ] **Step 3: Render browser outputs**

Draw source quadrilaterals and cell numbers on a copy of the uploaded image, create the fixed 3×3 heatmap, convert BGR previews back to RGB, and construct the nine-row table.

- [ ] **Step 4: Build isolated download artifacts**

Use `tempfile.mkdtemp(prefix="coagulation_web_")`; write `cell_01.png` through `cell_09.png`, `grid_overlay.png`, `heatmap.png`, `results.csv`, and `results.json`; then create `coagulation_results.zip` in that request directory. Do not reuse filenames across requests.

- [ ] **Step 5: Return fail-safe errors**

Catch `DetectionError` and return `(None, None, [], None, message)` without creating a result bundle. Apply the same empty contract for missing uploads and undersized images.

- [ ] **Step 6: Run web tests**

Run: `python3 -m pytest tests/test_web_pipeline.py -q`

Expected: all web-pipeline tests pass.

- [ ] **Step 7: Commit the implementation**

Run: `git add app.py && git commit -m "feat: use fixed-grid detector in web app"`

Expected: one implementation commit.

### Task 4: Replace the legacy Gradio interface

**Files:**
- Modify: `app.py`
- Test: `tests/test_web_pipeline.py`

- [ ] **Step 1: Remove unsupported controls**

Remove row/column sliders and manual ROI sliders. Keep one image upload and one primary Analyze button.

- [ ] **Step 2: Update user guidance and outputs**

Explain that the complete fixed 3×3 fixture must be visible and at least 600 pixels in both dimensions. Label overlay, heatmap, results table, and ZIP download clearly. Wire the button directly to `process_pipeline(image, filename)` while deriving a safe default filename when Gradio provides only an array.

- [ ] **Step 3: Add a UI construction smoke test**

Import `app` and assert the module exposes `demo` and the fixed-grid `process_pipeline`, proving interface construction does not require launching a server.

- [ ] **Step 4: Run the web tests**

Run: `python3 -m pytest tests/test_web_pipeline.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the UI**

Run: `git add app.py tests/test_web_pipeline.py && git commit -m "feat: simplify fixed-grid web interface"`

Expected: one UI commit.

### Task 5: Verify and deploy

**Files:**
- Verify: `requirements.txt`
- Verify: `app.py`
- Verify: `grid_detector.py`

- [ ] **Step 1: Run the complete lightweight test suite**

Run: `python3 -m pytest tests/test_grid_detector.py tests/test_standalone_pipeline.py tests/test_web_pipeline.py -q`

Expected: all selected tests pass with no failures.

- [ ] **Step 2: Check formatting and repository state**

Run: `git diff --check && git status --short --branch`

Expected: no whitespace errors; only the pre-existing untracked handoff files remain.

- [ ] **Step 3: Push the updated GitHub branch**

Run: `git push origin main`

Expected: GitHub accepts the new commits.

- [ ] **Step 4: Push the Space files to Hugging Face**

Push `app.py`, `grid_detector.py`, and `requirements.txt` to the existing `Yilunaria/coagulation-quantification` Space repository using the configured Hugging Face credentials.

Expected: the Space rebuild starts successfully.

- [ ] **Step 5: Verify the deployed Space**

Open `https://huggingface.co/spaces/Yilunaria/coagulation-quantification`, confirm the Space is running, and perform one fixed 3×3 fixture analysis or equivalent health check.

Expected: the page loads, reports the fixed 3×3 workflow, and produces the complete result bundle.
