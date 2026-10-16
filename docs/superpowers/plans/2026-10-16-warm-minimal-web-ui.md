# Warm Minimal Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current plain Gradio presentation with the approved warm-minimal C direction while preserving every analysis workflow and hosted/local safety behavior.

**Architecture:** Keep the analysis controllers and component event wiring unchanged. Restructure only the Gradio composition in `app.py` with stable IDs/classes and replace `web_styles.css` with a responsive design system; tests assert the public UI contract and stylesheet tokens.

**Tech Stack:** Python 3.12, Gradio 6.16, CSS, `unittest`

---

## File Map

- `app.py`: page copy, component grouping, stable element IDs/classes, and hosted/local environment badge.
- `web_styles.css`: warm-minimal colors, typography, cards, controls, results, downloads, and responsive behavior.
- `tests/test_web_controller.py`: serialized Gradio UI contract and stylesheet regression coverage.

### Task 1: Define and implement the new page structure

**Files:**
- Modify: `tests/test_web_controller.py`
- Modify: `app.py`

- [ ] **Step 1: Write failing UI-contract tests**

Add tests that build hosted and local apps, then assert the serialized config contains `Coagula`, `Image analysis`, `Clear results. Confident decisions.`, `Single image`, `Batch processing`, `Upload sample`, `Analysis settings`, `Detected cells`, `Visual results`, `Measurements`, and `Download results`. Assert hosted mode contains `Online workspace`, local mode contains `Local workspace`, and the expected stable IDs are present.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
./.venv/bin/python -m unittest tests.test_web_controller.WebApplicationTests.test_warm_minimal_ui_contract tests.test_web_controller.WebApplicationTests.test_environment_badge_matches_runtime
```

Expected: failure because the new copy and IDs are absent.

- [ ] **Step 3: Implement the new Gradio composition**

Add a compact HTML product bar and hero in `create_app()`. Give the tabs, input cards, result cards, settings groups, action buttons, status panels, and download rows stable IDs or classes. Keep the existing components, callbacks, labels needed by controller tests, output ordering, hosted states, local folder buttons, LAN panel, and cache policy unchanged.

- [ ] **Step 4: Run focused tests**

Run the two new tests plus existing hosted/local and button dependency tests. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_web_controller.py
git commit -m "feat: restructure warm minimal web interface"
```

### Task 2: Apply the approved visual system

**Files:**
- Modify: `tests/test_web_controller.py`
- Modify: `web_styles.css`

- [ ] **Step 1: Write a failing stylesheet regression test**

Read `web_styles.css` and assert it defines the approved tokens `--canvas: #fbf7ef`, `--ink: #2c251e`, `--accent: #bd4b33`, responsive breakpoints, and selectors for `#product-bar`, `#hero-card`, `#single-workspace`, `#batch-workspace`, `.workspace-card`, `.result-card`, `.download-card`, and `.primary-action`.

- [ ] **Step 2: Run the stylesheet test and verify failure**

Run:

```bash
./.venv/bin/python -m unittest tests.test_web_controller.WebApplicationTests.test_stylesheet_implements_warm_minimal_design
```

Expected: failure because the old navy/red stylesheet lacks the new tokens and selectors.

- [ ] **Step 3: Replace the stylesheet**

Implement the approved warm off-white canvas, white/sand surfaces, brown text, brick-red primary actions, pill tabs, rounded cards, subtle shadows, styled file drops/sliders/tables/galleries, visible keyboard focus, and single-column mobile layout. Keep QR images crisp and tables horizontally scrollable.

- [ ] **Step 4: Run the stylesheet and UI tests**

Run all `WebApplicationTests`. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add web_styles.css tests/test_web_controller.py
git commit -m "style: apply warm minimal responsive design"
```

### Task 3: Verify, review, publish, and inspect production

**Files:**
- Verify: `app.py`, `web_styles.css`, `tests/test_web_controller.py`

- [ ] **Step 1: Run complete verification**

Run:

```bash
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
git diff --check
```

Expected: all tests pass and the diff check is empty.

- [ ] **Step 2: Inspect desktop and mobile layouts**

Launch the app locally, open it in a browser, and inspect a desktop viewport and a phone-width viewport. Verify no horizontal page overflow, the primary workflow appears above results on mobile, and all output components remain visible.

- [ ] **Step 3: Request a code review and resolve any Critical or Important finding**

Review only the UI commits against the approved spec. Re-run focused and full verification after any correction.

- [ ] **Step 4: Push GitHub**

```bash
git push origin main
```

Expected: `main` updates successfully.

- [ ] **Step 5: Deploy the verified runtime files to the existing Space**

Create one Hugging Face Space commit containing `app.py`, `web_styles.css`, and any other changed runtime file. Wait for runtime stage `RUNNING`.

- [ ] **Step 6: Verify production**

Request `/` and `/config` from `https://yilunaria-coagulation-quantification.hf.space`. Expected: HTTP 200 responses and config containing `Coagula`, the new mode labels, and the hosted environment badge.
