# Inner-square cropping experiments

## Research question

How accurately and robustly do three cropping strategies recover all nine inner
content squares from the fixed 3×3 fixture?

The three primary methods are:

1. **Fixed ratio** — locate and rectify the outer fixture, then use the
   calibrated 22%/78% cell template and its safe content inset without local
   edge refinement.
2. **Contour only** — rectify the fixture and select a plausible near-square
   contour independently in each cell, without shared-grid recovery.
3. **Hybrid** — use the normal detector, including rectification, local edge
   refinement, grid validation, and recovery of at most one globally unique
   weak inner edge. Structural or multiple-edge failures are never recovered.

## Reference labels

Real-image reference boxes must be labeled manually, independently of every
evaluated detector. Label cells 1–9 in row-major order. Never use the output of
the hybrid or any other evaluated method as reference truth.

Run the local annotation tool:

```bash
python -m research.annotate_inner_squares path/to/image.png \
  --output path/to/image.annotations.json
```

Click the top-left and then bottom-right content corner for each cell.
Backspace undoes the current point or previous box, Enter saves only after nine
valid boxes, and Escape cancels. The tool does not create automatic prelabels
and does not use a network service.

For real data, keep original images under an untracked `research/real_data/`
directory and store reviewed annotation JSON beside each image. Supply a UTF-8
CSV manifest with the exact columns
`case,image,annotations,condition,level`; image and annotation paths are
relative to the manifest:

```csv
case,image,annotations,condition,level
photo-001,images/photo-001.png,labels/photo-001.json,lighting,normal
```

Case IDs must be nonempty and unique. Each image must decode successfully and
be at least 600×600 pixels. Each annotation must name the image basename
exactly, retain `annotator: "manual"`, and contain exactly nine valid,
row-major boxes. The evaluator rejects missing, malformed, mismatched,
non-manual, or duplicate inputs; algorithm-generated truth is prohibited.

Two people should independently review a representative subset of labels. Any
disagreement should be resolved before evaluation, with the final reviewed
annotation retained as the reference.

## Perturbation matrix

The synthetic robustness matrix uses seed `20260730`.

| Condition | Levels |
|---|---|
| Rotation | −5°, −3°, +3°, +5° |
| Brightness multiplier | 0.70, 0.85, 1.15, 1.30 |
| Scale | 0.85, 1.15 (with padding when needed to remain at least 600 px) |
| Additive Gaussian noise σ | 8, 16 |
| Content pattern | empty, alternating filled/empty |

Geometric transforms are applied to both the image and manual/synthetic truth
coordinates. Photometric and content perturbations do not alter truth boxes.

## Metrics

Both the manual-image and synthetic evaluations score the box returned directly
by each evaluation method. They do not apply the production
`AnalysisSettings.inset_percent` step afterward. For the hybrid method this is
the detector box (`InnerSquare.source_bbox`), before the website's final inset
box is calculated. The fixed-ratio and contour-only methods retain their own
method-specific safety margins described in the implementation, but receive no
additional shared production inset. Therefore these results must not be
described as metrics for the website's final 5% inset crops.

Every reported experiment must identify the box stage used for metrics as
either `detected/evaluator-returned` or `final inset`, and must report the exact
`inset_percent` and `no_clot_threshold` settings alongside its results. For the
evaluation commands in this document, report `inset_percent: not applied`
and `no_clot_threshold: not applied`; the threshold does not affect cropping
metrics. If website or desktop result artifacts are evaluated instead, report
their recorded numerical values (defaults: 5% inset and threshold 60). Do not
mix detected-box and final-inset measurements in one summary.

For an `xyxy` box \(A\) and reference box \(T\):

- Intersection over union: \(IoU(A,T)=|A∩T|/|A∪T|\).
- Boundary error:
  \((|x1_A-x1_T|+|y1_A-y1_T|+|x2_A-x2_T|+|y2_A-y2_T|)/4\), in pixels.
- Cell success: `IoU >= 0.85`.
- All-nine success: all nine cells in the image are successful.
- Measurement error: mean absolute difference between the nine predicted and
  reference crop means after ImageJ-equivalent 8-bit grayscale conversion and
  inversion.
- Runtime: median wall-clock milliseconds from three detector-only
  `perf_counter` samples after one untimed warm-up.

Detection failures are recorded as failed rows rather than removed.

## Ablations

The optional ablation run compares the full hybrid detector with:

- rectification disabled;
- local edge refinement disabled;
- grid validation/recovery disabled.

Run the primary synthetic evaluation:

```bash
python -m research.evaluate_cropping --synthetic \
  --output research/results/primary-new
```

Run it with ablations:

```bash
python -m research.evaluate_cropping --synthetic \
  --output research/results/ablations-new --ablations
```

Run independently labeled images, with or without `--ablations`:

```bash
python -m research.evaluate_cropping \
  --manifest research/real_data/manifest.csv \
  --output research/results/manual-new
```

The output directory must be new or empty. The evaluator refuses a nonempty
directory and never deletes previous results; choose a fresh name for every run.

## Expected artifacts

Each output directory contains:

- `per_image_results.csv` — one row per case and method;
- `summary.json` — complete metrics and counts by method, by condition, and
  separately for every method/condition pair. IoU and success rates use all
  rows, with detection failures contributing zero. Boundary and measurement
  error means use successful detections because those values are undefined for
  failures. Plots include failed rows at IoU zero, matching the summaries.
  Runtime is the median of three detector-only repetitions after one untimed
  warm-up; it includes failed detector calls and excludes metric and plotting
  work. The method evaluated first rotates deterministically between cases to
  counterbalance order effects;
- `method_comparison.png` — method-level IoU comparison;
- `robustness_by_condition.png` — condition-level robustness comparison;
- `failures/` — research-only truth/prediction comparison overlays for failed
  methods and incomplete all-nine results. These are evaluation artifacts, not
  partial production results.

## Results

Alongside any completed table, state whether the source is manual or synthetic,
the metric box stage, the inset percentage, and the no-clot threshold. A value
that was not applied must be written explicitly as `not applied`, not omitted.

| Method | Mean IoU | Mean boundary error | All-nine success | Measurement MAE | Runtime |
|---|---:|---:|---:|---:|---:|
| Fixed ratio | — | — | — | — | — |
| Contour only | — | — | — | — | — |
| Hybrid | — | — | — | — | — |

Do not infer, estimate, or fill real-data results without independently reviewed
manual labels. Synthetic checks demonstrate reproducibility and software
behavior; they are not substitutes for real experiment values.
