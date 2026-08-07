"""Shared settings and crop geometry for analysis results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from grid_detector import DetectionError


BBox: TypeAlias = tuple[int, int, int, int]
MIN_FINAL_CROP_SIDE = 32


@dataclass(frozen=True)
class AnalysisSettings:
    inset_percent: float = 5.0
    no_clot_threshold: float = 60.0
    results_root: Path | None = None

    def validate(self) -> AnalysisSettings:
        if not 0.0 <= self.inset_percent <= 15.0:
            raise ValueError("inset_percent must be between 0 and 15.")
        if not 0.0 <= self.no_clot_threshold <= 255.0:
            raise ValueError("no_clot_threshold must be between 0 and 255.")
        return self


def inset_bbox(
    bbox: BBox,
    percent: float,
    minimum_side: int = MIN_FINAL_CROP_SIDE,
) -> BBox:
    """Return a uniformly inset half-open bounding box."""
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    inset = round(min(width, height) * float(percent) / 100)
    inner_bbox = (left + inset, top + inset, right - inset, bottom - inset)
    inner_width = inner_bbox[2] - inner_bbox[0]
    inner_height = inner_bbox[3] - inner_bbox[1]
    if inner_width < minimum_side or inner_height < minimum_side:
        raise DetectionError(
            "A detected cell is too small after the inner inset. Reduce the inset "
            "or use a higher-resolution image."
        )
    return inner_bbox
