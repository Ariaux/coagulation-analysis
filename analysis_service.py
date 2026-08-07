"""Shared settings and crop geometry for analysis results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from grid_detector import DetectionError


BBox: TypeAlias = tuple[int, int, int, int]
MIN_FINAL_CROP_SIDE = 32
NO_CLOT_BLUE_RGB = (63, 120, 181)
LIGHT_RED_RGB = (246, 210, 207)
MEDIUM_RED_RGB = (212, 95, 98)
DEEP_RED_RGB = (126, 16, 36)
PALETTE_VERSION = "publication-blue-red-v1"


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
    inset_percent: float,
    minimum_side: int = MIN_FINAL_CROP_SIDE,
) -> BBox:
    """Return a uniformly inset half-open bounding box."""
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    inset = round(min(width, height) * float(inset_percent) / 100)
    inner_bbox = (left + inset, top + inset, right - inset, bottom - inset)
    inner_width = inner_bbox[2] - inner_bbox[0]
    inner_height = inner_bbox[3] - inner_bbox[1]
    if inner_width < minimum_side or inner_height < minimum_side:
        raise DetectionError(
            "A detected cell is too small after the inner inset. Reduce the inset "
            "or use a higher-resolution image."
        )
    return inner_bbox


def _lerp_rgb(
    start: tuple[int, int, int], end: tuple[int, int, int], amount: float
) -> tuple[int, int, int]:
    """Return the rounded per-channel interpolation between two RGB colors."""
    return tuple(
        round(start_channel + (end_channel - start_channel) * amount)
        for start_channel, end_channel in zip(start, end)
    )


def heatmap_color_rgb(value: float, threshold: float) -> tuple[int, int, int]:
    """Map a clot intensity to the publication blue-to-red palette."""
    if value <= threshold:
        return NO_CLOT_BLUE_RGB
    if threshold >= 255:
        return LIGHT_RED_RGB

    position = max(0.0, min(1.0, (value - threshold) / (255 - threshold)))
    if position <= 0.5:
        return _lerp_rgb(LIGHT_RED_RGB, MEDIUM_RED_RGB, position * 2)
    return _lerp_rgb(MEDIUM_RED_RGB, DEEP_RED_RGB, (position - 0.5) * 2)
