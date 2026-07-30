"""Detection of the nine inner content squares in a fixed 3x3 fixture."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class DetectionError(RuntimeError):
    """Raised when an image does not contain a usable nine-square fixture."""


@dataclass(frozen=True)
class InnerSquare:
    idx: int
    row: int
    col: int
    source_quad: np.ndarray
    source_bbox: tuple[int, int, int, int]
    rectified_bbox: tuple[int, int, int, int]
    confidence: float
    recovered: bool = False


@dataclass(frozen=True)
class GridDetection:
    squares: tuple[InnerSquare, ...]
    rectified: np.ndarray
    source_to_rectified: np.ndarray
    rectified_to_source: np.ndarray
    outer_quad: np.ndarray
    confidence: float


@dataclass(frozen=True)
class DetectorOptions:
    rectify: bool = True
    refine_edges: bool = True
    validate_grid: bool = True


@dataclass
class _Candidate:
    bbox: tuple[int, int, int, int]
    edge_strength: float
    confidence: float = 0.0
    recovered: bool = False


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype=np.float32,
    )


def _find_outer_quad(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    short_side = min(image.shape[:2])
    kernel_size = max(3, int(round(short_side * 0.015)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    connected = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = float(image.shape[0] * image.shape[1])
    candidates = [
        contour
        for contour in contours
        if 0.20 <= cv2.contourArea(contour) / image_area <= 0.95
    ]
    if not candidates:
        raise DetectionError(
            "Could not find the outer fixture. Use a front-facing photo with "
            "the complete dark frame visible."
        )

    contour = max(candidates, key=cv2.contourArea)
    return _order_quad(cv2.boxPoints(cv2.minAreaRect(contour)))


def _rectify(
    image: np.ndarray, quad: np.ndarray, enabled: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not enabled:
        identity = np.eye(3, dtype=np.float64)
        return image.copy(), identity, identity

    top_left, top_right, bottom_right, bottom_left = _order_quad(quad)
    lengths = [
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - top_right),
        np.linalg.norm(bottom_left - bottom_right),
        np.linalg.norm(top_left - bottom_left),
    ]
    side = max(2, int(round(float(np.mean(lengths)))))
    destination = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
        dtype=np.float32,
    )
    forward = cv2.getPerspectiveTransform(
        np.asarray(quad, dtype=np.float32), destination
    )
    inverse = cv2.getPerspectiveTransform(
        destination, np.asarray(quad, dtype=np.float32)
    )
    rectified = cv2.warpPerspective(image, forward, (side, side))
    return rectified, forward, inverse


def _ridge(
    darkness: np.ndarray,
    expected: float,
    radius: int,
    axis: int,
    band_start: int,
    band_end: int,
) -> tuple[float, float, float]:
    limit = darkness.shape[1] if axis == 0 else darkness.shape[0]
    low = max(0, int(round(expected)) - radius)
    high = min(limit - 1, int(round(expected)) + radius)
    if axis == 0:
        projection = darkness[band_start:band_end, low : high + 1].mean(axis=0)
    else:
        projection = darkness[low : high + 1, band_start:band_end].mean(axis=1)
    peak_offset = int(np.argmax(projection))
    peak = float(projection[peak_offset])
    baseline = float(projection.min())
    threshold = baseline + (peak - baseline) * 0.55
    left = peak_offset
    right = peak_offset
    while left > 0 and projection[left - 1] >= threshold:
        left -= 1
    while right + 1 < projection.size and projection[right + 1] >= threshold:
        right += 1
    center = low + (left + right) / 2.0
    thickness = float(right - left + 1)
    strength = float(np.clip(peak - baseline, 0.0, 1.0))
    return center, thickness, strength


def _refine_template_squares(
    rectified: np.ndarray, enabled: bool
) -> list[_Candidate]:
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    darkness = 1.0 - gray.astype(np.float32) / 255.0
    height, width = gray.shape
    cell_x = width / 3.0
    cell_y = height / 3.0
    radius_x = max(2, int(round(cell_x * 0.07)))
    radius_y = max(2, int(round(cell_y * 0.07)))
    candidates: list[_Candidate] = []

    for row in range(3):
        for col in range(3):
            predicted_left = (col + 0.22) * cell_x
            predicted_right = (col + 0.78) * cell_x
            predicted_top = (row + 0.22) * cell_y
            predicted_bottom = (row + 0.78) * cell_y

            if enabled:
                vertical_start = max(0, int((row + 0.30) * cell_y))
                vertical_end = min(height, int((row + 0.70) * cell_y))
                horizontal_start = max(0, int((col + 0.30) * cell_x))
                horizontal_end = min(width, int((col + 0.70) * cell_x))
                left, left_width, left_strength = _ridge(
                    darkness,
                    predicted_left,
                    radius_x,
                    0,
                    vertical_start,
                    vertical_end,
                )
                right, right_width, right_strength = _ridge(
                    darkness,
                    predicted_right,
                    radius_x,
                    0,
                    vertical_start,
                    vertical_end,
                )
                top, top_width, top_strength = _ridge(
                    darkness,
                    predicted_top,
                    radius_y,
                    1,
                    horizontal_start,
                    horizontal_end,
                )
                bottom, bottom_width, bottom_strength = _ridge(
                    darkness,
                    predicted_bottom,
                    radius_y,
                    1,
                    horizontal_start,
                    horizontal_end,
                )
                x0 = int(round(left + left_width / 2.0 + 2))
                x1 = int(round(right - right_width / 2.0 - 2))
                y0 = int(round(top + top_width / 2.0 + 2))
                y1 = int(round(bottom - bottom_width / 2.0 - 2))
                strength = float(
                    np.mean(
                        [
                            left_strength,
                            right_strength,
                            top_strength,
                            bottom_strength,
                        ]
                    )
                )
            else:
                x0 = int(round(predicted_left + 2))
                x1 = int(round(predicted_right - 2))
                y0 = int(round(predicted_top + 2))
                y1 = int(round(predicted_bottom - 2))
                strength = 0.7

            if x1 <= x0 or y1 <= y0:
                continue
            candidates.append(_Candidate((x0, y0, x1, y1), strength))
    return candidates


def _validate_grid(
    raw: list[_Candidate], shape: tuple[int, ...], enabled: bool
) -> list[_Candidate]:
    if not raw:
        raise DetectionError(
            "No inner frames were found. Make sure the complete fixture is visible."
        )
    if not enabled:
        for candidate in raw:
            candidate.confidence = min(1.0, max(0.0, candidate.edge_strength))
        return raw
    if len(raw) != 9:
        raise DetectionError(
            f"Expected nine inner frames but found {len(raw)}. "
            "Retake the photo with all cells unobstructed."
        )

    widths = np.array(
        [candidate.bbox[2] - candidate.bbox[0] for candidate in raw],
        dtype=float,
    )
    heights = np.array(
        [candidate.bbox[3] - candidate.bbox[1] for candidate in raw],
        dtype=float,
    )
    median_width = float(np.median(widths))
    median_height = float(np.median(heights))
    centers = np.array(
        [
            ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            for x1, y1, x2, y2 in (candidate.bbox for candidate in raw)
        ]
    ).reshape(3, 3, 2)

    if np.any((widths / heights < 0.85) | (widths / heights > 1.15)):
        raise DetectionError(
            "The inner frames are too distorted. Use a more front-facing photo."
        )
    if np.any(np.abs(widths - median_width) > median_width * 0.15) or np.any(
        np.abs(heights - median_height) > median_height * 0.15
    ):
        raise DetectionError(
            "The detected inner frames have inconsistent sizes. "
            "Retake the photo with the whole fixture in view."
        )
    if any(np.any(np.diff(centers[row, :, 0]) <= 0) for row in range(3)) or any(
        np.any(np.diff(centers[:, col, 1]) <= 0) for col in range(3)
    ):
        raise DetectionError(
            "The inner frames could not be ordered into a regular 3x3 grid."
        )

    for candidate in raw:
        x1, y1, x2, y2 = candidate.bbox
        width = x2 - x1
        height = y2 - y1
        aspect = min(width, height) / max(width, height)
        size_agreement = 1.0 - max(
            abs(width - median_width) / median_width,
            abs(height - median_height) / median_height,
        )
        candidate.confidence = float(
            np.clip(
                0.50 * candidate.edge_strength
                + 0.25 * aspect
                + 0.25 * size_agreement,
                0.0,
                1.0,
            )
        )

    if min(candidate.confidence for candidate in raw) < 0.55:
        raise DetectionError(
            "The inner frames are not clear enough to detect reliably. "
            "Use even lighting and keep the dark frame in focus."
        )
    return raw


def _map_squares(
    candidates: list[_Candidate],
    inverse: np.ndarray,
    source_shape: tuple[int, ...],
) -> tuple[InnerSquare, ...]:
    source_height, source_width = source_shape[:2]
    squares = []
    for position, candidate in enumerate(candidates):
        x1, y1, x2, y2 = candidate.bbox
        rectified_corners = np.array(
            [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
            dtype=np.float32,
        )
        source_quad = cv2.perspectiveTransform(
            rectified_corners, inverse
        )[0].astype(np.float32)
        source_quad[:, 0] = np.clip(source_quad[:, 0], 0, source_width - 1)
        source_quad[:, 1] = np.clip(source_quad[:, 1], 0, source_height - 1)
        left = int(np.floor(source_quad[:, 0].min()))
        top = int(np.floor(source_quad[:, 1].min()))
        right = int(np.ceil(source_quad[:, 0].max()))
        bottom = int(np.ceil(source_quad[:, 1].max()))
        row, col = divmod(position, 3)
        squares.append(
            InnerSquare(
                idx=position + 1,
                row=row + 1,
                col=col + 1,
                source_quad=source_quad,
                source_bbox=(left, top, right, bottom),
                rectified_bbox=candidate.bbox,
                confidence=candidate.confidence,
                recovered=candidate.recovered,
            )
        )
    return tuple(squares)


def detect_inner_squares(
    image: np.ndarray, options: DetectorOptions | None = None
) -> GridDetection:
    """Detect and number the inner content areas of a front-facing 3x3 fixture."""

    if (
        image is None
        or not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise DetectionError("Expected a three-channel color image.")
    if min(image.shape[:2]) < 180:
        raise DetectionError(
            "The image is too small. Use an image at least 180 pixels on each side."
        )

    options = options or DetectorOptions()
    outer_quad = _find_outer_quad(image)
    rectified, forward, inverse = _rectify(image, outer_quad, options.rectify)
    raw = _refine_template_squares(rectified, options.refine_edges)
    candidates = _validate_grid(raw, rectified.shape, options.validate_grid)
    squares = _map_squares(candidates, inverse, image.shape)
    confidence = float(np.mean([square.confidence for square in squares]))
    return GridDetection(
        squares=squares,
        rectified=rectified,
        source_to_rectified=forward,
        rectified_to_source=inverse,
        outer_quad=outer_quad,
        confidence=confidence,
    )
