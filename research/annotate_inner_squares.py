"""Manual row-major annotation tool for nine inner content squares."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

BBox = tuple[int, int, int, int]


def validate_annotations(image_name: str, boxes: Sequence[BBox]) -> dict:
    """Validate nine nonempty row-major xyxy boxes and return a JSON payload."""
    if len(boxes) != 9:
        raise ValueError("Exactly nine boxes are required.")
    normalized = []
    centers = []
    for index, box in enumerate(boxes, 1):
        if len(box) != 4:
            raise ValueError(f"Box {index} must contain four coordinates.")
        x1, y1, x2, y2 = map(int, box)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Box {index} must be nonempty xyxy coordinates.")
        normalized.append({"idx": index, "bbox": [x1, y1, x2, y2]})
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
    for row in range(3):
        row_centers = centers[row * 3 : row * 3 + 3]
        if not (row_centers[0][0] < row_centers[1][0] < row_centers[2][0]):
            raise ValueError("Boxes must be ordered left-to-right within each row.")
    row_y = [float(np.mean([centers[row * 3 + col][1] for col in range(3)])) for row in range(3)]
    if not row_y[0] < row_y[1] < row_y[2]:
        raise ValueError("Boxes must be ordered top-to-bottom by row.")
    return {"image": image_name, "annotator": "manual", "boxes": normalized}


def save_annotations(
    output: str | Path, image_name: str, boxes: Sequence[BBox]
) -> Path:
    """Validate and save annotations as UTF-8 JSON."""
    path = Path(output)
    payload = validate_annotations(image_name, boxes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def annotate(image_path: Path, output: Path) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    boxes: list[BBox] = []
    current: list[tuple[int, int]] = []
    window = "Manual inner-square annotation"

    def on_mouse(event: int, x: int, y: int, _flags: int, _parameter: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or len(boxes) >= 9:
            return
        current.append((x, y))
        if len(current) == 2:
            (x1, y1), (x2, y2) = current
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
            current.clear()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        display = image.copy()
        for index, box in enumerate(boxes, 1):
            cv2.rectangle(display, box[:2], box[2:], (0, 255, 0), 2)
            cv2.putText(
                display,
                str(index),
                (box[0] + 4, box[1] + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        if current:
            cv2.circle(display, current[0], 5, (0, 255, 255), -1)
        cv2.putText(
            display,
            f"Cell {min(len(boxes) + 1, 9)}/9: click top-left, then bottom-right",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 0),
            2,
        )
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(window)
            return False
        if key in (8, 127):
            if current:
                current.clear()
            elif boxes:
                boxes.pop()
        if key in (10, 13) and len(boxes) == 9:
            save_annotations(output, image_path.name, boxes)
            cv2.destroyWindow(window)
            return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    output = arguments.output or arguments.image.with_suffix(".annotations.json")
    saved = annotate(arguments.image, output)
    if saved:
        print(output)
        return 0
    print("Annotation cancelled.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
