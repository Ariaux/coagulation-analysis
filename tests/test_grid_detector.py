import unittest

import cv2
import numpy as np

from grid_detector import DetectionError, detect_inner_squares


def make_fixture(
    size=900,
    angle=0.0,
    brightness=210,
    filled=(),
):
    image = np.full((size, size, 3), brightness, dtype=np.uint8)
    margin = int(round(size * 0.04))
    cell = (size - 2 * margin) // 3
    line = max(8, size // 90)
    dark = (25, 25, 25)

    fixture_end = margin + 3 * cell
    cv2.rectangle(
        image,
        (margin, margin),
        (fixture_end, fixture_end),
        dark,
        line,
    )

    expected = []
    for row in range(3):
        for col in range(3):
            x0 = margin + col * cell
            y0 = margin + row * cell
            x1 = x0 + cell
            y1 = y0 + cell
            cv2.rectangle(image, (x0, y0), (x1, y1), dark, line)

            inset = int(round(cell * 0.22))
            inner_left = x0 + inset
            inner_top = y0 + inset
            inner_right = x0 + cell - inset
            inner_bottom = y0 + cell - inset
            cv2.rectangle(
                image,
                (inner_left, inner_top),
                (inner_right, inner_bottom),
                dark,
                line,
            )

            content_left = inner_left + line // 2 + 2
            content_top = inner_top + line // 2 + 2
            content_right = inner_right - line // 2 - 2
            content_bottom = inner_bottom - line // 2 - 2
            expected.append(
                (
                    content_left,
                    content_top,
                    content_right - content_left,
                    content_bottom - content_top,
                )
            )

            idx = row * 3 + col + 1
            if idx in filled:
                cv2.rectangle(
                    image,
                    (content_left, content_top),
                    (content_right, content_bottom),
                    (65, 65, 65),
                    -1,
                )

    if angle:
        center = ((size - 1) / 2.0, (size - 1) / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(
            image,
            matrix,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderValue=(brightness, brightness, brightness),
        )

    return image, expected


class GridDetectorTests(unittest.TestCase):
    def test_detects_nine_squares_in_row_major_order(self):
        image, expected = make_fixture()

        detection = detect_inner_squares(image)

        self.assertEqual(9, len(detection.squares))
        self.assertEqual(list(range(1, 10)), [square.idx for square in detection.squares])
        self.assertEqual(
            [(row, col) for row in range(1, 4) for col in range(1, 4)],
            [(square.row, square.col) for square in detection.squares],
        )
        for square, truth in zip(detection.squares, expected):
            coordinate_error = max(
                abs(actual - wanted)
                for actual, wanted in zip(square.source_bbox, truth)
            )
            self.assertLessEqual(coordinate_error, 12)
            self.assertGreaterEqual(square.confidence, 0.55)

    def test_detects_filled_cells_after_small_rotation(self):
        image, _ = make_fixture(angle=-5.0, brightness=150, filled=(1, 5, 9))

        detection = detect_inner_squares(image)

        self.assertEqual(9, len(detection.squares))
        self.assertGreaterEqual(
            min(square.confidence for square in detection.squares), 0.55
        )


if __name__ == "__main__":
    unittest.main()
