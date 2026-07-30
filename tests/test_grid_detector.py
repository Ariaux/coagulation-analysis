import unittest

import cv2
import numpy as np

from grid_detector import (
    DetectionError,
    DetectorOptions,
    _Candidate,
    _map_squares,
    detect_inner_squares,
)


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
                    content_right,
                    content_bottom,
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


def make_incomplete_fixture(size=900, brightness=210):
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
            cv2.line(
                image,
                (inner_left, inner_top),
                (inner_left, inner_bottom),
                dark,
                line,
            )
            cv2.line(
                image,
                (inner_left, inner_top),
                (inner_right, inner_top),
                dark,
                line,
            )
    return image


def make_corner_only_fixture(size=900, brightness=210):
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
            dot_radius = 6
            for x, y in (
                (inner_left, inner_top),
                (inner_right, inner_top),
                (inner_left, inner_bottom),
                (inner_right, inner_bottom),
            ):
                cv2.rectangle(
                    image,
                    (x - dot_radius, y - dot_radius),
                    (x + dot_radius, y + dot_radius),
                    dark,
                    -1,
                )
    return image


def make_disconnected_edge_marks_fixture(size=900, brightness=210):
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
    for row in range(3):
        for col in range(3):
            x0 = margin + col * cell
            y0 = margin + row * cell
            x1 = x0 + cell
            y1 = y0 + cell
            cv2.rectangle(image, (x0, y0), (x1, y1), dark, line)

            inset = int(round(cell * 0.22))
            left = x0 + inset
            top = y0 + inset
            right = x0 + cell - inset
            bottom = y0 + cell - inset
            middle_x = (left + right) // 2
            middle_y = (top + bottom) // 2
            dot_radius = 6
            for x, y in (
                (left, top),
                (middle_x, top),
                (right, top),
                (left, middle_y),
                (right, middle_y),
                (left, bottom),
                (middle_x, bottom),
                (right, bottom),
            ):
                cv2.rectangle(
                    image,
                    (x - dot_radius, y - dot_radius),
                    (x + dot_radius, y + dot_radius),
                    dark,
                    -1,
                )
    return image


def weaken_inner_edges(edge_count, size=900, brightness=210):
    image, expected = make_fixture(size=size, brightness=brightness)
    margin = int(round(size * 0.04))
    cell = (size - 2 * margin) // 3
    line = max(8, size // 90)
    inset = int(round(cell * 0.22))
    for index in range(edge_count):
        row, col = divmod(index, 3)
        left = margin + col * cell + inset
        top = margin + row * cell + inset
        bottom = margin + (row + 1) * cell - inset
        cv2.line(
            image,
            (left, top + line * 2),
            (left, bottom - line * 2),
            (brightness, brightness, brightness),
            line * 3,
        )
    return image, expected


class GridDetectorTests(unittest.TestCase):
    def test_identity_mapping_preserves_half_open_source_boundary(self):
        candidate = _Candidate(
            (0, 0, 100, 100),
            (1.0, 1.0, 1.0, 1.0),
            confidence=1.0,
        )

        square = _map_squares(
            [candidate], np.eye(3), (100, 100, 3)
        )[0]

        self.assertEqual((0, 0, 100, 100), square.source_bbox)
        self.assertTrue(np.all(square.source_quad >= 0))
        self.assertTrue(np.all(square.source_quad <= 99))

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
            x1, y1, x2, y2 = square.rectified_bbox
            content = detection.rectified[y1:y2, x1:x2]
            self.assertGreater(content.shape[1], detection.rectified.shape[1] * 0.15)
            self.assertGreater(content.shape[0], detection.rectified.shape[0] * 0.15)
            coordinate_error = max(
                abs(actual - wanted)
                for actual, wanted in zip(square.source_bbox, truth)
            )
            self.assertLessEqual(coordinate_error, 12)
            self.assertGreaterEqual(square.confidence, 0.55)

    def test_detects_complete_fixture_touching_all_image_edges(self):
        image, _ = make_fixture()
        margin = int(round(image.shape[0] * 0.04))
        cell = (image.shape[0] - 2 * margin) // 3
        line = max(8, image.shape[0] // 90)
        fixture_end = margin + 3 * cell
        edge_start = margin - line // 2
        edge_end = fixture_end + line // 2 + 1
        edge_to_edge = image[
            edge_start:edge_end,
            edge_start:edge_end,
        ]

        detection = detect_inner_squares(edge_to_edge)

        self.assertEqual(9, len(detection.squares))
        self.assertGreaterEqual(
            min(square.confidence for square in detection.squares),
            0.55,
        )

    def test_detects_filled_cells_after_small_rotation(self):
        image, _ = make_fixture(angle=-5.0, brightness=150, filled=(1, 5, 9))

        detection = detect_inner_squares(image)

        self.assertEqual(9, len(detection.squares))
        self.assertGreaterEqual(
            min(square.confidence for square in detection.squares), 0.55
        )

    def test_rejects_images_with_either_dimension_below_600(self):
        images = [
            make_fixture(size=size)[0]
            for size in (180, 300, 599)
        ]
        images.append(
            cv2.resize(make_fixture(size=800)[0], (800, 599))
        )
        for image in images:
            with self.subTest(shape=image.shape):
                with self.assertRaises(DetectionError) as raised:
                    detect_inner_squares(image)
                self.assertIn("600", str(raised.exception))

    def test_accepts_600_fixture_with_accurate_content_geometry(self):
        image, expected = make_fixture(size=600, brightness=210)

        detection = detect_inner_squares(image)

        self.assertEqual(9, len(detection.squares))
        for square, truth in zip(detection.squares, expected):
            actual = square.source_bbox
            intersection_width = max(
                0,
                min(actual[2], truth[2]) - max(actual[0], truth[0]),
            )
            intersection_height = max(
                0,
                min(actual[3], truth[3]) - max(actual[1], truth[1]),
            )
            intersection = intersection_width * intersection_height
            actual_area = (actual[2] - actual[0]) * (actual[3] - actual[1])
            truth_area = (truth[2] - truth[0]) * (truth[3] - truth[1])
            union = actual_area + truth_area - intersection
            self.assertGreaterEqual(intersection / union, 0.75)

    def test_sample_content_does_not_move_inner_square_edges(self):
        for size in (900, 1200, 1400):
            with self.subTest(size=size):
                empty, _ = make_fixture(size=size)
                filled, _ = make_fixture(
                    size=size, filled=(1, 2, 4, 5, 8)
                )

                empty_detection = detect_inner_squares(empty)
                filled_detection = detect_inner_squares(filled)

                for empty_square, filled_square in zip(
                    empty_detection.squares, filled_detection.squares
                ):
                    coordinate_delta = max(
                        abs(empty_coordinate - filled_coordinate)
                        for empty_coordinate, filled_coordinate in zip(
                            empty_square.source_bbox,
                            filled_square.source_bbox,
                        )
                    )
                    self.assertLessEqual(coordinate_delta, 5)

    def test_tolerates_small_rotations_and_brightness_changes(self):
        for angle in (-4.0, 3.0):
            for brightness in (165, 235):
                with self.subTest(angle=angle, brightness=brightness):
                    image, _ = make_fixture(
                        angle=angle,
                        brightness=brightness,
                        filled=(1, 2, 4, 5, 8),
                    )

                    detection = detect_inner_squares(image)

                    self.assertEqual(9, len(detection.squares))
                    self.assertGreaterEqual(detection.confidence, 0.55)

    def test_rejects_image_without_complete_fixture(self):
        image = np.full((900, 900, 3), 235, dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (430, 430), (25, 25, 25), 10)

        with self.assertRaises(DetectionError) as raised:
            detect_inner_squares(image)
        self.assertEqual(
            "Could not detect a complete 3x3 fixture. Keep the full grid in "
            "frame and photograph it straight on.",
            str(raised.exception),
        )

    def test_structure_validation_uses_rectified_fixture_coordinates(self):
        image, _ = make_fixture(angle=-4.0, filled=(1, 2, 4, 5, 8))

        detection = detect_inner_squares(
            image,
            DetectorOptions(
                rectify=False,
                refine_edges=False,
                validate_grid=False,
            ),
        )

        self.assertEqual(9, len(detection.squares))

    def test_rejects_solid_dark_area_without_inner_edge_structure(self):
        size = 900
        image = np.full((size, size, 3), 210, dtype=np.uint8)
        margin = int(round(size * 0.04))
        cv2.rectangle(
            image,
            (margin, margin),
            (size - margin, size - margin),
            (25, 25, 25),
            -1,
        )

        with self.assertRaises(DetectionError):
            detect_inner_squares(image)

    def test_rejects_cells_with_incomplete_inner_frames(self):
        image = make_incomplete_fixture()

        for options in (None, DetectorOptions(validate_grid=False)):
            with self.subTest(options=options):
                with self.assertRaises(DetectionError):
                    detect_inner_squares(image, options)

    def test_rejects_corner_marks_without_continuous_inner_frames(self):
        image = make_corner_only_fixture()

        with self.assertRaises(DetectionError):
            detect_inner_squares(image)

    def test_rejects_disconnected_marks_along_every_inner_edge(self):
        image = make_disconnected_edge_marks_fixture()

        with self.assertRaises(DetectionError):
            detect_inner_squares(image)

    def test_recovers_exactly_one_globally_unique_weak_inner_edge(self):
        image, _ = weaken_inner_edges(1)

        detection = detect_inner_squares(image)

        recovered = [square for square in detection.squares if square.recovered]
        self.assertEqual(1, len(recovered))
        square = recovered[0]
        peer_lefts = [
            candidate.rectified_bbox[0]
            for candidate in detection.squares
            if candidate.col == square.col and candidate.idx != square.idx
        ]
        self.assertLessEqual(
            abs(square.rectified_bbox[0] - int(round(np.median(peer_lefts)))),
            2,
        )
        self.assertGreaterEqual(square.confidence, 0.55)
        self.assertLessEqual(square.confidence, 0.70)
        self.assertLess(
            square.confidence,
            min(
                candidate.confidence
                for candidate in detection.squares
                if candidate.col == square.col and candidate.idx != square.idx
            ),
        )

    def test_no_grid_validation_disables_weak_edge_recovery(self):
        image, _ = weaken_inner_edges(1)

        with self.assertRaises(DetectionError):
            detect_inner_squares(
                image,
                DetectorOptions(validate_grid=False),
            )

    def test_rejects_multiple_weak_inner_edges(self):
        image, _ = weaken_inner_edges(2)

        with self.assertRaises(DetectionError):
            detect_inner_squares(image)

    def test_rejects_non_uint8_images_with_actionable_message(self):
        image, _ = make_fixture()

        for dtype in (np.float32, np.int32):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(DetectionError, "uint8"):
                    detect_inner_squares(image.astype(dtype))


if __name__ == "__main__":
    unittest.main()
