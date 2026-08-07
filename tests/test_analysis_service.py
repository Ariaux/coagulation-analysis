import unittest

import grid_detector
from analysis_service import (
    DEEP_RED_RGB,
    LIGHT_RED_RGB,
    NO_CLOT_BLUE_RGB,
    AnalysisSettings,
    heatmap_color_rgb,
    inset_bbox,
)


class AnalysisSettingsTests(unittest.TestCase):
    def test_defaults(self):
        settings = AnalysisSettings()

        self.assertEqual(5.0, settings.inset_percent)
        self.assertEqual(60.0, settings.no_clot_threshold)

    def test_validate_rejects_out_of_range_inset_percent(self):
        for inset_percent in (-0.1, 15.1):
            with self.subTest(inset_percent=inset_percent):
                with self.assertRaises(ValueError):
                    AnalysisSettings(inset_percent=inset_percent).validate()

    def test_validate_rejects_out_of_range_no_clot_threshold(self):
        for no_clot_threshold in (-0.1, 255.1):
            with self.subTest(no_clot_threshold=no_clot_threshold):
                with self.assertRaises(ValueError):
                    AnalysisSettings(
                        no_clot_threshold=no_clot_threshold
                    ).validate()

    def test_inset_bbox_returns_inner_bbox_and_rejects_too_small_crops(self):
        self.assertEqual(
            (15, 25, 195, 115),
            inset_bbox((10, 20, 200, 120), inset_percent=5.0),
        )

        with self.assertRaisesRegex(
            grid_detector.DetectionError, "too small after the inner inset"
        ):
            inset_bbox((10, 10, 48, 48), 15.0, minimum_side=32)


class PublicationPaletteTests(unittest.TestCase):
    def test_values_at_or_below_threshold_are_exact_no_clot_blue(self):
        for value in (0.0, 59.9, 60.0):
            with self.subTest(value=value):
                self.assertEqual(NO_CLOT_BLUE_RGB, heatmap_color_rgb(value, 60.0))

    def test_value_above_threshold_uses_red_color_family(self):
        red, green, blue = heatmap_color_rgb(60.1, 60.0)

        self.assertNotEqual(NO_CLOT_BLUE_RGB, (red, green, blue))
        self.assertGreaterEqual(red, green)
        self.assertGreaterEqual(red, blue)

    def test_above_threshold_gradient_has_stable_endpoints(self):
        near_threshold = heatmap_color_rgb(60.000001, 60.0)

        for actual, expected in zip(near_threshold, LIGHT_RED_RGB):
            self.assertLessEqual(abs(actual - expected), 1)
        self.assertEqual(DEEP_RED_RGB, heatmap_color_rgb(255, 60.0))
        self.assertEqual(
            heatmap_color_rgb(100, 60.0), heatmap_color_rgb(100, 60.0)
        )
