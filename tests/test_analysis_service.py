import unittest

import grid_detector
from analysis_service import AnalysisSettings, inset_bbox


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
