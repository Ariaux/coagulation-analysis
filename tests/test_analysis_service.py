import csv
import json
import os
import shutil
import tempfile
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import analysis_service
import grid_detector
from analysis_service import (
    DEEP_RED_RGB,
    LIGHT_RED_RGB,
    NO_CLOT_BLUE_RGB,
    AnalysisSettings,
    analyze_image,
    heatmap_color_rgb,
    inset_bbox,
)
from tests.test_grid_detector import make_fixture


class SingleImageServiceTests(unittest.TestCase):
    def test_analyze_image_publishes_inset_cells_and_metadata(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            self.assertTrue(cv2.imwrite(str(source), image))

            result = analyze_image(
                source,
                AnalysisSettings(5.0, 60.0, root / "results"),
            )

            self.assertEqual(9, len(result["cells"]))
            self.assertTrue(Path(result["zip_path"]).is_file())
            self.assertEqual(
                Path(result["output_dir"]), Path(result["zip_path"]).parent
            )
            for cell in result["cells"]:
                detected_x1, detected_y1, detected_x2, detected_y2 = cell[
                    "detected_bbox"
                ]
                final_x1, final_y1, final_x2, final_y2 = cell["final_bbox"]
                self.assertGreater(final_x1, detected_x1)
                self.assertGreater(final_y1, detected_y1)
                self.assertLess(final_x2, detected_x2)
                self.assertLess(final_y2, detected_y2)
                self.assertTrue(Path(cell["crop_path"]).is_file())
                self.assertEqual(
                    Path(result["output_dir"]), Path(cell["crop_path"]).parent
                )

            with Path(result["json_path"]).open(encoding="utf-8") as results_file:
                payload = json.load(results_file)
            self.assertEqual(
                "ImageJ-equivalent inverted 8-bit grayscale mean",
                payload["measurement_method"],
            )
            self.assertEqual(5.0, payload["settings"]["inset_percent"])
            self.assertEqual(60.0, payload["settings"]["no_clot_threshold"])
            self.assertEqual("publication-blue-red-v1", payload["palette_version"])
            self.assertEqual(
                "publication-blue-red-v1",
                payload["settings"]["palette_version"],
            )
            self.assertEqual(
                [Path(cell["crop_path"]).name for cell in result["cells"]],
                [cell["crop_file"] for cell in payload["cells"]],
            )
            self.assertTrue(all("crop_path" not in cell for cell in payload["cells"]))

            with Path(result["csv_path"]).open(
                newline="", encoding="utf-8"
            ) as results_file:
                rows = list(csv.DictReader(results_file))
            self.assertEqual(9, len(rows))
            self.assertIn("final_x1", rows[0])
            self.assertEqual("5.0", rows[0]["inset_percent"])
            self.assertEqual(
                Path(result["cells"][0]["crop_path"]).name,
                rows[0]["crop_file"],
            )

    def test_precommit_bundle_publish_failure_restores_the_prior_complete_result(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            results_root = root / "results"
            self.assertTrue(cv2.imwrite(str(source), image))
            prior = analyze_image(
                source,
                AnalysisSettings(5.0, 60.0, results_root),
            )
            prior_crop = cv2.imread(prior["crop_paths"][0])
            self.assertIsNotNone(prior_crop)

            real_replace = analysis_service.os.replace

            def fail_staging_bundle_publish(source_path, destination_path):
                source_path = Path(source_path)
                destination_path = Path(destination_path)
                if "_staging_" in source_path.name and destination_path == Path(
                    prior["output_dir"]
                ):
                    raise OSError("simulated precommit bundle publish failure")
                return real_replace(source_path, destination_path)

            with mock.patch.object(
                analysis_service.os,
                "replace",
                side_effect=fail_staging_bundle_publish,
            ):
                with self.assertRaisesRegex(
                    OSError, "simulated precommit bundle publish failure"
                ):
                    analyze_image(
                        source,
                        AnalysisSettings(10.0, 60.0, results_root),
                    )

            with Path(prior["json_path"]).open(encoding="utf-8") as results_file:
                metadata = json.load(results_file)
            self.assertEqual(5.0, metadata["settings"]["inset_percent"])
            restored_crop = cv2.imread(prior["crop_paths"][0])
            self.assertIsNotNone(restored_crop)
            self.assertEqual(prior_crop.shape, restored_crop.shape)
            with zipfile.ZipFile(prior["zip_path"]) as archive:
                json_name = next(
                    name
                    for name in archive.namelist()
                    if name.endswith("_results.json")
                )
                archived_metadata = json.loads(archive.read(json_name))
            self.assertEqual(
                5.0,
                archived_metadata["settings"]["inset_percent"],
            )
            self.assertFalse(
                any(
                    ".previous-" in path.name or "_staging_" in path.name
                    for path in results_root.iterdir()
                )
            )

    def test_failed_rollback_preserves_owned_bundle_for_later_recovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final_dir = root / "fixture_analysis"
            staging_dir = root / ".fixture_staging"
            final_dir.mkdir()
            staging_dir.mkdir()
            (final_dir / "prior.txt").write_text("prior", encoding="utf-8")
            (staging_dir / "new.txt").write_text("new", encoding="utf-8")
            real_replace = analysis_service.os.replace

            def fail_publish_and_rollback(source_path, destination_path):
                source_path = Path(source_path)
                destination_path = Path(destination_path)
                if destination_path == final_dir and source_path == staging_dir:
                    raise OSError("simulated publish failure")
                if destination_path == final_dir and source_path.name == "output":
                    raise OSError("simulated rollback failure")
                return real_replace(source_path, destination_path)

            with mock.patch.object(
                analysis_service.os,
                "replace",
                side_effect=fail_publish_and_rollback,
            ):
                with self.assertRaisesRegex(OSError, "simulated publish failure"):
                    analysis_service._atomic_publish_bundle(staging_dir, final_dir)

            transactions = [
                path
                for path in root.iterdir()
                if analysis_service._is_owned_transaction(path, final_dir)
            ]
            self.assertEqual(1, len(transactions))
            self.assertEqual(
                "prior",
                (transactions[0] / "output" / "prior.txt").read_text(encoding="utf-8"),
            )

            analysis_service._recover_previous_bundle(final_dir)

            self.assertEqual(
                "prior",
                (final_dir / "prior.txt").read_text(encoding="utf-8"),
            )
            self.assertFalse(transactions[0].exists())

    def test_partial_postcommit_cleanup_failure_keeps_new_result_complete(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            results_root = root / "results"
            self.assertTrue(cv2.imwrite(str(source), image))
            prior = analyze_image(
                source,
                AnalysisSettings(5.0, 60.0, results_root),
            )
            prior_crop = cv2.imread(prior["crop_paths"][0])
            self.assertIsNotNone(prior_crop)

            real_remove_directory = analysis_service._remove_known_directory

            def partially_remove_previous_result(path):
                path = Path(path)
                if ".previous-" in path.name:
                    shutil.rmtree(path / "output")
                    raise OSError("simulated partial postcommit cleanup failure")
                return real_remove_directory(path)

            with mock.patch.object(
                analysis_service,
                "_remove_known_directory",
                side_effect=partially_remove_previous_result,
            ):
                result = analyze_image(
                    source,
                    AnalysisSettings(10.0, 60.0, results_root),
                )

            with Path(result["json_path"]).open(encoding="utf-8") as results_file:
                metadata = json.load(results_file)
            self.assertEqual(10.0, metadata["settings"]["inset_percent"])
            new_crop = cv2.imread(result["crop_paths"][0])
            self.assertIsNotNone(new_crop)
            self.assertLess(new_crop.shape[0], prior_crop.shape[0])
            self.assertLess(new_crop.shape[1], prior_crop.shape[1])
            with zipfile.ZipFile(result["zip_path"]) as archive:
                names = archive.namelist()
                json_name = next(
                    name for name in names if name.endswith("_results.json")
                )
                archived_metadata = json.loads(archive.read(json_name))
            self.assertEqual(13, len(names))
            self.assertEqual(
                10.0,
                archived_metadata["settings"]["inset_percent"],
            )
            self.assertFalse(
                any("_staging_" in path.name for path in results_root.iterdir())
            )

            transactions = [
                path for path in results_root.iterdir() if ".previous-" in path.name
            ]
            self.assertEqual(1, len(transactions))
            for transaction in transactions:
                real_remove_directory(transaction)
            self.assertFalse(
                any(".previous-" in path.name for path in results_root.iterdir())
            )

    def test_stranded_previous_transaction_is_recovered(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            results_root = root / "results"
            self.assertTrue(cv2.imwrite(str(source), image))
            prior = analyze_image(
                source,
                AnalysisSettings(5.0, 60.0, results_root),
            )
            final_dir = Path(prior["output_dir"])
            transaction = analysis_service._create_transaction_root(final_dir)
            os.replace(final_dir, transaction / "output")
            self.assertFalse(final_dir.exists())

            analysis_service._recover_previous_bundle(final_dir)

            self.assertTrue(Path(prior["json_path"]).is_file())
            self.assertTrue(Path(prior["zip_path"]).is_file())
            self.assertFalse(transaction.exists())

    def test_recovery_preserves_unowned_prefix_matching_directories(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            results_root = root / "results"
            self.assertTrue(cv2.imwrite(str(source), image))
            result = analyze_image(
                source,
                AnalysisSettings(5.0, 60.0, results_root),
            )
            final_dir = Path(result["output_dir"])
            lookalike = final_dir.parent / f".{final_dir.name}.previous-user-notes"
            unowned_uuid = final_dir.parent / (f".{final_dir.name}.previous-{'0' * 32}")
            for directory in (lookalike, unowned_uuid):
                directory.mkdir()
                (directory / "do-not-delete.txt").write_text(
                    "user content",
                    encoding="utf-8",
                )

            analysis_service._recover_previous_bundle(final_dir)

            for directory in (lookalike, unowned_uuid):
                self.assertEqual(
                    "user content",
                    (directory / "do-not-delete.txt").read_text(encoding="utf-8"),
                )

    def test_shared_results_root_distinguishes_duplicate_basenames(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_source = root / "first" / "fixture.png"
            second_source = root / "second" / "fixture.png"
            results_root = root / "results"
            first_source.parent.mkdir()
            second_source.parent.mkdir()
            self.assertTrue(cv2.imwrite(str(first_source), image))
            self.assertTrue(cv2.imwrite(str(second_source), image))

            first = analyze_image(
                first_source,
                AnalysisSettings(5.0, 60.0, results_root),
            )
            second = analyze_image(
                second_source,
                AnalysisSettings(10.0, 60.0, results_root),
            )
            repeated = analyze_image(
                first_source,
                AnalysisSettings(5.0, 60.0, results_root),
            )

            self.assertNotEqual(first["output_dir"], second["output_dir"])
            self.assertEqual(first["output_dir"], repeated["output_dir"])
            self.assertTrue(Path(first["json_path"]).is_file())
            self.assertTrue(Path(second["json_path"]).is_file())

    def test_concurrent_same_target_publications_keep_bundle_consistent(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            results_root = root / "results"
            self.assertTrue(cv2.imwrite(str(source), image))

            def run_analysis(inset_percent):
                return analyze_image(
                    source,
                    AnalysisSettings(inset_percent, 60.0, results_root),
                )

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(run_analysis, (5.0, 10.0, 5.0, 10.0)))

            self.assertEqual(1, len({result["output_dir"] for result in results}))
            final = results[-1]
            with Path(final["json_path"]).open(encoding="utf-8") as results_file:
                metadata = json.load(results_file)
            with zipfile.ZipFile(final["zip_path"]) as archive:
                json_name = next(
                    name
                    for name in archive.namelist()
                    if name.endswith("_results.json")
                )
                archived_metadata = json.loads(archive.read(json_name))
            self.assertEqual(metadata["settings"], archived_metadata["settings"])
            self.assertEqual(metadata["cells"], archived_metadata["cells"])

    def test_relative_source_uses_absolute_default_artifact_paths(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            self.assertTrue(cv2.imwrite(str(source), image))
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                result = analyze_image(Path("fixture.png"), AnalysisSettings())
            finally:
                os.chdir(previous_cwd)

            for key in (
                "output_dir",
                "overlay_path",
                "heatmap_path",
                "csv_path",
                "json_path",
                "zip_path",
            ):
                self.assertTrue(Path(result[key]).is_absolute(), key)
            self.assertTrue(
                all(Path(path).is_absolute() for path in result["crop_paths"])
            )
            self.assertTrue(
                all(Path(cell["crop_path"]).is_absolute() for cell in result["cells"])
            )

    def test_zip_metadata_uses_portable_crop_references(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            results_root = root / "private-results"
            extracted_root = root / "extracted"
            self.assertTrue(cv2.imwrite(str(source), image))
            result = analyze_image(
                source,
                AnalysisSettings(5.0, 60.0, results_root),
            )
            self.assertTrue(
                all(Path(cell["crop_path"]).is_absolute() for cell in result["cells"])
            )

            with zipfile.ZipFile(result["zip_path"]) as archive:
                archive.extractall(extracted_root)
                names = archive.namelist()
            json_path = extracted_root / next(
                name for name in names if name.endswith("_results.json")
            )
            csv_path = extracted_root / next(
                name for name in names if name.endswith("_results.csv")
            )
            json_text = json_path.read_text(encoding="utf-8")
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertNotIn(str(results_root.resolve()), json_text)
            self.assertNotIn(str(results_root.resolve()), csv_text)

            payload = json.loads(json_text)
            with csv_path.open(newline="", encoding="utf-8") as results_file:
                rows = list(csv.DictReader(results_file))
            for crop_file in [cell["crop_file"] for cell in payload["cells"]] + [
                row["crop_file"] for row in rows
            ]:
                reference = Path(crop_file)
                self.assertFalse(reference.is_absolute())
                self.assertNotIn("..", reference.parts)
                self.assertTrue((extracted_root / reference).is_file())


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
                    AnalysisSettings(no_clot_threshold=no_clot_threshold).validate()

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
        self.assertEqual(heatmap_color_rgb(100, 60.0), heatmap_color_rgb(100, 60.0))


class PublicationLayoutTests(unittest.TestCase):
    def test_heatmap_legend_stays_in_header_and_handles_threshold_255(self):
        results = [
            {
                "idx": idx,
                "row": (idx - 1) // 3 + 1,
                "col": (idx - 1) % 3 + 1,
                "mean": 100.0,
            }
            for idx in range(1, 10)
        ]

        heatmap = analysis_service.heatmap_image(results, 60.0)
        cell_color = np.array(heatmap_color_rgb(100.0, 60.0)[::-1])
        self.assertEqual((430, 384, 3), heatmap.shape)
        np.testing.assert_array_equal(cell_color, heatmap[405, 280])
        np.testing.assert_array_equal(
            np.array(NO_CLOT_BLUE_RGB[::-1]),
            heatmap[10, 250],
        )
        np.testing.assert_array_equal(
            np.array(DEEP_RED_RGB[::-1]),
            heatmap[10, 369],
        )

        threshold_255 = analysis_service.heatmap_image(results, 255.0)
        np.testing.assert_array_equal(
            np.array(NO_CLOT_BLUE_RGB[::-1]),
            threshold_255[10, 250],
        )
        np.testing.assert_array_equal(
            np.array(NO_CLOT_BLUE_RGB[::-1]),
            threshold_255[10, 369],
        )

    def test_overlay_draws_half_open_boxes_at_last_included_pixel(self):
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        cells = [
            {
                "idx": 1,
                "detected_bbox": [2, 2, 20, 20],
                "final_bbox": [5, 5, 15, 15],
            }
        ]

        with mock.patch.object(
            analysis_service.cv2,
            "rectangle",
            wraps=analysis_service.cv2.rectangle,
        ) as rectangle, mock.patch.object(
            analysis_service.cv2,
            "line",
            wraps=analysis_service.cv2.line,
        ) as line:
            analysis_service.draw_detection_overlay(image, cells)

        self.assertEqual((5, 5), rectangle.call_args.args[1])
        self.assertEqual((14, 14), rectangle.call_args.args[2])
        for line_call in line.call_args_list:
            start, end = line_call.args[1:3]
            for x, y in (start, end):
                self.assertLessEqual(x, 19)
                self.assertLessEqual(y, 19)
