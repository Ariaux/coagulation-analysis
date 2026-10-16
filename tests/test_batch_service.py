import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import analysis_service
import batch_service
from analysis_service import AnalysisSettings
from batch_service import analyze_batch
from tests.test_grid_detector import make_fixture


class BatchServiceTests(unittest.TestCase):
    def assert_results_root_is_empty(self, results_root):
        self.assertTrue(results_root.is_dir())
        self.assertEqual([], list(results_root.iterdir()))

    def test_batch_continues_after_failure_and_archives_reports(self):
        good, _ = make_fixture(filled=(1, 5, 9))
        bad = np.full((900, 900, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            good_path = root / "good.png"
            bad_path = root / "bad.png"
            self.assertTrue(cv2.imwrite(str(good_path), good))
            self.assertTrue(cv2.imwrite(str(bad_path), bad))

            result = analyze_batch(
                [good_path, bad_path],
                AnalysisSettings(results_root=root / "results"),
            )

            self.assertEqual(1, result["success_count"])
            self.assertEqual(1, result["failure_count"])
            self.assertTrue(Path(result["batch_dir"]).is_absolute())
            with Path(result["summary_csv"]).open(
                encoding="utf-8", newline=""
            ) as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(
                ["good.png", "bad.png"],
                [row["image"] for row in summary],
            )
            self.assertEqual("Success", summary[0]["status"])
            self.assertEqual("9", summary[0]["cells"])
            self.assertEqual(
                Path(result["successes"][0]["output_dir"]).name,
                summary[0]["result"],
            )
            self.assertEqual("Failed", summary[1]["status"])
            self.assertEqual("", summary[1]["cells"])
            self.assertTrue(summary[1]["result"])
            with Path(result["failures_csv"]).open(
                encoding="utf-8", newline=""
            ) as handle:
                failures = list(csv.DictReader(handle))
            self.assertEqual("bad.png", failures[0]["image"])
            metadata_path = Path(result["batch_dir"]) / "batch-metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(5.0, metadata["settings"]["inset_percent"])
            self.assertEqual(60.0, metadata["settings"]["no_clot_threshold"])
            self.assertEqual(1, metadata["success_count"])
            self.assertEqual(1, metadata["failure_count"])
            with zipfile.ZipFile(result["zip_path"]) as archive:
                names = archive.namelist()
            self.assertIn("batch-summary.csv", names)
            self.assertIn("failures.csv", names)
            self.assertTrue(any(name.endswith("cell_09.png") for name in names))
            self.assertTrue(any(name.endswith("_analysis.zip") for name in names))
            self.assertNotIn("batch-results.zip", names)
            self.assertTrue(all(not name.startswith("/") for name in names))
            self.assertTrue(all("_staging_" not in name for name in names))

            success = result["successes"][0]
            returned_paths = [
                result["batch_dir"],
                result["summary_csv"],
                result["failures_csv"],
                result["zip_path"],
                success["output_dir"],
                success["overlay_path"],
                success["heatmap_path"],
                success["csv_path"],
                success["json_path"],
                success["zip_path"],
                *success["crop_paths"],
                *(cell["crop_path"] for cell in success["cells"]),
            ]
            for returned_path in returned_paths:
                with self.subTest(returned_path=returned_path):
                    self.assertTrue(Path(returned_path).exists())
                    self.assertNotIn("_staging_", returned_path)
            self.assertNotIn(
                "_staging_",
                Path(success["json_path"]).read_text(encoding="utf-8"),
            )
            self.assertNotIn(
                "_staging_",
                Path(success["csv_path"]).read_text(encoding="utf-8"),
            )

    def test_duplicate_unicode_sources_keep_distinct_results_in_archive(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "病例甲" / "sample.png"
            second = root / "病例乙" / "sample.png"
            first.parent.mkdir()
            second.parent.mkdir()
            encoded, png = cv2.imencode(".png", image)
            self.assertTrue(encoded)
            first.write_bytes(png.tobytes())
            second.write_bytes(png.tobytes())

            result = analyze_batch(
                (path for path in (first, second)),
                AnalysisSettings(results_root=root / "results"),
            )

            self.assertEqual(2, result["success_count"])
            output_names = [
                Path(success["output_dir"]).name for success in result["successes"]
            ]
            self.assertEqual(2, len(set(output_names)))
            with zipfile.ZipFile(result["zip_path"]) as archive:
                names = archive.namelist()
            for output_name in output_names:
                self.assertTrue(
                    any(name.startswith(f"{output_name}/") for name in names)
                )

    def test_progress_reports_each_source_in_input_order(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.png"
            second = root / "second.png"
            self.assertTrue(cv2.imwrite(str(first), image))
            self.assertTrue(cv2.imwrite(str(second), image))
            updates = []

            analyze_batch(
                [first, second],
                AnalysisSettings(results_root=root / "results"),
                progress=lambda index, total, name: updates.append(
                    (index, total, name)
                ),
            )

            self.assertEqual(
                [(1, 2, "first.png"), (2, 2, "second.png")],
                updates,
            )

    def test_empty_batch_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "^Select at least one image for batch processing\\.$",
        ):
            analyze_batch([], AnalysisSettings())

    def test_invalid_settings_are_rejected_before_results_are_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_root = Path(temp_dir) / "results"

            with self.assertRaisesRegex(ValueError, "inset_percent"):
                analyze_batch(
                    [Path(temp_dir) / "unused.png"],
                    AnalysisSettings(inset_percent=15.1, results_root=results_root),
                )

            self.assertFalse(results_root.exists())

    def test_progress_failure_removes_unpublished_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_root = root / "results"

            def fail_progress(_index, _total, _name):
                raise RuntimeError("progress callback failed")

            with self.assertRaisesRegex(RuntimeError, "progress callback failed"):
                analyze_batch(
                    [root / "unused.png"],
                    AnalysisSettings(results_root=results_root),
                    progress=fail_progress,
                )

            self.assert_results_root_is_empty(results_root)

    def test_report_failure_removes_unpublished_batch(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            results_root = root / "results"
            self.assertTrue(cv2.imwrite(str(source), image))

            with mock.patch.object(
                batch_service,
                "_write_batch_reports",
                side_effect=OSError("report write failed"),
            ):
                with self.assertRaisesRegex(OSError, "report write failed"):
                    analyze_batch(
                        [source],
                        AnalysisSettings(results_root=results_root),
                    )

            self.assert_results_root_is_empty(results_root)

    def test_zip_failure_removes_unpublished_batch(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            results_root = root / "results"
            self.assertTrue(cv2.imwrite(str(source), image))

            with mock.patch.object(
                batch_service,
                "_create_batch_zip",
                side_effect=OSError("ZIP write failed"),
            ):
                with self.assertRaisesRegex(OSError, "ZIP write failed"):
                    analyze_batch(
                        [source],
                        AnalysisSettings(results_root=results_root),
                    )

            self.assert_results_root_is_empty(results_root)

    def test_existing_empty_batch_directory_is_preserved_on_name_collision(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            results_root = root / "results"
            results_root.mkdir()
            self.assertTrue(cv2.imwrite(str(source), image))
            first_name = "batch_20260807_120000_aaaaaaaa"
            second_name = "batch_20260807_120000_bbbbbbbb"
            collision = results_root / first_name
            collision.mkdir()
            collision_inode = collision.stat().st_ino

            with mock.patch.object(batch_service, "datetime") as current_time:
                current_time.now.return_value.strftime.return_value = (
                    "batch_20260807_120000_"
                )
                with mock.patch.object(
                    batch_service.uuid,
                    "uuid4",
                    side_effect=[
                        mock.Mock(hex="a" * 32),
                        mock.Mock(hex="b" * 32),
                    ],
                ):
                    result = analyze_batch(
                        [source],
                        AnalysisSettings(results_root=results_root),
                    )

            self.assertEqual(collision_inode, collision.stat().st_ino)
            self.assertEqual(
                results_root.resolve() / second_name,
                Path(result["batch_dir"]),
            )
            self.assertTrue(Path(result["zip_path"]).is_file())

    def test_existing_symlink_is_preserved_on_name_collision(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            results_root = root / "results"
            unrelated = root / "unrelated"
            results_root.mkdir()
            unrelated.mkdir()
            sentinel = unrelated / "sentinel.txt"
            sentinel.write_text("preserve me", encoding="utf-8")
            self.assertTrue(cv2.imwrite(str(source), image))
            first_name = "batch_20260807_120000_aaaaaaaa"
            second_name = "batch_20260807_120000_bbbbbbbb"
            collision = results_root / first_name
            linked_target = Path("..") / "unrelated" / "missing"
            try:
                collision.symlink_to(linked_target, target_is_directory=True)
            except OSError as exception:
                self.skipTest(f"Directory symlinks are unavailable: {exception}")

            with mock.patch.object(batch_service, "datetime") as current_time:
                current_time.now.return_value.strftime.return_value = (
                    "batch_20260807_120000_"
                )
                with mock.patch.object(
                    batch_service.uuid,
                    "uuid4",
                    side_effect=[
                        mock.Mock(hex="a" * 32),
                        mock.Mock(hex="b" * 32),
                    ],
                ):
                    result = analyze_batch(
                        [source],
                        AnalysisSettings(results_root=results_root),
                    )

            self.assertTrue(collision.is_symlink())
            self.assertFalse(collision.exists())
            self.assertEqual(linked_target, collision.readlink())
            self.assertEqual("preserve me", sentinel.read_text(encoding="utf-8"))
            self.assertEqual(
                results_root.resolve() / second_name,
                Path(result["batch_dir"]),
            )

    def test_failure_reasons_do_not_leak_deleted_staging_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_root = root / "results"
            staging_paths = []

            def fail_with_staging_path(_source, per_image_settings):
                staging_dir = Path(per_image_settings.results_root)
                staging_paths.append(str(staging_dir))
                raise OSError(f"could not write {staging_dir / 'partial.txt'}")

            with mock.patch.object(
                batch_service,
                "analyze_image",
                side_effect=fail_with_staging_path,
            ):
                result = analyze_batch(
                    [root / "fixture.png"],
                    AnalysisSettings(results_root=results_root),
                )

            staging_prefix = staging_paths[0]
            reason = result["failures"][0]["reason"]
            self.assertIn("partial.txt", reason)
            self.assertNotIn(staging_prefix, reason)
            self.assertNotIn("_staging_", reason)

            report_paths = [
                Path(result["summary_csv"]),
                Path(result["failures_csv"]),
                Path(result["batch_dir"]) / "batch-metadata.json",
            ]
            for report_path in report_paths:
                with self.subTest(report_path=report_path):
                    contents = report_path.read_text(encoding="utf-8")
                    self.assertNotIn(staging_prefix, contents)
                    self.assertNotIn("_staging_", contents)

            with zipfile.ZipFile(result["zip_path"]) as archive:
                for name in archive.namelist():
                    self.assertNotIn("_staging_", name)
                    if name.endswith((".csv", ".json")):
                        contents = archive.read(name).decode("utf-8")
                        self.assertNotIn(staging_prefix, contents)
                        self.assertNotIn("_staging_", contents)

    def test_real_publisher_failure_hides_nested_image_staging_directory(self):
        image, _ = make_fixture(filled=(1, 5, 9))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            results_root = root / "results"
            attempted_paths = []
            self.assertTrue(cv2.imwrite(str(source), image))

            def fail_image_write(path, _image):
                attempted_paths.append(str(path))
                return False

            with mock.patch.object(
                analysis_service,
                "_write_image",
                side_effect=fail_image_write,
            ):
                result = analyze_batch(
                    [source],
                    AnalysisSettings(results_root=results_root),
                )

            nested_staging_dir = str(Path(attempted_paths[0]).parent)
            nested_staging_name = Path(nested_staging_dir).name
            reason = result["failures"][0]["reason"]
            self.assertIn("Could not write crop", reason)
            self.assertIn("<image>", reason)
            self.assertIn("cell_01.png", reason)

            report_paths = [
                Path(result["summary_csv"]),
                Path(result["failures_csv"]),
                Path(result["batch_dir"]) / "batch-metadata.json",
            ]
            published_text = [
                reason,
                *(path.read_text(encoding="utf-8") for path in report_paths),
            ]
            with zipfile.ZipFile(result["zip_path"]) as archive:
                for name in archive.namelist():
                    self.assertNotIn("_staging_", name)
                    if name.endswith((".csv", ".json")):
                        published_text.append(archive.read(name).decode("utf-8"))

            for contents in published_text:
                self.assertNotIn(nested_staging_dir, contents)
                self.assertNotIn(nested_staging_name, contents)
                self.assertNotIn("_staging_", contents)


if __name__ == "__main__":
    unittest.main()
