import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2

from tests.test_grid_detector import make_fixture
import web_controller
from web_controller import (
    open_result_folder,
    run_batch_analysis,
    run_single_analysis,
)


def write_fixture(path: Path) -> None:
    image, _ = make_fixture(filled=(1, 5, 9))
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise AssertionError(f"Could not encode fixture as {path.suffix}")
    path.write_bytes(encoded.tobytes())


class WebControllerTests(unittest.TestCase):
    def assert_single_response_has_no_artifacts(self, response):
        self.assertEqual([], response.crops)
        self.assertEqual([], response.rows)
        for field_name in (
            "overlay_path",
            "heatmap_path",
            "csv_path",
            "zip_path",
            "output_dir",
        ):
            with self.subTest(field_name=field_name):
                self.assertIsNone(getattr(response, field_name))

    def test_single_response_contains_nine_previews_and_downloads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            write_fixture(source)

            response = run_single_analysis(source, 5.0, 60.0, root / "results")

            self.assertTrue(response.ok)
            self.assertEqual(9, len(response.crops))
            self.assertTrue(Path(response.csv_path).is_file())
            self.assertTrue(Path(response.zip_path).is_file())
            self.assertEqual("9 cells detected", response.status)
            self.assertEqual(
                [[
                    cell_index,
                    (cell_index - 1) // 3 + 1,
                    (cell_index - 1) % 3 + 1,
                ] for cell_index in range(1, 10)],
                [row[:3] for row in response.rows],
            )
            self.assertTrue(all(Path(path).is_absolute() for path in response.crops))

    def test_batch_response_exposes_successes_and_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            write_fixture(source)

            response = run_batch_analysis(
                [source, root / "missing.png"],
                5.0,
                60.0,
                root / "results",
            )

            self.assertEqual(1, response.success_count)
            self.assertEqual(1, response.failure_count)
            self.assertTrue(Path(response.zip_path).is_file())
            self.assertTrue(response.ok)
            self.assertEqual("1 succeeded, 1 failed", response.status)
            self.assertEqual(
                [source.name, 9, "Success", "", response.rows[0][4]],
                response.rows[0],
            )
            self.assertTrue(Path(response.rows[0][4]).is_dir())
            self.assertEqual(
                ["missing.png", "", "Failed", response.rows[1][3], ""],
                response.rows[1],
            )
            self.assertTrue(response.rows[1][3])

    def test_invalid_single_sliders_return_actionable_failures_without_artifacts(self):
        cases = (
            (-0.1, 60.0, "inset_percent must be between 0 and 15."),
            (15.1, 60.0, "inset_percent must be between 0 and 15."),
            (5.0, -0.1, "no_clot_threshold must be between 0 and 255."),
            (5.0, 255.1, "no_clot_threshold must be between 0 and 255."),
            ("not-a-number", 60.0, "Inner crop inset must be a number."),
            (5.0, "not-a-number", "No-clot threshold must be a number."),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fixture.png"
            write_fixture(source)
            for inset, threshold, message in cases:
                with self.subTest(inset=inset, threshold=threshold):
                    response = run_single_analysis(
                        source,
                        inset,
                        threshold,
                        root / "results",
                    )
                    self.assertFalse(response.ok)
                    self.assertEqual(message, response.status)
                    self.assert_single_response_has_no_artifacts(response)

    def test_missing_or_unreadable_single_image_returns_failure_without_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            unreadable = root / "unreadable.png"
            unreadable.write_bytes(b"not an image")
            for source in (root / "missing.png", unreadable):
                with self.subTest(source=source):
                    response = run_single_analysis(
                        source,
                        5.0,
                        60.0,
                        root / "results",
                    )
                    self.assertFalse(response.ok)
                    self.assertEqual(
                        f"Could not read image: {source.name}",
                        response.status,
                    )
                    self.assert_single_response_has_no_artifacts(response)

    def test_unexpected_single_service_error_returns_failure_without_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_controller,
            "analyze_image",
            side_effect=Exception("unexpected analysis failure"),
        ):
            response = run_single_analysis(
                Path(temp_dir) / "fixture.png",
                5.0,
                60.0,
                Path(temp_dir) / "results",
            )

        self.assertFalse(response.ok)
        self.assertEqual("unexpected analysis failure", response.status)
        self.assert_single_response_has_no_artifacts(response)

    def test_batch_user_input_error_returns_explicit_failure_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            response = run_batch_analysis(
                [],
                5.0,
                60.0,
                Path(temp_dir) / "results",
            )

            self.assertFalse(response.ok)
            self.assertEqual(
                "Select at least one image for batch processing.",
                response.status,
            )
            self.assertEqual(0, response.success_count)
            self.assertEqual(0, response.failure_count)
            self.assertEqual([], response.rows)
            self.assertIsNone(response.summary_csv)
            self.assertIsNone(response.failures_csv)
            self.assertIsNone(response.zip_path)
            self.assertIsNone(response.batch_dir)

    def test_unexpected_batch_service_error_returns_failure_without_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_controller,
            "analyze_batch",
            side_effect=Exception("unexpected batch failure"),
        ):
            response = run_batch_analysis(
                [Path(temp_dir) / "fixture.png"],
                5.0,
                60.0,
                Path(temp_dir) / "results",
            )

        self.assertFalse(response.ok)
        self.assertEqual("unexpected batch failure", response.status)
        self.assertEqual(0, response.success_count)
        self.assertEqual(0, response.failure_count)
        self.assertEqual([], response.rows)
        self.assertIsNone(response.summary_csv)
        self.assertIsNone(response.failures_csv)
        self.assertIsNone(response.zip_path)
        self.assertIsNone(response.batch_dir)


class ResultFolderTests(unittest.TestCase):
    def test_opens_contained_result_folder_with_platform_command(self):
        commands = (
            ("win32", "explorer"),
            ("darwin", "open"),
            ("linux", "xdg-open"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "results"
            candidate = root / "fixture_analysis"
            candidate.mkdir(parents=True)
            for platform, executable in commands:
                with self.subTest(platform=platform), mock.patch.object(
                    web_controller.sys,
                    "platform",
                    platform,
                ), mock.patch.object(web_controller.subprocess, "Popen") as popen:
                    message = open_result_folder(candidate, root)

                    self.assertEqual(
                        "Opened result folder: fixture_analysis",
                        message,
                    )
                    popen.assert_called_once_with([executable, str(candidate.resolve())])

    def test_rejects_sibling_folder_without_opening_process(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = base / "results"
            sibling = base / "results-sibling"
            root.mkdir()
            sibling.mkdir()
            with mock.patch.object(web_controller.subprocess, "Popen") as popen:
                message = open_result_folder(sibling, root)

            self.assertEqual("Result folder is unavailable.", message)
            popen.assert_not_called()

    def test_rejects_symlink_that_resolves_outside_results_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = base / "results"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            link = root / "linked-result"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError as exception:
                self.skipTest(f"Directory symlinks are unavailable: {exception}")
            with mock.patch.object(web_controller.subprocess, "Popen") as popen:
                message = open_result_folder(link, root)

            self.assertEqual("Result folder is unavailable.", message)
            popen.assert_not_called()

    def test_missing_or_invalid_folder_input_does_not_raise(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_controller.subprocess,
            "Popen",
        ) as popen:
            root = Path(temp_dir) / "results"
            root.mkdir()
            for candidate in (None, "", root / "missing"):
                with self.subTest(candidate=candidate):
                    self.assertEqual(
                        "Result folder is unavailable.",
                        open_result_folder(candidate, root),
                    )
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
