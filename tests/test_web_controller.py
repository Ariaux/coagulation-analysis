from contextlib import contextmanager
import logging
import os
import tempfile
import threading
import unittest
from pathlib import Path, PureWindowsPath
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
    def call_without_ui_exception(self, function, *args):
        try:
            return function(*args)
        except Exception as exception:
            self.fail(f"Controller exception escaped into the UI: {exception!r}")

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

    def assert_batch_response_has_no_artifacts(self, response):
        self.assertEqual(0, response.success_count)
        self.assertEqual(0, response.failure_count)
        self.assertEqual([], response.rows)
        for field_name in (
            "summary_csv",
            "failures_csv",
            "zip_path",
            "batch_dir",
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
        with tempfile.TemporaryDirectory() as temp_dir:
            secret_path = Path(temp_dir) / "private" / "partial.txt"
            with mock.patch.object(
                web_controller,
                "analyze_image",
                side_effect=Exception(f"unexpected failure at {secret_path}"),
            ), mock.patch.object(
                logging.getLogger("web_controller"), "exception"
            ) as log_error:
                response = run_single_analysis(
                    Path(temp_dir) / "fixture.png",
                    5.0,
                    60.0,
                    Path(temp_dir) / "results",
                )

        self.assertFalse(response.ok)
        self.assertEqual(
            "Analysis failed unexpectedly. Check the local log and try again.",
            response.status,
        )
        self.assertNotIn(str(secret_path), response.status)
        log_error.assert_called_once()
        self.assert_single_response_has_no_artifacts(response)

    def test_single_service_value_error_is_logged_and_sanitized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            secret_path = Path(temp_dir) / "private" / "value-error.txt"
            with mock.patch.object(
                web_controller,
                "analyze_image",
                side_effect=ValueError(f"internal failure at {secret_path}"),
            ), mock.patch.object(
                logging.getLogger("web_controller"), "exception"
            ) as log_error:
                response = run_single_analysis(
                    Path(temp_dir) / "fixture.png",
                    5.0,
                    60.0,
                    Path(temp_dir) / "results",
                )

        self.assertFalse(response.ok)
        self.assertEqual(
            "Analysis failed unexpectedly. Check the local log and try again.",
            response.status,
        )
        self.assertNotIn(str(secret_path), response.status)
        log_error.assert_called_once()
        self.assert_single_response_has_no_artifacts(response)

    def test_malformed_single_service_payload_is_contained_without_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_controller,
            "analyze_image",
            return_value={},
        ), mock.patch.object(
            logging.getLogger("web_controller"), "exception"
        ) as log_error:
            response = self.call_without_ui_exception(
                run_single_analysis,
                Path(temp_dir) / "fixture.png",
                5.0,
                60.0,
                Path(temp_dir) / "results",
            )

        self.assertFalse(response.ok)
        self.assertEqual(
            "Analysis failed unexpectedly. Check the local log and try again.",
            response.status,
        )
        log_error.assert_called_once()
        self.assert_single_response_has_no_artifacts(response)

    def test_single_filesystem_error_uses_stable_results_folder_message(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            private_path = Path(temp_dir) / "results" / "private.tmp"
            with mock.patch.object(
                web_controller,
                "analyze_image",
                side_effect=OSError(f"cannot write {private_path}"),
            ), mock.patch.object(
                logging.getLogger("web_controller"), "exception"
            ) as log_error:
                response = run_single_analysis(
                    Path(temp_dir) / "fixture.png",
                    5.0,
                    60.0,
                    Path(temp_dir) / "results",
                )

        self.assertFalse(response.ok)
        self.assertEqual(
            "Could not write results. Check that the results folder is available and writable.",
            response.status,
        )
        self.assertNotIn(str(private_path), response.status)
        log_error.assert_called_once()
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
            self.assert_batch_response_has_no_artifacts(response)

    def test_unexpected_batch_service_error_returns_failure_without_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            secret_path = Path(temp_dir) / "private" / "batch.tmp"
            with mock.patch.object(
                web_controller,
                "analyze_batch",
                side_effect=Exception(f"unexpected failure at {secret_path}"),
            ), mock.patch.object(
                logging.getLogger("web_controller"), "exception"
            ) as log_error:
                response = run_batch_analysis(
                    [Path(temp_dir) / "fixture.png"],
                    5.0,
                    60.0,
                    Path(temp_dir) / "results",
                )

        self.assertFalse(response.ok)
        self.assertEqual(
            "Batch processing failed unexpectedly. Check the local log and try again.",
            response.status,
        )
        self.assertNotIn(str(secret_path), response.status)
        log_error.assert_called_once()
        self.assert_batch_response_has_no_artifacts(response)

    def test_malformed_batch_service_payload_is_contained_without_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_controller,
            "analyze_batch",
            return_value={},
        ), mock.patch.object(
            logging.getLogger("web_controller"), "exception"
        ) as log_error:
            response = self.call_without_ui_exception(
                run_batch_analysis,
                [Path(temp_dir) / "fixture.png"],
                5.0,
                60.0,
                Path(temp_dir) / "results",
            )

        self.assertFalse(response.ok)
        self.assertEqual(
            "Batch processing failed unexpectedly. Check the local log and try again.",
            response.status,
        )
        log_error.assert_called_once()
        self.assert_batch_response_has_no_artifacts(response)

    def test_batch_filesystem_error_uses_stable_results_folder_message(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            private_path = Path(temp_dir) / "results" / "private.tmp"
            with mock.patch.object(
                web_controller,
                "analyze_batch",
                side_effect=OSError(f"cannot write {private_path}"),
            ), mock.patch.object(
                logging.getLogger("web_controller"), "exception"
            ) as log_error:
                response = run_batch_analysis(
                    [Path(temp_dir) / "fixture.png"],
                    5.0,
                    60.0,
                    Path(temp_dir) / "results",
                )

        self.assertFalse(response.ok)
        self.assertEqual(
            "Could not write results. Check that the results folder is available and writable.",
            response.status,
        )
        self.assertNotIn(str(private_path), response.status)
        log_error.assert_called_once()
        self.assert_batch_response_has_no_artifacts(response)

    def test_batch_rejects_scalar_path_inputs_without_calling_service(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_controller,
            "analyze_batch",
        ) as analyze:
            root = Path(temp_dir)
            for paths in (str(root / "fixture.png"), root / "fixture.png"):
                with self.subTest(paths=paths):
                    response = run_batch_analysis(
                        paths,
                        5.0,
                        60.0,
                        root / "results",
                    )
                    self.assertFalse(response.ok)
                    self.assertEqual(
                        "Select images as a batch, not as a single path.",
                        response.status,
                    )
                    self.assert_batch_response_has_no_artifacts(response)
            analyze.assert_not_called()

    def test_batch_materializes_generator_once_before_calling_service(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = [root / "first.png", root / "second.png"]
            yields = []

            def one_shot_sources():
                for source in sources:
                    yields.append(source)
                    yield source

            with mock.patch.object(
                web_controller,
                "analyze_batch",
                side_effect=ValueError("service received materialized inputs"),
            ) as analyze, mock.patch.object(
                logging.getLogger("web_controller"), "exception"
            ) as log_error:
                response = run_batch_analysis(
                    one_shot_sources(),
                    5.0,
                    60.0,
                    root / "results",
                )

        self.assertFalse(response.ok)
        self.assertEqual(
            "Batch processing failed unexpectedly. Check the local log and try again.",
            response.status,
        )
        self.assertEqual(sources, yields)
        self.assertEqual(sources, analyze.call_args.args[0])
        log_error.assert_called_once()


class ResultFolderTests(unittest.TestCase):
    class FakeWindowsHandleApi:
        FILE_ATTRIBUTE_DIRECTORY = 0x00000010
        FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400

        def __init__(self, reparse_paths=(), final_paths=None):
            self.reparse_paths = set(reparse_paths)
            self.final_paths = final_paths or {}
            self.opened = []
            self.closed = []

        def open_directory(self, path):
            self.opened.append(path)
            return path

        def attributes(self, handle):
            attributes = self.FILE_ATTRIBUTE_DIRECTORY
            if handle in self.reparse_paths:
                attributes |= self.FILE_ATTRIBUTE_REPARSE_POINT
            return attributes

        def final_path(self, handle):
            return self.final_paths.get(handle, handle)

        def close(self, handle):
            self.closed.append(handle)

    def test_opens_contained_result_folder_with_platform_command(self):
        cases = (
            ("win32", "explorer", "C:\\results\\fixture_analysis", {}),
            ("darwin", "open", "/dev/fd/91", {"pass_fds": (91,)}),
            ("linux", "xdg-open", "/proc/self/fd/91", {"pass_fds": (91,)}),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "results"
            candidate = root / "fixture_analysis"
            candidate.mkdir(parents=True)
            for platform, executable, launch_path, popen_kwargs in cases:
                events = []

                @contextmanager
                def stable_directory(_root, _candidate, _platform):
                    events.append("handle-open")
                    yield (
                        launch_path,
                        popen_kwargs,
                        lambda _process: events.append("launch-complete"),
                    )
                    events.append("handle-closed")

                def launch(command, **kwargs):
                    events.append("popen")
                    self.assertEqual([executable, launch_path], command)
                    self.assertEqual(popen_kwargs, kwargs)

                with self.subTest(platform=platform), mock.patch.object(
                    web_controller.sys,
                    "platform",
                    platform,
                ), mock.patch.object(
                    web_controller,
                    "_open_stable_result_directory",
                    side_effect=stable_directory,
                    create=True,
                ), mock.patch.object(web_controller.subprocess, "Popen") as popen:
                    popen.side_effect = launch
                    message = open_result_folder(candidate, root)

                    self.assertEqual(
                        "Opened result folder: fixture_analysis",
                        message,
                    )
                    self.assertEqual(
                        [
                            "handle-open",
                            "popen",
                            "launch-complete",
                            "handle-closed",
                        ],
                        events,
                    )

    def test_rejects_lexical_sibling_before_candidate_filesystem_access(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = base / "results"
            sibling = base / "results-sibling"
            root.mkdir()
            sibling.mkdir()
            real_resolve = Path.resolve
            real_is_dir = Path.is_dir
            candidate_operations = []

            def guarded_resolve(path, *args, **kwargs):
                if os.fspath(path) == os.fspath(sibling):
                    candidate_operations.append("resolve")
                    raise AssertionError("off-root candidate was resolved")
                return real_resolve(path, *args, **kwargs)

            def guarded_is_dir(path):
                if os.fspath(path) == os.fspath(sibling):
                    candidate_operations.append("stat")
                    raise AssertionError("off-root candidate was statted")
                return real_is_dir(path)

            with mock.patch.object(Path, "resolve", guarded_resolve), mock.patch.object(
                Path, "is_dir", guarded_is_dir
            ), mock.patch.object(web_controller.subprocess, "Popen") as popen:
                message = open_result_folder(sibling, root)

            self.assertEqual("Result folder is unavailable.", message)
            self.assertEqual([], candidate_operations)
            popen.assert_not_called()

    def test_rejects_windows_other_drive_and_unc_before_candidate_access(self):
        candidates = (r"D:\outside\result", r"\\server\share\result")
        real_resolve = Path.resolve
        real_is_dir = Path.is_dir
        for candidate in candidates:
            candidate_operations = []

            def guarded_resolve(path, *args, **kwargs):
                if os.fspath(path) == candidate:
                    candidate_operations.append("resolve")
                    raise AssertionError("off-root candidate was resolved")
                return real_resolve(path, *args, **kwargs)

            def guarded_is_dir(path):
                if os.fspath(path) == candidate:
                    candidate_operations.append("stat")
                    raise AssertionError("off-root candidate was statted")
                return real_is_dir(path)

            with self.subTest(candidate=candidate), mock.patch.object(
                web_controller.sys, "platform", "win32"
            ), mock.patch.object(Path, "resolve", guarded_resolve), mock.patch.object(
                Path, "is_dir", guarded_is_dir
            ), mock.patch.object(web_controller.subprocess, "Popen") as popen:
                message = open_result_folder(candidate, r"C:\results")

            self.assertEqual("Result folder is unavailable.", message)
            self.assertEqual([], candidate_operations)
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

    def test_accepts_canonical_candidate_beneath_symlinked_root_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            canonical_root = base / "canonical-results"
            candidate = canonical_root / "fixture_analysis"
            alias_root = base / "results-alias"
            candidate.mkdir(parents=True)
            try:
                alias_root.symlink_to(canonical_root, target_is_directory=True)
            except OSError as exception:
                self.skipTest(f"Directory symlinks are unavailable: {exception}")
            with mock.patch.object(web_controller.subprocess, "Popen") as popen:
                message = open_result_folder(candidate.resolve(), alias_root)

            self.assertEqual("Opened result folder: fixture_analysis", message)
            popen.assert_called_once()

    def test_swap_of_root_ancestor_after_resolution_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            trusted_parent = base / "trusted"
            root = trusted_parent / "results"
            candidate = root / "fixture_analysis"
            moved_parent = base / "trusted-original"
            outside_parent = base / "outside"
            outside_candidate = outside_parent / "results" / "fixture_analysis"
            candidate.mkdir(parents=True)
            outside_candidate.mkdir(parents=True)
            real_resolve_candidate = web_controller._resolve_candidate_path
            swapped = False

            def resolve_then_swap(value):
                nonlocal swapped
                resolved = real_resolve_candidate(value)
                trusted_parent.rename(moved_parent)
                try:
                    trusted_parent.symlink_to(outside_parent, target_is_directory=True)
                except OSError as exception:
                    self.skipTest(
                        f"Directory symlinks are unavailable: {exception}"
                    )
                swapped = True
                return resolved

            with mock.patch.object(
                web_controller,
                "_resolve_candidate_path",
                side_effect=resolve_then_swap,
            ), mock.patch.object(web_controller.subprocess, "Popen") as popen:
                message = open_result_folder(candidate, root)

            self.assertTrue(swapped)
            self.assertEqual("Result folder is unavailable.", message)
            popen.assert_not_called()

    def test_windows_intermediate_reparse_handle_is_rejected_without_launch(self):
        root = PureWindowsPath(r"C:\trusted\results")
        candidate = root / "fixture_analysis"
        api = self.FakeWindowsHandleApi(reparse_paths={r"C:\trusted"})
        with mock.patch.object(
            web_controller,
            "_resolve_root_path",
            return_value=root,
        ), mock.patch.object(
            web_controller,
            "_resolve_candidate_path",
            return_value=candidate,
        ), mock.patch.object(
            web_controller,
            "_windows_handle_api",
            return_value=api,
        ), mock.patch.object(
            web_controller.sys,
            "platform",
            "win32",
        ), mock.patch.object(web_controller.subprocess, "Popen") as popen:
            message = open_result_folder(str(candidate), str(root))

        self.assertEqual("Result folder is unavailable.", message)
        self.assertIn(r"C:\trusted", api.opened)
        popen.assert_not_called()

    def test_windows_final_root_mismatch_is_rejected_without_launch(self):
        root = PureWindowsPath(r"C:\trusted\results")
        candidate = root / "fixture_analysis"
        api = self.FakeWindowsHandleApi(
            final_paths={
                str(root): r"D:\outside\results",
                str(candidate): r"D:\outside\results\fixture_analysis",
            }
        )
        with mock.patch.object(
            web_controller,
            "_resolve_root_path",
            return_value=root,
        ), mock.patch.object(
            web_controller,
            "_resolve_candidate_path",
            return_value=candidate,
        ), mock.patch.object(
            web_controller,
            "_windows_handle_api",
            return_value=api,
        ), mock.patch.object(
            web_controller.sys,
            "platform",
            "win32",
        ), mock.patch.object(web_controller.subprocess, "Popen") as popen:
            message = open_result_folder(str(candidate), str(root))

        self.assertEqual("Result folder is unavailable.", message)
        popen.assert_not_called()

    def test_windows_handles_remain_open_until_spawned_process_exits(self):
        root = PureWindowsPath(r"C:\trusted\results")
        candidate = root / "fixture_analysis"
        api = self.FakeWindowsHandleApi()
        process_started_waiting = threading.Event()
        allow_process_exit = threading.Event()
        all_handles_closed = threading.Event()
        original_close = api.close

        def record_close(handle):
            original_close(handle)
            if len(api.closed) == len(api.opened):
                all_handles_closed.set()

        api.close = record_close

        class BlockingProcess:
            def wait(self):
                process_started_waiting.set()
                if not allow_process_exit.wait(timeout=2):
                    raise TimeoutError("test child did not receive exit signal")
                return 0

        process = BlockingProcess()
        with mock.patch.object(
            web_controller,
            "_resolve_root_path",
            return_value=root,
        ), mock.patch.object(
            web_controller,
            "_resolve_candidate_path",
            return_value=candidate,
        ), mock.patch.object(
            web_controller,
            "_windows_handle_api",
            return_value=api,
        ), mock.patch.object(
            web_controller.sys,
            "platform",
            "win32",
        ), mock.patch.object(
            web_controller.subprocess,
            "Popen",
            return_value=process,
        ):
            message = open_result_folder(str(candidate), str(root))

        self.assertEqual("Opened result folder: fixture_analysis", message)
        self.assertEqual([], api.closed)
        self.assertTrue(process_started_waiting.wait(timeout=1))
        self.assertEqual([], api.closed)
        allow_process_exit.set()
        self.assertTrue(all_handles_closed.wait(timeout=1))
        self.assertEqual(list(reversed(api.opened)), api.closed)

    def test_windows_handles_close_when_process_wait_raises(self):
        root = PureWindowsPath(r"C:\trusted\results")
        candidate = root / "fixture_analysis"
        api = self.FakeWindowsHandleApi()
        wait_called = threading.Event()
        all_handles_closed = threading.Event()
        original_close = api.close

        def record_close(handle):
            original_close(handle)
            if len(api.closed) == len(api.opened):
                all_handles_closed.set()

        api.close = record_close

        class FailingProcess:
            def wait(self):
                wait_called.set()
                raise OSError("wait failed")

        with mock.patch.object(
            web_controller,
            "_resolve_root_path",
            return_value=root,
        ), mock.patch.object(
            web_controller,
            "_resolve_candidate_path",
            return_value=candidate,
        ), mock.patch.object(
            web_controller,
            "_windows_handle_api",
            return_value=api,
        ), mock.patch.object(
            web_controller.sys,
            "platform",
            "win32",
        ), mock.patch.object(
            web_controller.subprocess,
            "Popen",
            return_value=FailingProcess(),
        ), mock.patch.object(logging.getLogger("web_controller"), "exception"):
            message = open_result_folder(str(candidate), str(root))

        self.assertEqual("Opened result folder: fixture_analysis", message)
        self.assertTrue(wait_called.wait(timeout=1))
        self.assertTrue(all_handles_closed.wait(timeout=1))
        self.assertEqual(list(reversed(api.opened)), api.closed)

    def test_windows_handles_close_if_waiter_cannot_be_created(self):
        root = PureWindowsPath(r"C:\trusted\results")
        candidate = root / "fixture_analysis"
        api = self.FakeWindowsHandleApi()
        with mock.patch.object(
            web_controller,
            "_resolve_root_path",
            return_value=root,
        ), mock.patch.object(
            web_controller,
            "_resolve_candidate_path",
            return_value=candidate,
        ), mock.patch.object(
            web_controller,
            "_windows_handle_api",
            return_value=api,
        ), mock.patch.object(
            web_controller.sys,
            "platform",
            "win32",
        ), mock.patch.object(
            web_controller.subprocess,
            "Popen",
        ), mock.patch.object(
            web_controller.threading,
            "Thread",
            side_effect=RuntimeError("thread unavailable"),
        ):
            message = open_result_folder(str(candidate), str(root))

        self.assertEqual("Result folder is unavailable.", message)
        self.assertEqual(list(reversed(api.opened)), api.closed)

    def test_swap_to_symlink_between_resolve_and_open_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = base / "results"
            candidate = root / "fixture_analysis"
            moved = root / "fixture_original"
            outside = base / "outside"
            candidate.mkdir(parents=True)
            outside.mkdir()
            real_resolve = Path.resolve
            swapped = False

            def resolve_then_swap(path, *args, **kwargs):
                nonlocal swapped
                resolved = real_resolve(path, *args, **kwargs)
                if os.fspath(path) == os.fspath(candidate) and not swapped:
                    candidate.rename(moved)
                    try:
                        candidate.symlink_to(outside, target_is_directory=True)
                    except OSError as exception:
                        self.skipTest(
                            f"Directory symlinks are unavailable: {exception}"
                        )
                    swapped = True
                return resolved

            with mock.patch.object(Path, "resolve", resolve_then_swap), mock.patch.object(
                web_controller.subprocess, "Popen"
            ) as popen:
                message = open_result_folder(candidate, root)

            self.assertTrue(swapped)
            self.assertEqual("Result folder is unavailable.", message)
            popen.assert_not_called()

    def test_stable_handle_failure_does_not_launch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "results"
            candidate = root / "fixture_analysis"
            candidate.mkdir(parents=True)
            with mock.patch.object(
                web_controller,
                "_open_stable_result_directory",
                side_effect=OSError("no stable handle"),
                create=True,
            ), mock.patch.object(web_controller.subprocess, "Popen") as popen:
                message = open_result_folder(candidate, root)

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
