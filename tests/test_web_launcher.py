import importlib
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import app
import web_launcher


class WebLauncherPathTests(unittest.TestCase):
    def test_default_results_root_is_beside_packaged_launcher(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executable = Path(temp_dir) / "LabApp" / "StartWebsite.exe"
            expected = executable.resolve().parent / "results"
            with mock.patch.object(
                web_launcher.sys, "frozen", True, create=True
            ), mock.patch.object(
                web_launcher.sys,
                "executable",
                str(executable),
            ):
                actual = web_launcher.default_results_root()

            self.assertEqual(
                expected,
                actual,
            )

    def test_default_results_root_is_beside_development_module(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            module_path = Path(temp_dir) / "source" / "web_launcher.py"
            expected = module_path.resolve().parent / "results"
            with mock.patch.object(
                web_launcher.sys, "frozen", False, create=True
            ), mock.patch.object(
                web_launcher,
                "__file__",
                str(module_path),
            ):
                actual = web_launcher.default_results_root()

            self.assertEqual(
                expected,
                actual,
            )

    def test_available_port_is_valid_and_released_for_loopback_use(self):
        port = web_launcher.available_loopback_port()

        self.assertGreaterEqual(port, 1)
        self.assertLessEqual(port, 65535)
        with web_launcher.socket.socket(
            web_launcher.socket.AF_INET,
            web_launcher.socket.SOCK_STREAM,
        ) as server:
            server.bind(("127.0.0.1", port))


class WebLauncherStartupTests(unittest.TestCase):
    def test_launch_is_loopback_only_and_never_shared(self):
        fake_app = mock.Mock()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_launcher,
            "create_app",
            return_value=fake_app,
        ) as create_app:
            requested_root = Path(temp_dir) / "nested" / ".." / "results"

            web_launcher.launch_site(
                results_root=requested_root,
                port=7860,
                open_browser=False,
            )

            resolved_root = requested_root.resolve()
            create_app.assert_called_once_with(resolved_root)
            fake_app.launch.assert_called_once_with(
                server_name="127.0.0.1",
                server_port=7860,
                share=False,
                inbrowser=False,
                show_error=True,
                allowed_paths=[str(resolved_root)],
                footer_links=[],
            )
            self.assertTrue(resolved_root.is_dir())

    def test_launch_selects_an_available_port_when_one_is_not_supplied(self):
        fake_app = mock.Mock()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_launcher,
            "create_app",
            return_value=fake_app,
        ), mock.patch.object(
            web_launcher,
            "available_loopback_port",
            return_value=43123,
        ) as available_port:
            web_launcher.launch_site(Path(temp_dir), None, True)

        available_port.assert_called_once_with()
        self.assertEqual(
            43123,
            fake_app.launch.call_args.kwargs["server_port"],
        )
        self.assertTrue(fake_app.launch.call_args.kwargs["inbrowser"])

    def test_write_probe_does_not_remove_existing_user_data(self):
        fake_app = mock.Mock()
        with tempfile.TemporaryDirectory() as temp_dir:
            results_root = Path(temp_dir) / "results"
            results_root.mkdir()
            sentinel = results_root / "existing-result.csv"
            sentinel.write_text("user data", encoding="utf-8")

            with mock.patch.object(
                web_launcher,
                "create_app",
                return_value=fake_app,
            ):
                web_launcher.launch_site(results_root, 7860, False)

            self.assertEqual("user data", sentinel.read_text(encoding="utf-8"))
            self.assertEqual([sentinel], list(results_root.iterdir()))

    def test_launch_fails_before_creating_app_for_unwritable_root(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_launcher.tempfile,
            "mkstemp",
            side_effect=PermissionError("read-only"),
        ), mock.patch.object(web_launcher, "create_app") as create_app:
            with self.assertRaisesRegex(PermissionError, "read-only"):
                web_launcher.launch_site(Path(temp_dir), 7860, False)

        create_app.assert_not_called()

    def test_launch_rejects_invalid_explicit_ports_before_touching_disk(self):
        invalid_ports = (-1, 0, 65536)
        for port in invalid_ports:
            with self.subTest(port=port), tempfile.TemporaryDirectory() as temp_dir:
                results_root = Path(temp_dir) / "not-created"
                with self.assertRaisesRegex(
                    ValueError,
                    "Port must be between 1 and 65535.",
                ):
                    web_launcher.launch_site(results_root, port, False)
                self.assertFalse(results_root.exists())

    def test_launch_accepts_both_explicit_port_boundaries(self):
        for port in (1, 65535):
            with self.subTest(port=port), tempfile.TemporaryDirectory() as temp_dir:
                fake_app = mock.Mock()
                with mock.patch.object(
                    web_launcher,
                    "create_app",
                    return_value=fake_app,
                ):
                    web_launcher.launch_site(Path(temp_dir), port, False)

                self.assertEqual(
                    port,
                    fake_app.launch.call_args.kwargs["server_port"],
                )


class WebLauncherConsoleTests(unittest.TestCase):
    def test_non_ascii_output_uses_console_encoding_with_backslash_fallback(self):
        console = SimpleNamespace(encoding="ascii")
        with mock.patch.object(web_launcher.sys, "stdout", console), mock.patch(
            "builtins.print"
        ) as print_message:
            web_launcher._print_console("Failure: café")

        print_message.assert_called_once_with("Failure: caf\\xe9")


class WebLauncherMainTests(unittest.TestCase):
    def test_main_forwards_switches_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_launcher,
            "launch_site",
        ) as launch_site:
            exit_code = web_launcher.main(
                [
                    "--port",
                    "8123",
                    "--no-browser",
                    "--results-root",
                    temp_dir,
                ]
            )

        self.assertEqual(0, exit_code)
        launch_site.assert_called_once_with(Path(temp_dir), 8123, False)

    def test_main_reports_each_startup_error_exactly_once(self):
        for exception in (
            OSError("read-only"),
            RuntimeError("server unavailable"),
            ValueError("Port must be between 1 and 65535."),
        ):
            with self.subTest(exception=exception), mock.patch.object(
                web_launcher,
                "launch_site",
                side_effect=exception,
            ), mock.patch.object(
                web_launcher,
                "_print_console",
            ) as print_console:
                exit_code = web_launcher.main(["--no-browser"])

            self.assertEqual(1, exit_code)
            print_console.assert_called_once_with(
                f"Website startup failed: {exception}"
            )

    def test_main_catches_an_out_of_range_port(self):
        with mock.patch.object(web_launcher, "_print_console") as print_console:
            exit_code = web_launcher.main(["--port", "0", "--no-browser"])

        self.assertEqual(1, exit_code)
        print_console.assert_called_once_with(
            "Website startup failed: Port must be between 1 and 65535."
        )


class WebLauncherImportTests(unittest.TestCase):
    def test_launcher_import_does_not_build_or_launch_the_site(self):
        original_module = sys.modules.pop("web_launcher")
        self.addCleanup(sys.modules.__setitem__, "web_launcher", original_module)
        try:
            with mock.patch.object(app, "create_app") as create_app:
                imported_module = importlib.import_module("web_launcher")
        finally:
            sys.modules["web_launcher"] = original_module

        self.assertIsNotNone(imported_module)
        create_app.assert_not_called()

    def test_main_guard_exits_with_main_return_code(self):
        with mock.patch.object(
            sys,
            "argv",
            ["web_launcher.py", "--port", "0", "--no-browser"],
        ), mock.patch("builtins.print"):
            with self.assertRaises(SystemExit) as exit_context:
                runpy.run_module("web_launcher", run_name="__main__")

        self.assertEqual(1, exit_context.exception.code)


if __name__ == "__main__":
    unittest.main()
