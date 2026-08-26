from contextlib import redirect_stdout
import importlib
import io
import runpy
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import app
from lan_access import LanAccessInfo
import web_launcher


def fake_application(local_url="http://127.0.0.1:7860/"):
    application = mock.Mock()
    application.launch.return_value = (
        mock.sentinel.gradio_app,
        local_url,
        None,
    )
    return application


class BrokenConsole:
    encoding = "utf-8"

    def write(self, _text):
        raise OSError("stdout closed")

    def flush(self):
        raise OSError("stdout closed")


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
    def test_launch_binds_lan_and_never_uses_gradio_sharing(self):
        fake_app = fake_application()
        access = LanAccessInfo(
            "http://127.0.0.1:7860",
            ("http://192.168.1.44:7860",),
            "http://192.168.1.44:7860",
        )
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_launcher,
            "create_app",
            return_value=fake_app,
        ) as create_app, mock.patch.object(
            web_launcher,
            "discover_lan_access",
            return_value=access,
        ):
            requested_root = Path(temp_dir) / "nested" / ".." / "results"

            web_launcher.launch_site(
                results_root=requested_root,
                port=7860,
                open_browser=False,
            )

            resolved_root = requested_root.resolve()
            create_app.assert_called_once_with(resolved_root, access)
            fake_app.launch.assert_called_once_with(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                inbrowser=False,
                prevent_thread_lock=True,
                show_error=True,
                allowed_paths=[str(resolved_root)],
                footer_links=[],
            )
            fake_app.block_thread.assert_called_once_with()
            self.assertTrue(resolved_root.is_dir())

    def test_launch_reserves_automatic_port_for_connection_metadata(self):
        fake_app = fake_application()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_launcher,
            "create_app",
            return_value=fake_app,
        ), mock.patch.object(
            web_launcher,
            "available_loopback_port",
            return_value=43123,
        ) as available_port, mock.patch.object(
            web_launcher.webbrowser,
            "open",
            return_value=True,
        ) as open_browser:
            web_launcher.launch_site(Path(temp_dir), None, True)

        available_port.assert_called_once_with()
        self.assertEqual(43123, fake_app.launch.call_args.kwargs["server_port"])
        self.assertFalse(fake_app.launch.call_args.kwargs["inbrowser"])
        open_browser.assert_called_once_with("http://127.0.0.1:43123")
        fake_app.block_thread.assert_called_once_with()

    def test_no_browser_never_opens_a_browser_and_blocks_normally(self):
        fake_app = fake_application()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_launcher,
            "create_app",
            return_value=fake_app,
        ), mock.patch.object(web_launcher.webbrowser, "open") as open_browser:
            web_launcher.launch_site(Path(temp_dir), None, False)

        open_browser.assert_not_called()
        fake_app.block_thread.assert_called_once_with()

    def test_false_browser_result_closes_started_application(self):
        fake_app = fake_application()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_launcher,
            "create_app",
            return_value=fake_app,
        ), mock.patch.object(
            web_launcher.webbrowser,
            "open",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "default browser"):
                web_launcher.launch_site(Path(temp_dir), None, True)

        fake_app.close.assert_called_once_with()
        fake_app.block_thread.assert_not_called()

    def test_browser_exception_closes_started_application(self):
        fake_app = fake_application()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_launcher,
            "create_app",
            return_value=fake_app,
        ), mock.patch.object(
            web_launcher.webbrowser,
            "open",
            side_effect=OSError("browser unavailable"),
        ):
            with self.assertRaisesRegex(OSError, "browser unavailable"):
                web_launcher.launch_site(Path(temp_dir), None, True)

        fake_app.close.assert_called_once_with()

    def test_gradio_launch_failure_closes_application(self):
        fake_app = fake_application()
        fake_app.launch.side_effect = LookupError("launch failed")
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_launcher,
            "create_app",
            return_value=fake_app,
        ):
            with self.assertRaisesRegex(LookupError, "launch failed"):
                web_launcher.launch_site(Path(temp_dir), None, False)

        fake_app.close.assert_called_once_with()

    def test_block_failure_closes_started_application(self):
        fake_app = fake_application()
        fake_app.block_thread.side_effect = ArithmeticError("block failed")
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_launcher,
            "create_app",
            return_value=fake_app,
        ):
            with self.assertRaisesRegex(ArithmeticError, "block failed"):
                web_launcher.launch_site(Path(temp_dir), None, False)

        fake_app.close.assert_called_once_with()

    def test_write_probe_does_not_remove_existing_user_data(self):
        fake_app = fake_application()
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
                fake_app = fake_application()
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

    def test_symlink_results_leaf_is_rejected_before_app_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            target = parent / "target"
            target.mkdir()
            results_root = parent / "results"
            try:
                results_root.symlink_to(target, target_is_directory=True)
            except OSError as exception:
                self.skipTest(f"Symlinks unavailable: {exception}")

            with mock.patch.object(web_launcher, "create_app") as create_app:
                with self.assertRaisesRegex(
                    ValueError,
                    "symlink, junction, or reparse point",
                ):
                    web_launcher.launch_site(results_root, None, False)

        create_app.assert_not_called()

    def test_junction_results_leaf_is_rejected_before_app_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            Path,
            "is_junction",
            return_value=True,
            create=True,
        ), mock.patch.object(web_launcher, "create_app") as create_app:
            with self.assertRaisesRegex(
                ValueError,
                "symlink, junction, or reparse point",
            ):
                web_launcher.launch_site(Path(temp_dir), None, False)

        create_app.assert_not_called()

    def test_reparse_results_leaf_is_rejected_before_app_creation(self):
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        file_status = SimpleNamespace(
            st_mode=stat.S_IFDIR,
            st_file_attributes=reparse_flag,
        )
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            Path,
            "lstat",
            return_value=file_status,
        ), mock.patch.object(web_launcher, "create_app") as create_app:
            with self.assertRaisesRegex(
                ValueError,
                "symlink, junction, or reparse point",
            ):
                web_launcher.launch_site(Path(temp_dir), None, False)

        create_app.assert_not_called()


class WebLauncherConsoleTests(unittest.TestCase):
    def test_non_ascii_output_uses_console_encoding_with_backslash_fallback(self):
        console = SimpleNamespace(encoding="ascii")
        with mock.patch.object(web_launcher.sys, "stdout", console), mock.patch(
            "builtins.print"
        ) as print_message:
            web_launcher._print_console("Failure: café")

        print_message.assert_called_once_with("Failure: caf\\xe9")


class WebLauncherAuditTests(unittest.TestCase):
    def test_audit_falls_back_and_preserves_unicode_traceback_and_timestamp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            blocked_parent = base / "blocked"
            blocked_parent.write_text("not a directory", encoding="utf-8")
            fallback = base / "fallback-logs"
            with mock.patch.object(
                web_launcher,
                "_audit_log_directories",
                return_value=[blocked_parent / "logs", fallback],
            ):
                try:
                    raise RuntimeError("启动失败 café")
                except RuntimeError as exception:
                    web_launcher._write_startup_audit(exception)

            log_path = fallback / "website-startup.log"
            content = log_path.read_text(encoding="utf-8")

        self.assertIn("启动失败 café", content)
        self.assertIn("Traceback (most recent call last):", content)
        self.assertRegex(content, r"\[\d{4}-\d{2}-\d{2}T")

    def test_audit_reaches_temp_fallback_if_launcher_and_home_are_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_launcher,
            "_launcher_directory",
            side_effect=OSError("launcher unavailable"),
        ), mock.patch.object(
            Path,
            "home",
            side_effect=RuntimeError("home unavailable"),
        ), mock.patch.object(
            web_launcher.tempfile,
            "gettempdir",
            return_value=temp_dir,
        ), mock.patch.dict(
            web_launcher.os.environ,
            {"LOCALAPPDATA": "", "APPDATA": ""},
        ):
            try:
                raise RuntimeError("startup failed")
            except RuntimeError as exception:
                web_launcher._write_startup_audit(exception)

            log_path = (
                Path(temp_dir)
                / "coagulation-analysis"
                / "logs"
                / "website-startup.log"
            )
            content = log_path.read_text(encoding="utf-8")

        self.assertIn("startup failed", content)

    def test_audit_skips_log_symlink_redirected_into_served_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            results_root = base / "results"
            results_root.mkdir()
            redirected_log = results_root / "exposed-startup.log"
            unsafe_logs = base / "unsafe-logs"
            unsafe_logs.mkdir()
            linked_log = unsafe_logs / "website-startup.log"
            try:
                linked_log.symlink_to(redirected_log)
            except OSError as exception:
                self.skipTest(f"Symlinks unavailable: {exception}")
            fallback_logs = base / "fallback-logs"

            with mock.patch.object(
                web_launcher,
                "_audit_log_directories",
                return_value=[unsafe_logs, fallback_logs],
            ):
                try:
                    raise RuntimeError("private traceback")
                except RuntimeError as exception:
                    web_launcher._write_startup_audit(exception, results_root)

            fallback_content = (
                fallback_logs / "website-startup.log"
            ).read_text(encoding="utf-8")

        self.assertFalse(redirected_log.exists())
        self.assertIn("private traceback", fallback_content)

    def test_unusable_results_root_is_audited_outside_served_tree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            results_root = base / "results"
            results_root.write_text("not a directory", encoding="utf-8")
            audit_dir = base / "private-logs"
            with mock.patch.object(
                web_launcher,
                "_audit_log_directories",
                return_value=[audit_dir],
            ), mock.patch.object(
                web_launcher,
                "_print_console",
            ) as print_console:
                exit_code = web_launcher.main(
                    ["--results-root", str(results_root), "--no-browser"]
                )

            log_path = audit_dir / "website-startup.log"
            content = log_path.read_text(encoding="utf-8")

        self.assertEqual(1, exit_code)
        print_console.assert_called_once()
        self.assertIn("Traceback (most recent call last):", content)
        self.assertNotEqual(results_root, log_path.parent)

    def test_unwritable_results_root_still_creates_fallback_audit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            results_root = base / "results"
            results_root.mkdir()
            audit_dir = base / "fallback-logs"
            with mock.patch.object(
                web_launcher.tempfile,
                "mkstemp",
                side_effect=PermissionError("read-only results"),
            ), mock.patch.object(
                web_launcher,
                "_audit_log_directories",
                return_value=[audit_dir],
            ), mock.patch.object(web_launcher, "_print_console"):
                exit_code = web_launcher.main(
                    ["--results-root", str(results_root), "--no-browser"]
                )

            content = (audit_dir / "website-startup.log").read_text(
                encoding="utf-8"
            )

        self.assertEqual(1, exit_code)
        self.assertIn("read-only results", content)

    def test_self_linked_results_root_still_writes_external_utf8_audit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            results_root = base / "自引用-results"
            try:
                results_root.symlink_to(results_root, target_is_directory=True)
            except OSError as exception:
                self.skipTest(f"Symlinks unavailable: {exception}")
            audit_dir = base / "private-logs"
            with mock.patch.object(
                web_launcher,
                "_audit_log_directories",
                return_value=[audit_dir],
            ), mock.patch.object(web_launcher, "_print_console"):
                exit_code = web_launcher.main(
                    ["--results-root", str(results_root), "--no-browser"]
                )

            log_path = audit_dir / "website-startup.log"
            content = log_path.read_bytes().decode("utf-8")

        self.assertEqual(1, exit_code)
        self.assertIn("symlink, junction, or reparse point", content)
        self.assertFalse(log_path.is_relative_to(results_root))


class WebLauncherMainTests(unittest.TestCase):
    def test_main_forwards_switches_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            web_launcher,
            "launch_site",
        ) as launch_site, mock.patch.object(
            web_launcher,
            "_write_startup_audit",
        ) as write_audit:
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
        write_audit.assert_not_called()

    def test_main_reports_each_startup_error_exactly_once(self):
        for exception in (
            OSError("read-only"),
            RuntimeError("server unavailable"),
            ValueError("Port must be between 1 and 65535."),
            LookupError("unexpected Gradio failure"),
        ):
            with self.subTest(exception=exception), mock.patch.object(
                web_launcher,
                "launch_site",
                side_effect=exception,
            ), mock.patch.object(
                web_launcher,
                "_print_console",
            ) as print_console, mock.patch.object(
                web_launcher,
                "_write_startup_audit",
            ) as write_audit:
                exit_code = web_launcher.main(["--no-browser"])

            self.assertEqual(1, exit_code)
            print_console.assert_called_once_with(
                f"Website startup failed: {exception}"
            )
            write_audit.assert_called_once()
            self.assertIs(exception, write_audit.call_args.args[0])
            self.assertEqual(
                web_launcher.default_results_root(),
                write_audit.call_args.args[1],
            )

    def test_main_catches_an_out_of_range_port(self):
        with mock.patch.object(
            web_launcher, "_print_console"
        ) as print_console, mock.patch.object(
            web_launcher,
            "_write_startup_audit",
        ) as write_audit:
            exit_code = web_launcher.main(["--port", "0", "--no-browser"])

        self.assertEqual(1, exit_code)
        print_console.assert_called_once_with(
            "Website startup failed: Port must be between 1 and 65535."
        )
        write_audit.assert_called_once()

    def test_main_contains_audit_write_failure(self):
        exception = RuntimeError("server unavailable")
        with mock.patch.object(
            web_launcher,
            "launch_site",
            side_effect=exception,
        ), mock.patch.object(
            web_launcher,
            "_print_console",
        ) as print_console, mock.patch.object(
            web_launcher,
            "_write_startup_audit",
            side_effect=OSError("log unavailable"),
        ):
            exit_code = web_launcher.main(["--no-browser"])

        self.assertEqual(1, exit_code)
        print_console.assert_called_once_with(
            "Website startup failed: server unavailable"
        )

    def test_broken_stdout_does_not_suppress_original_audit_or_return_code(self):
        original_exception = RuntimeError("原始启动错误")
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_dir = Path(temp_dir) / "private-logs"
            with mock.patch.object(
                web_launcher,
                "launch_site",
                side_effect=original_exception,
            ), mock.patch.object(
                web_launcher,
                "_audit_log_directories",
                return_value=[audit_dir],
            ), mock.patch.object(
                web_launcher.sys,
                "stdout",
                BrokenConsole(),
            ):
                exit_code = web_launcher.main(["--no-browser"])

            content = (audit_dir / "website-startup.log").read_text(
                encoding="utf-8"
            )

        self.assertEqual(1, exit_code)
        self.assertIn("原始启动错误", content)
        self.assertNotIn("stdout closed", content)

    def test_default_results_path_failure_is_contained(self):
        exception = OSError("launcher path unavailable")
        with mock.patch.object(
            web_launcher,
            "default_results_root",
            side_effect=exception,
        ), mock.patch.object(
            web_launcher,
            "_print_console",
        ) as print_console, mock.patch.object(
            web_launcher,
            "_write_startup_audit",
        ) as write_audit:
            exit_code = web_launcher.main([])

        self.assertEqual(1, exit_code)
        print_console.assert_called_once_with(
            "Website startup failed: launcher path unavailable"
        )
        write_audit.assert_called_once_with(exception, None)

    def test_invalid_port_text_is_handled_without_argparse_exit(self):
        with mock.patch.object(
            web_launcher,
            "_print_console",
        ) as print_console, mock.patch.object(
            web_launcher,
            "_write_startup_audit",
        ) as write_audit:
            exit_code = web_launcher.main(["--port", "not-a-number"])

        self.assertEqual(1, exit_code)
        self.assertIn("invalid int value", print_console.call_args.args[0])
        write_audit.assert_called_once()

    def test_unknown_option_is_handled_without_argparse_exit(self):
        with mock.patch.object(
            web_launcher,
            "_print_console",
        ) as print_console, mock.patch.object(
            web_launcher,
            "_write_startup_audit",
        ) as write_audit:
            exit_code = web_launcher.main(["--unknown"])

        self.assertEqual(1, exit_code)
        self.assertIn("unrecognized arguments: --unknown", print_console.call_args.args[0])
        write_audit.assert_called_once()

    def test_help_retains_normal_zero_system_exit(self):
        with redirect_stdout(io.StringIO()) as output, mock.patch.object(
            web_launcher,
            "launch_site",
        ) as launch_site:
            with self.assertRaises(SystemExit) as exit_context:
                web_launcher.main(["--help"])

        self.assertEqual(0, exit_context.exception.code)
        self.assertIn("Start the offline coagulation website.", output.getvalue())
        launch_site.assert_not_called()


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

    def test_main_guard_preserves_help_system_exit(self):
        with mock.patch.object(
            sys,
            "argv",
            ["web_launcher.py", "--help"],
        ), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_context:
                runpy.run_module("web_launcher", run_name="__main__")

        self.assertEqual(0, exit_context.exception.code)


if __name__ == "__main__":
    unittest.main()
