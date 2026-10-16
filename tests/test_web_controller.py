from contextlib import contextmanager
import importlib
import json
import logging
import os
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from pathlib import Path, PureWindowsPath
from unittest import mock

import cv2
from fastapi.testclient import TestClient
from gradio import routes as gradio_routes

import app as web_app
from app import create_app
from lan_access import LanAccessInfo
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


class WebApplicationTests(unittest.TestCase):
    def make_config(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        application = create_app(Path(temp_dir.name) / "results")
        return application.get_config_file()

    def components_with_label(self, config, label):
        return [
            component
            for component in config["components"]
            if component.get("props", {}).get("label") == label
        ]

    def one_component(self, config, label):
        components = self.components_with_label(config, label)
        self.assertEqual(1, len(components), label)
        return components[0]

    def component_name(self, components_by_id, component_id):
        component = components_by_id[component_id]
        props = component.get("props", {})
        return props.get("label", props.get("value", component["type"]))

    def dependency_for_button(self, config, button_value):
        components_by_id = {
            component["id"]: component for component in config["components"]
        }
        button = next(
            component
            for component in config["components"]
            if component["type"] == "button"
            and component.get("props", {}).get("value") == button_value
        )
        dependencies = [
            dependency
            for dependency in config["dependencies"]
            if any(
                target[0] == button["id"] and target[1] == "click"
                for target in dependency["targets"]
            )
        ]
        self.assertEqual(1, len(dependencies), button_value)
        dependency = dependencies[0]
        return (
            dependency,
            [
                self.component_name(components_by_id, component_id)
                for component_id in dependency["inputs"]
            ],
            [
                self.component_name(components_by_id, component_id)
                for component_id in dependency["outputs"]
            ],
        )

    def test_create_app_builds_without_starting_a_server(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app = create_app(Path(temp_dir) / "results")

        self.assertIsNotNone(app)
        serialized = str(app.get_config_file())
        for label in (
            "Single Image",
            "Batch Processing",
            "Inner crop inset",
            "No-clot threshold",
            "Analyze Image",
            "Analyze Batch",
        ):
            with self.subTest(label=label):
                self.assertIn(label, serialized)

    def test_connection_panel_contains_private_url_and_local_qr(self):
        access = LanAccessInfo(
            loopback_url="http://127.0.0.1:7860",
            phone_urls=("http://192.168.1.44:7860",),
            preferred_url="http://192.168.1.44:7860",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            application = create_app(Path(temp_dir) / "results", access)
            config = application.get_config_file()

        serialized = str(config)
        self.assertIn("lan-connection-panel", serialized)
        self.assertIn("http://192.168.1.44:7860", serialized)
        self.assertIn("trusted private Wi-Fi", serialized)
        self.assertIn("No password", serialized)
        self.assertIn("data:image/png;base64,", serialized)

    def test_connection_panel_without_lan_has_restart_instructions_and_no_qr(self):
        access = LanAccessInfo(
            loopback_url="http://127.0.0.1:7860",
            phone_urls=(),
            preferred_url=None,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            application = create_app(Path(temp_dir) / "results", access)
            serialized = str(application.get_config_file())

        self.assertIn("Connect this PC and phone to the same Wi-Fi", serialized)
        self.assertNotIn("data:image/png;base64,", serialized)

    def test_single_adapter_returns_all_controller_values_in_output_order(self):
        adapter = getattr(web_app, "_single_values", None)
        self.assertIsNotNone(adapter)
        response = SimpleNamespace(
            crops=[],
            overlay_path=None,
            heatmap_path=None,
            rows=[],
            csv_path=None,
            zip_path=None,
            output_dir=None,
            status="Select an image to analyze.",
        )
        with mock.patch.object(
            web_app,
            "run_single_analysis",
            return_value=response,
        ) as run:
            values = adapter(None, 5, 60, Path("results"))

        run.assert_called_once_with(None, 5, 60, Path("results"))
        self.assertEqual(
            ([], None, None, [], None, None, None, response.status),
            values,
        )

    def test_batch_adapter_normalizes_empty_input_and_orders_outputs(self):
        adapter = getattr(web_app, "_batch_values", None)
        self.assertIsNotNone(adapter)
        response = SimpleNamespace(
            rows=[],
            summary_csv=None,
            failures_csv=None,
            zip_path=None,
            batch_dir=None,
            status="Select at least one image for batch processing.",
        )
        with mock.patch.object(
            web_app,
            "run_batch_analysis",
            return_value=response,
        ) as run:
            values = adapter(None, 5, 60, Path("results"))

        run.assert_called_once_with([], 5, 60, Path("results"))
        self.assertEqual(
            ([], None, None, None, None, response.status),
            values,
        )

    def test_single_tab_components_follow_browser_contract(self):
        config = self.make_config()

        for label, elem_id in (
            ("Choose from gallery or files", "gallery-source"),
            ("Take photo", "camera-source"),
        ):
            source = self.one_component(config, label)
            self.assertEqual("file", source["type"])
            self.assertEqual("filepath", source["props"]["type"])
            self.assertEqual("single", source["props"]["file_count"])
            self.assertEqual(elem_id, source["props"]["elem_id"])

        gallery = self.one_component(config, "Final inner crops")
        self.assertEqual("gallery", gallery["type"])
        self.assertEqual(3, gallery["props"]["columns"])
        self.assertEqual(3, gallery["props"]["rows"])

        for label in ("Detected and final boundaries", "Publication heatmap"):
            preview = self.one_component(config, label)
            self.assertEqual("image", preview["type"])
            self.assertEqual("filepath", preview["props"]["type"])

        table = self.one_component(config, "Per-cell results")
        self.assertEqual(
            ["Cell", "Row", "Column", "Mean", "Confidence", "Recovered"],
            table["props"]["headers"],
        )
        self.assertFalse(table["props"]["interactive"])

        for label in ("Status", "Saved result folder"):
            self.assertFalse(self.one_component(config, label)["props"]["interactive"])
        for label in ("Download CSV", "Download result ZIP"):
            self.assertEqual("file", self.one_component(config, label)["type"])

    def test_batch_tab_components_follow_browser_contract(self):
        config = self.make_config()

        sources = self.one_component(config, "Complete 3×3 fixture images")
        self.assertEqual("file", sources["type"])
        self.assertEqual("filepath", sources["props"]["type"])
        self.assertEqual("multiple", sources["props"]["file_count"])

        table = self.one_component(config, "Batch results")
        self.assertEqual(
            ["Image", "Cells", "Status", "Reason", "Result"],
            table["props"]["headers"],
        )
        self.assertFalse(table["props"]["interactive"])

        for label in ("Batch status", "Saved batch folder"):
            self.assertFalse(self.one_component(config, label)["props"]["interactive"])
        for label in (
            "Batch summary CSV",
            "Failure report CSV",
            "Download batch ZIP",
        ):
            self.assertEqual("file", self.one_component(config, label)["type"])

    def test_callbacks_have_exact_wiring_and_independent_tab_settings(self):
        config = self.make_config()

        single, single_inputs, single_outputs = self.dependency_for_button(
            config,
            "Analyze Image",
        )
        self.assertEqual(
            [
                "Choose from gallery or files",
                "Take photo",
                "Inner crop inset",
                "No-clot threshold",
            ],
            single_inputs,
        )
        self.assertEqual(
            [
                "Final inner crops",
                "Detected and final boundaries",
                "Publication heatmap",
                "Per-cell results",
                "Download CSV",
                "Download result ZIP",
                "Saved result folder",
                "Status",
            ],
            single_outputs,
        )

        batch, batch_inputs, batch_outputs = self.dependency_for_button(
            config,
            "Analyze Batch",
        )
        self.assertEqual(
            [
                "Complete 3×3 fixture images",
                "Inner crop inset",
                "No-clot threshold",
            ],
            batch_inputs,
        )
        self.assertEqual(
            [
                "Batch results",
                "Batch summary CSV",
                "Failure report CSV",
                "Download batch ZIP",
                "Saved batch folder",
                "Batch status",
            ],
            batch_outputs,
        )
        self.assertTrue(single["backend_fn"])
        self.assertTrue(batch["backend_fn"])
        self.assertTrue(set(single["inputs"][1:]).isdisjoint(batch["inputs"][1:]))

        _, folder_inputs, folder_outputs = self.dependency_for_button(
            config,
            "Open folder on Windows PC",
        )
        self.assertEqual(["Saved result folder"], folder_inputs)
        self.assertEqual(["Status"], folder_outputs)
        _, folder_inputs, folder_outputs = self.dependency_for_button(
            config,
            "Open batch folder on Windows PC",
        )
        self.assertEqual(["Saved batch folder"], folder_inputs)
        self.assertEqual(["Batch status"], folder_outputs)

        for label, expected in (
            ("Inner crop inset", (0, 15, 5, 0.5)),
            ("No-clot threshold", (0, 255, 60, 1)),
        ):
            sliders = self.components_with_label(config, label)
            self.assertEqual(2, len(sliders), label)
            self.assertEqual(2, len({slider["id"] for slider in sliders}))
            for slider in sliders:
                props = slider["props"]
                self.assertEqual(
                    expected,
                    (
                        props["minimum"],
                        props["maximum"],
                        props["value"],
                        props["step"],
                    ),
                )

    def test_single_source_accepts_one_input_and_rejects_ambiguous_input(self):
        selector = getattr(web_app, "_single_source", None)
        self.assertIsNotNone(selector)
        self.assertEqual("gallery.png", selector("gallery.png", None))
        self.assertEqual("camera.jpg", selector(None, "camera.jpg"))
        with self.assertRaisesRegex(ValueError, "Choose or take"):
            selector(None, None)
        with self.assertRaisesRegex(ValueError, "only one"):
            selector("gallery.png", "camera.jpg")

    def test_camera_bootstrap_is_local_and_uses_native_capture(self):
        bootstrap = getattr(web_app, "_MOBILE_CAPTURE_BOOTSTRAP", "")
        self.assertIn("#camera-source", bootstrap)
        self.assertIn("capture", bootstrap)
        self.assertIn("environment", bootstrap)
        self.assertIn("image/*", bootstrap)
        self.assertNotIn("getUserMedia", bootstrap)
        self.assertNotIn("https://", bootstrap)

    def test_mobile_css_overrides_gradio_minimum_widths(self):
        css = web_app._load_local_styles()
        mobile = css[css.index("@media (max-width: 720px)") :]
        self.assertIn("\n@media (max-width: 720px)", css)
        self.assertNotIn("\n+@media", css)
        self.assertIn("box-sizing: border-box !important", mobile)
        self.assertIn("min-width: 0 !important", mobile)
        self.assertIn("flex-direction: column", mobile)

    def test_single_analysis_has_stable_api_name(self):
        config = self.make_config()
        api_names = {
            dependency.get("api_name")
            for dependency in config["dependencies"]
        }
        self.assertIn("analyze_single", api_names)

    def test_application_is_offline_and_includes_local_styles(self):
        config = self.make_config()

        self.assertEqual("Coagulation Analysis", config["title"])
        self.assertFalse(config["analytics_enabled"])
        self.assertEqual("False", os.environ["GRADIO_ANALYTICS_ENABLED"])
        markdown_values = [
            component.get("props", {}).get("value")
            for component in config["components"]
            if component["type"] == "markdown"
        ]
        self.assertIn("# Coagulation Analysis\nLocal and offline", markdown_values)

        stylesheet = config["css"]
        self.assertIsInstance(stylesheet, str)
        for value in (
            "#182a45",
            "#3f78b5",
            "#a42e3d",
            "#ffffff",
            "#f5f7fa",
            "max-width: 1440px",
        ):
            with self.subTest(value=value):
                self.assertIn(value, stylesheet)
        for external_reference in ("http://", "https://", "@import", "url("):
            with self.subTest(external_reference=external_reference):
                self.assertNotIn(external_reference, stylesheet)

    def test_title_and_status_styles_override_gradio_component_defaults(self):
        config = self.make_config()
        stylesheet = config["css"]
        title = next(
            component
            for component in config["components"]
            if component.get("props", {}).get("elem_id") == "application-title"
        )

        self.assertEqual("markdown", title["type"])
        self.assertRegex(
            stylesheet,
            r"(?s)#application-title h1\s*\{[^}]*color:\s*var\(--navy\)\s*!important",
        )
        self.assertRegex(
            stylesheet,
            r"(?s)body \.gradio-container div\.status-field\.block\s*\{[^}]*border-left:\s*4px solid var\(--blue\)\s*!important",
        )
        for label in ("Status", "Batch status"):
            classes = self.one_component(config, label)["props"]["elem_classes"]
            self.assertIn("status-field", classes)

    def test_config_uses_system_fonts_and_contains_no_remote_resources(self):
        config = self.make_config()
        serialized = json.dumps(config)

        self.assertEqual([], config["stylesheets"])
        self.assertEqual([], config["footer_links"])
        self.assertNotIn("http://", serialized)
        self.assertNotIn("https://", serialized)
        self.assertEqual("base", config["theme"])
        theme = getattr(web_app, "_LOCAL_THEME", None)
        self.assertIsNotNone(theme)
        self.assertIn("system-ui", theme.font)
        self.assertNotIn("http://", theme._get_theme_css())
        self.assertNotIn("https://", theme._get_theme_css())

    def test_server_templates_are_local_and_bootstrap_english_before_module(self):
        config = self.make_config()
        config.update(
            body_css=config["body_css"] or {},
            thumbnail=None,
            simple_description="",
        )
        api_info = {
            "named_endpoints": {
                "/endpoint-marker": {
                    "parameters": [],
                    "returns": [],
                    "code_snippets": {
                        "python": 'Client("http://127.0.0.1:7860")',
                    },
                },
            },
            "unnamed_endpoints": {},
        }

        for template_name in ("frontend/index.html", "frontend/share.html"):
            with self.subTest(template_name=template_name):
                html = gradio_routes.templates.get_template(template_name).render(
                    config=config,
                    gradio_api_info=api_info,
                )
                lowered = html.lower()
                self.assertNotIn("http://", lowered)
                self.assertNotIn("https://", lowered)
                for domain in (
                    "fonts.googleapis.com",
                    "fonts.gstatic.com",
                    "cdnjs.cloudflare.com",
                    "gradio.app",
                ):
                    self.assertNotIn(domain, lowered)
                self.assertIn("window.gradio_config", html)
                self.assertIn('type="module"', html)
                self.assertIn("./assets/", html)
                bootstrap_position = html.index("data-offline-language")
                module_position = html.index('type="module"')
                self.assertLess(bootstrap_position, module_position)
                self.assertIn('document.documentElement.lang = "en"', html)
                self.assertIn(
                    "window.gradio_config.root = window.location.origin",
                    html,
                )
                self.assertIn('navigator, "language"', html)
                self.assertIn('navigator, "languages"', html)
                self.assertIn("data-mobile-layout", html)
                self.assertIn("width: 100vw !important", html)
                self.assertIn("overflow-x: hidden", html)
                self.assertIn("endpoint-marker", html)
                self.assertNotIn("code_snippets", html)

    def test_offline_template_installation_is_thread_safe_and_idempotent(self):
        installer = getattr(web_app, "_install_offline_templates", None)
        self.assertIsNotNone(installer)
        environment = gradio_routes.templates.env
        installed_loader = environment.loader
        base_loader = installed_loader
        while getattr(base_loader, "_coagulation_offline_templates", False):
            base_loader = base_loader.loaders[-1]

        errors = []
        barrier = threading.Barrier(8)

        def install_at_once():
            try:
                barrier.wait()
                installer()
            except BaseException as exception:
                errors.append(exception)

        try:
            environment.loader = base_loader
            environment.cache.clear()
            threads = [threading.Thread(target=install_at_once) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual([], errors)
            concurrent_loader = environment.loader
            self.assertTrue(
                getattr(concurrent_loader, "_coagulation_offline_templates", False)
            )
            installer()
            self.assertIs(concurrent_loader, environment.loader)
        finally:
            environment.loader = installed_loader
            environment.cache.clear()

    def test_live_server_html_and_theme_have_no_http_resources(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        application = create_app(Path(temp_dir.name) / "results")
        server = gradio_routes.App.create_app(application)

        with TestClient(server) as client:
            response = client.get(
                "/",
                headers={"Accept-Language": "zh-CN,zh;q=0.9"},
            )
            theme = client.get("/theme.css")
            configuration = client.get("/config")

        self.assertEqual(200, response.status_code)
        self.assertEqual(200, theme.status_code)
        self.assertEqual(200, configuration.status_code)
        for payload in (
            response.text.lower(),
            theme.text.lower(),
            configuration.text.lower(),
        ):
            self.assertNotIn("http://", payload)
            self.assertNotIn("https://", payload)
        self.assertEqual("", configuration.json()["root"])
        self.assertIn("./assets/", response.text)
        self.assertIn("data-offline-language", response.text)

    def test_frozen_app_uses_identical_embedded_css_when_file_is_missing(self):
        stylesheet_path = Path(web_app.__file__).with_name("web_styles.css")
        expected = stylesheet_path.read_text(encoding="utf-8")
        original_read_text = Path.read_text

        def read_without_sibling(path, *args, **kwargs):
            if path == stylesheet_path:
                raise FileNotFoundError(path)
            return original_read_text(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            sys,
            "frozen",
            True,
            create=True,
        ), mock.patch.object(
            Path,
            "read_text",
            autospec=True,
            side_effect=read_without_sibling,
        ):
            try:
                application = create_app(Path(temp_dir) / "results")
            except FileNotFoundError as exception:
                self.fail(f"Frozen application required sibling CSS: {exception}")

        self.assertEqual(expected, application.get_config_file()["css"])

    def test_importing_app_does_not_start_a_server(self):
        with mock.patch.object(web_app.gr.Blocks, "launch") as launch:
            importlib.reload(web_app)

        launch.assert_not_called()

    def test_main_launches_the_offline_site_on_loopback(self):
        main = getattr(web_app, "main", None)
        self.assertIsNotNone(main)
        application = mock.Mock()
        working_directory = Path("/tmp/coagulation-website")
        with mock.patch.object(
            web_app.Path,
            "cwd",
            return_value=working_directory,
        ), mock.patch.object(
            web_app,
            "create_app",
            return_value=application,
        ) as create:
            main()

        create.assert_called_once_with(working_directory / "results")
        application.launch.assert_called_once_with(
            server_name="127.0.0.1",
            share=False,
            inbrowser=True,
            footer_links=[],
        )


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
