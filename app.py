"""Local Gradio interface for fixed nine-grid coagulation analysis."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import threading

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
from gradio import route_utils as gradio_route_utils
from gradio import routes as gradio_routes
from jinja2 import ChoiceLoader, DictLoader

from web_controller import open_result_folder, run_batch_analysis, run_single_analysis


_SYSTEM_FONT_STACK = (
    "ui-sans-serif",
    "system-ui",
    "-apple-system",
    "BlinkMacSystemFont",
    "Segoe UI",
    "sans-serif",
)
_SYSTEM_MONO_STACK = (
    "ui-monospace",
    "SFMono-Regular",
    "Consolas",
    "monospace",
)
_LOCAL_THEME = gr.themes.Base(
    font=_SYSTEM_FONT_STACK,
    font_mono=_SYSTEM_MONO_STACK,
).set(
    checkbox_check="none",
    radio_circle="none",
)
_TEMPLATE_NAMES = ("frontend/index.html", "frontend/share.html")
_EXTERNAL_HEAD_TAG = re.compile(
    r"""
    (?is)
    [ \t]*<(?:link|meta)\b
    (?=[^>]*\b(?:href|content)\s*=\s*["']https?://)
    [^>]*>\s*
    |
    [ \t]*<script\b
    (?=[^>]*\bsrc\s*=\s*["']https?://)
    [^>]*>\s*</script>\s*
    """,
    re.VERBOSE,
)
_LOCAL_ASSET_TAG = re.compile(
    r"""(?is)<(?:script|link)\b(?=[^>]*(?:src|href)=["']\./assets/)[^>]*>"""
    r"(?:\s*</script>)?"
)
_ENGLISH_BOOTSTRAP = """\
<script data-offline-language>
document.documentElement.lang = "en";
window.gradio_config.root = window.location.origin;
Object.defineProperty(window.navigator, "language", {
  configurable: true,
  get: () => "en-US"
});
Object.defineProperty(window.navigator, "languages", {
  configurable: true,
  get: () => ["en-US", "en"]
});
</script>
"""
_TEMPLATE_LOCK = threading.RLock()
_TEMPLATE_LOADER_MARKER = "_coagulation_offline_templates"
_CONFIG_PATCH_LOCK = threading.RLock()
_CONFIG_PATCH_MARKER = "_coagulation_offline_app_ids"
_EMBEDDED_CSS = """:root {
  --navy: #182a45;
  --blue: #3f78b5;
  --red: #a42e3d;
  --surface: #ffffff;
  --background: #f5f7fa;
}

body {
  background: var(--background);
}

.gradio-container {
  max-width: 1440px !important;
  margin: 0 auto;
  padding: 24px !important;
  color: var(--navy);
  background: var(--background);
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
  color: var(--navy);
  letter-spacing: -0.015em;
}

#application-title h1 {
  color: var(--navy) !important;
}

.gradio-container .block,
.gradio-container .form {
  border-color: rgba(24, 42, 69, 0.16);
  border-radius: 10px;
  background: var(--surface);
}

.gradio-container .gap {
  gap: 18px;
}

.gradio-container button.primary {
  background: var(--red) !important;
  border-color: var(--red) !important;
  color: var(--surface) !important;
  font-weight: 650;
}

.gradio-container button.primary:hover {
  background: var(--navy) !important;
  border-color: var(--navy) !important;
}

body .gradio-container div.status-field.block {
  border-left: 4px solid var(--blue) !important;
}

.gradio-container .status-field textarea,
.gradio-container .status-field input {
  color: var(--navy);
  font-weight: 600;
}
"""


def _offline_example_inputs():
    return None


def _load_local_styles() -> str:
    candidates = [Path(__file__).with_name("web_styles.css")]
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root is not None:
        packaged_path = Path(frozen_root) / "web_styles.css"
        if packaged_path not in candidates:
            candidates.append(packaged_path)

    for path in candidates:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
    return _EMBEDDED_CSS


def _strip_external_head_tags(source: str) -> str:
    return _EXTERNAL_HEAD_TAG.sub("", source)


def _offline_config_json(value):
    config = dict(value)
    config["root"] = ""
    return gradio_routes.toorjson(config)


def _offline_api_json(value):
    def without_code_snippets(item):
        if isinstance(item, dict):
            return {
                key: without_code_snippets(child)
                for key, child in item.items()
                if key != "code_snippets"
            }
        if isinstance(item, list):
            return [without_code_snippets(child) for child in item]
        return item

    return gradio_routes.toorjson(without_code_snippets(value))


def _register_offline_config(app_id: int) -> None:
    """Keep Gradio 6.16's live config local without embedding its origin."""
    with _CONFIG_PATCH_LOCK:
        update_root = gradio_route_utils.update_root_in_config
        offline_app_ids = getattr(update_root, _CONFIG_PATCH_MARKER, None)
        if offline_app_ids is None:
            offline_app_ids = set()
            original_update_root = update_root

            def update_offline_root(config, root):
                updated = original_update_root(config, root)
                if updated.get("app_id") not in offline_app_ids:
                    return updated

                def make_local(item):
                    if isinstance(item, dict):
                        return {key: make_local(value) for key, value in item.items()}
                    if isinstance(item, list):
                        return [make_local(value) for value in item]
                    if isinstance(item, str) and root and item.startswith(root):
                        return item[len(root) :]
                    return item

                local_config = make_local(updated)
                local_config["root"] = ""
                return local_config

            setattr(update_offline_root, _CONFIG_PATCH_MARKER, offline_app_ids)
            gradio_route_utils.update_root_in_config = update_offline_root

        offline_app_ids.add(app_id)


def _insert_before_module(source: str, content: str) -> str:
    module = re.search(
        r"""(?is)<script\b(?=[^>]*\btype=["']module["'])""",
        source,
    )
    if module is None:
        raise RuntimeError("The packaged Gradio template has no module entry point.")
    return source[: module.start()] + content + source[module.start() :]


def _install_offline_templates() -> None:
    environment = gradio_routes.templates.env
    with _TEMPLATE_LOCK:
        current_loader = environment.loader
        if getattr(current_loader, _TEMPLATE_LOADER_MARKER, False):
            return
        if current_loader is None:
            raise RuntimeError("The packaged Gradio template loader is unavailable.")

        packaged = {
            name: current_loader.get_source(environment, name)[0]
            for name in _TEMPLATE_NAMES
        }
        local_assets = "\n".join(_LOCAL_ASSET_TAG.findall(packaged[_TEMPLATE_NAMES[0]]))
        if not local_assets:
            raise RuntimeError("The packaged Gradio local assets are unavailable.")

        overrides: dict[str, str] = {}
        for name, source in packaged.items():
            offline_source = _strip_external_head_tags(source)
            offline_source = offline_source.replace(
                "{{ config | toorjson }}",
                "{{ config | toofflinejson }}",
            )
            offline_source = offline_source.replace(
                "{{ gradio_api_info | toorjson }}",
                "{{ gradio_api_info | toofflineapi }}",
            )
            if not _LOCAL_ASSET_TAG.search(offline_source):
                offline_source = offline_source.replace(
                    "</head>",
                    f"{local_assets}\n</head>",
                    1,
                )
            offline_source = _insert_before_module(
                offline_source,
                _ENGLISH_BOOTSTRAP,
            )
            overrides[name] = offline_source

        environment.filters["toofflinejson"] = _offline_config_json
        environment.filters["toofflineapi"] = _offline_api_json
        offline_loader = ChoiceLoader([DictLoader(overrides), current_loader])
        setattr(offline_loader, _TEMPLATE_LOADER_MARKER, True)
        environment.loader = offline_loader
        environment.cache.clear()


def _single_values(path, inset, threshold, root):
    response = run_single_analysis(path, inset, threshold, root)
    return (
        response.crops,
        response.overlay_path,
        response.heatmap_path,
        response.rows,
        response.csv_path,
        response.zip_path,
        response.output_dir,
        response.status,
    )


def _batch_values(paths, inset, threshold, root):
    response = run_batch_analysis(paths or [], inset, threshold, root)
    return (
        response.rows,
        response.summary_csv,
        response.failures_csv,
        response.zip_path,
        response.batch_dir,
        response.status,
    )


def _build_single_tab(root: Path) -> None:
    with gr.Row():
        with gr.Column(scale=2):
            source = gr.File(
                label="Complete 3×3 fixture image",
                type="filepath",
            )
            inset = gr.Slider(
                0,
                15,
                value=5,
                step=0.5,
                label="Inner crop inset",
            )
            threshold = gr.Slider(
                0,
                255,
                value=60,
                step=1,
                label="No-clot threshold",
            )
            analyze = gr.Button("Analyze Image", variant="primary")
            status = gr.Textbox(
                label="Status",
                interactive=False,
                elem_classes="status-field",
            )
        with gr.Column(scale=3):
            crops = gr.Gallery(
                label="Final inner crops",
                columns=3,
                rows=3,
            )
            with gr.Row():
                overlay = gr.Image(
                    label="Detected and final boundaries",
                    type="filepath",
                )
                heatmap = gr.Image(
                    label="Publication heatmap",
                    type="filepath",
                )
    table = gr.Dataframe(
        headers=["Cell", "Row", "Column", "Mean", "Confidence", "Recovered"],
        interactive=False,
        label="Per-cell results",
    )
    with gr.Row():
        csv_file = gr.File(label="Download CSV", interactive=False)
        zip_file = gr.File(label="Download result ZIP", interactive=False)
    result_dir = gr.Textbox(label="Saved result folder", interactive=False)
    open_folder = gr.Button("Open result folder")

    analyze.click(
        lambda path, value, cutoff: _single_values(path, value, cutoff, root),
        [source, inset, threshold],
        [crops, overlay, heatmap, table, csv_file, zip_file, result_dir, status],
    )
    open_folder.click(
        lambda path: open_result_folder(path, root),
        result_dir,
        status,
    )



def _build_batch_tab(root: Path) -> None:
    sources = gr.File(
        label="Complete 3×3 fixture images",
        file_count="multiple",
        type="filepath",
    )
    with gr.Row():
        inset = gr.Slider(
            0,
            15,
            value=5,
            step=0.5,
            label="Inner crop inset",
        )
        threshold = gr.Slider(
            0,
            255,
            value=60,
            step=1,
            label="No-clot threshold",
        )
    analyze = gr.Button("Analyze Batch", variant="primary")
    status = gr.Textbox(
        label="Batch status",
        interactive=False,
        elem_classes="status-field",
    )
    table = gr.Dataframe(
        headers=["Image", "Cells", "Status", "Reason", "Result"],
        interactive=False,
        label="Batch results",
    )
    with gr.Row():
        summary = gr.File(label="Batch summary CSV", interactive=False)
        failures = gr.File(label="Failure report CSV", interactive=False)
        archive = gr.File(label="Download batch ZIP", interactive=False)
    batch_dir = gr.Textbox(label="Saved batch folder", interactive=False)
    open_folder = gr.Button("Open batch folder")

    analyze.click(
        lambda paths, value, cutoff: _batch_values(paths, value, cutoff, root),
        [sources, inset, threshold],
        [table, summary, failures, archive, batch_dir, status],
    )
    open_folder.click(
        lambda path: open_result_folder(path, root),
        batch_dir,
        status,
    )


def create_app(results_root: str | Path) -> gr.Blocks:
    """Build the offline analysis application without starting its server."""
    _install_offline_templates()
    root = Path(results_root)
    css = _load_local_styles()
    with gr.Blocks(
        title="Coagulation Analysis",
        analytics_enabled=False,
    ) as application:
        gr.Markdown(
            "# Coagulation Analysis\nLocal and offline",
            elem_id="application-title",
        )
        with gr.Tabs():
            with gr.Tab("Single Image"):
                _build_single_tab(root)
            with gr.Tab("Batch Processing"):
                _build_batch_tab(root)
    # Gradio 6 applies CSS at launch. Retaining it on both attributes keeps the
    # factory inspectable and lets any caller launch the returned Blocks object
    # without separately knowing the stylesheet path.
    application.css = css
    application._deprecated_css = css
    application.css_paths = []
    application.head_paths = []
    application.theme = _LOCAL_THEME
    application._deprecated_theme = _LOCAL_THEME
    application._set_html_css_theme_variables()
    application.footer_links = []
    for block in application.blocks.values():
        if hasattr(block, "example_inputs"):
            block.example_inputs = _offline_example_inputs
    application.config = application.get_config_file()
    _register_offline_config(application.config["app_id"])
    return application


def main() -> None:
    """Launch the local application when this module is run as a script."""
    create_app(Path.cwd() / "results").launch(
        server_name="127.0.0.1",
        share=False,
        inbrowser=True,
        footer_links=[],
    )


if __name__ == "__main__":
    main()
