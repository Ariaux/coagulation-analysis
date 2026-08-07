"""Local Gradio interface for fixed nine-grid coagulation analysis."""

from __future__ import annotations

import os
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr

from web_controller import open_result_folder, run_batch_analysis, run_single_analysis


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
    root = Path(results_root)
    css = Path(__file__).with_name("web_styles.css").read_text(encoding="utf-8")
    with gr.Blocks(
        title="Coagulation Analysis",
        analytics_enabled=False,
    ) as application:
        gr.Markdown("# Coagulation Analysis\nLocal and offline")
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
    return application


def main() -> None:
    """Launch the local application when this module is run as a script."""
    create_app(Path.cwd() / "results").launch(
        server_name="127.0.0.1",
        share=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
