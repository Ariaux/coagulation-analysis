"""Browser-facing adapters for the offline analysis services."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

from analysis_service import AnalysisSettings, analyze_image
from batch_service import analyze_batch


@dataclass
class SingleWebResponse:
    ok: bool
    status: str
    crops: list[str] = field(default_factory=list)
    overlay_path: str | None = None
    heatmap_path: str | None = None
    rows: list[list[Any]] = field(default_factory=list)
    csv_path: str | None = None
    zip_path: str | None = None
    output_dir: str | None = None


def _table_rows(cells: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            cell["idx"],
            cell["row"],
            cell["col"],
            cell["mean"],
            cell["confidence"],
            cell["recovered"],
        ]
        for cell in cells
    ]


def _number(value: object, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exception:
        raise ValueError(f"{label} must be a number.") from exception


def _results_path(results_root: str | Path) -> Path:
    try:
        return Path(results_root)
    except (TypeError, ValueError) as exception:
        raise ValueError("Results folder is unavailable.") from exception


def run_single_analysis(
    path: str | Path,
    inset: float | str,
    threshold: float | str,
    results_root: str | Path,
) -> SingleWebResponse:
    try:
        if path is None or (isinstance(path, str) and not path.strip()):
            raise ValueError("Select an image to analyze.")
        result = analyze_image(
            Path(path),
            AnalysisSettings(
                _number(inset, "Inner crop inset"),
                _number(threshold, "No-clot threshold"),
                _results_path(results_root),
            ),
        )
    except (TypeError, ValueError, OSError, RuntimeError) as exception:
        return SingleWebResponse(ok=False, status=str(exception))
    return SingleWebResponse(
        ok=True,
        status="9 cells detected",
        crops=[str(Path(cell["crop_path"]).resolve()) for cell in result["cells"]],
        overlay_path=result["overlay_path"],
        heatmap_path=result["heatmap_path"],
        rows=_table_rows(result["cells"]),
        csv_path=result["csv_path"],
        zip_path=result["zip_path"],
        output_dir=result["output_dir"],
    )


@dataclass
class BatchWebResponse:
    ok: bool
    status: str
    success_count: int = 0
    failure_count: int = 0
    rows: list[list[Any]] = field(default_factory=list)
    summary_csv: str | None = None
    failures_csv: str | None = None
    zip_path: str | None = None
    batch_dir: str | None = None


def run_batch_analysis(
    paths: Iterable[str | Path],
    inset: float | str,
    threshold: float | str,
    results_root: str | Path,
) -> BatchWebResponse:
    try:
        if paths is None:
            raise ValueError("Select at least one image for batch processing.")
        result = analyze_batch(
            paths,
            AnalysisSettings(
                _number(inset, "Inner crop inset"),
                _number(threshold, "No-clot threshold"),
                _results_path(results_root),
            ),
        )
    except (TypeError, ValueError, OSError, RuntimeError) as exception:
        return BatchWebResponse(ok=False, status=str(exception))
    rows = [
        [item["image"], 9, "Success", "", item["output_dir"]]
        for item in result["successes"]
    ] + [
        [item["image"], "", "Failed", item["reason"], ""]
        for item in result["failures"]
    ]
    return BatchWebResponse(
        ok=True,
        status=(
            f"{result['success_count']} succeeded, "
            f"{result['failure_count']} failed"
        ),
        success_count=result["success_count"],
        failure_count=result["failure_count"],
        rows=rows,
        summary_csv=result["summary_csv"],
        failures_csv=result["failures_csv"],
        zip_path=result["zip_path"],
        batch_dir=result["batch_dir"],
    )


def open_result_folder(path: str, results_root: str | Path) -> str:
    """Open a published result directory only when it is inside the result root."""
    if path is None or results_root is None:
        return "Result folder is unavailable."
    if isinstance(path, str) and not path.strip():
        return "Result folder is unavailable."
    if isinstance(results_root, str) and not results_root.strip():
        return "Result folder is unavailable."

    try:
        root = Path(results_root).expanduser().resolve()
        candidate = Path(path).expanduser().resolve()
    except (TypeError, ValueError, OSError, RuntimeError):
        return "Result folder is unavailable."

    if not candidate.is_dir() or not candidate.is_relative_to(root):
        return "Result folder is unavailable."

    if sys.platform == "win32":
        command = ["explorer", str(candidate)]
    elif sys.platform == "darwin":
        command = ["open", str(candidate)]
    else:
        command = ["xdg-open", str(candidate)]
    try:
        subprocess.Popen(command)
    except (OSError, ValueError) as exception:
        return f"Could not open result folder: {exception}"
    return f"Opened result folder: {candidate.name}"
