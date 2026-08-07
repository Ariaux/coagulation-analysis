"""Browser-facing adapters for the offline analysis services."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
from numbers import Real
import ntpath
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Iterator

from analysis_service import AnalysisSettings, analyze_image
from batch_service import analyze_batch
from grid_detector import DetectionError


LOGGER = logging.getLogger(__name__)
RESULTS_FOLDER_ERROR = (
    "Could not write results. Check that the results folder is available and writable."
)
SINGLE_UNEXPECTED_ERROR = (
    "Analysis failed unexpectedly. Check the local log and try again."
)
BATCH_UNEXPECTED_ERROR = (
    "Batch processing failed unexpectedly. Check the local log and try again."
)


class _ServicePayloadError(RuntimeError):
    """Raised when a service returns an invalid controller payload."""


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


def _payload_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _ServicePayloadError(f"{label} must be a mapping.")
    return value


def _payload_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise _ServicePayloadError(f"{label} must be a list.")
    return value


def _payload_path(
    value: object,
    label: str,
    container: Path,
    *,
    directory: bool = False,
) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise _ServicePayloadError(f"{label} must be a path.")
    path = Path(value)
    if not path.is_absolute():
        raise _ServicePayloadError(f"{label} must be absolute.")
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as exception:
        raise _ServicePayloadError(f"{label} could not be resolved.") from exception
    if not resolved.is_relative_to(container):
        raise _ServicePayloadError(f"{label} is outside its result directory.")
    exists_as_expected = resolved.is_dir() if directory else resolved.is_file()
    if not exists_as_expected:
        expected = "directory" if directory else "file"
        raise _ServicePayloadError(f"{label} is not an existing {expected}.")
    return str(resolved)


def _validated_cell(
    value: object,
    position: int,
    output_dir: Path,
) -> dict[str, Any]:
    cell = _payload_mapping(value, f"cell {position}")
    expected_row = (position - 1) // 3 + 1
    expected_col = (position - 1) % 3 + 1
    if type(cell.get("idx")) is not int or cell["idx"] != position:
        raise _ServicePayloadError("Cell indices must be row-major from 1 to 9.")
    if type(cell.get("row")) is not int or cell["row"] != expected_row:
        raise _ServicePayloadError("Cell rows must match the row-major index.")
    if type(cell.get("col")) is not int or cell["col"] != expected_col:
        raise _ServicePayloadError("Cell columns must match the row-major index.")
    for field_name in ("mean", "confidence"):
        value_to_check = cell.get(field_name)
        if isinstance(value_to_check, bool) or not isinstance(value_to_check, Real):
            raise _ServicePayloadError(f"Cell {field_name} must be numeric.")
    if type(cell.get("recovered")) is not bool:
        raise _ServicePayloadError("Cell recovered flag must be boolean.")
    validated = dict(cell)
    validated["crop_path"] = _payload_path(
        cell.get("crop_path"),
        f"cell {position} crop",
        output_dir,
    )
    return validated


def _build_single_response(
    value: object,
    results_root: Path,
) -> SingleWebResponse:
    result = _payload_mapping(value, "Single analysis result")
    root = results_root.expanduser().resolve()
    output_dir_string = _payload_path(
        result.get("output_dir"),
        "Analysis output directory",
        root,
        directory=True,
    )
    output_dir = Path(output_dir_string)
    cells = _payload_list(result.get("cells"), "Analysis cells")
    if len(cells) != 9:
        raise _ServicePayloadError("Analysis must contain exactly 9 cells.")
    validated_cells = [
        _validated_cell(cell, position, output_dir)
        for position, cell in enumerate(cells, start=1)
    ]
    return SingleWebResponse(
        ok=True,
        status="9 cells detected",
        crops=[cell["crop_path"] for cell in validated_cells],
        overlay_path=_payload_path(
            result.get("overlay_path"), "Grid overlay", output_dir
        ),
        heatmap_path=_payload_path(
            result.get("heatmap_path"), "Heatmap", output_dir
        ),
        rows=_table_rows(validated_cells),
        csv_path=_payload_path(result.get("csv_path"), "Results CSV", output_dir),
        zip_path=_payload_path(result.get("zip_path"), "Results ZIP", output_dir),
        output_dir=output_dir_string,
    )


def _single_failure(exception: Exception) -> SingleWebResponse:
    if isinstance(exception, DetectionError):
        return SingleWebResponse(ok=False, status=str(exception))
    if isinstance(exception, ValueError):
        return SingleWebResponse(ok=False, status=str(exception))
    if isinstance(exception, OSError):
        LOGGER.exception("Single analysis filesystem failure")
        return SingleWebResponse(ok=False, status=RESULTS_FOLDER_ERROR)
    LOGGER.exception("Unexpected single analysis controller failure")
    return SingleWebResponse(ok=False, status=SINGLE_UNEXPECTED_ERROR)


def run_single_analysis(
    path: str | Path,
    inset: float | str,
    threshold: float | str,
    results_root: str | Path,
) -> SingleWebResponse:
    try:
        if path is None or (isinstance(path, str) and not path.strip()):
            raise ValueError("Select an image to analyze.")
        results_path = _results_path(results_root)
        result = analyze_image(
            Path(path),
            AnalysisSettings(
                _number(inset, "Inner crop inset"),
                _number(threshold, "No-clot threshold"),
                results_path,
            ),
        )
        return _build_single_response(result, results_path)
    except Exception as exception:
        return _single_failure(exception)


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


def _build_batch_response(
    value: object,
    results_root: Path,
) -> BatchWebResponse:
    result = _payload_mapping(value, "Batch analysis result")
    root = results_root.expanduser().resolve()
    batch_dir_string = _payload_path(
        result.get("batch_dir"),
        "Batch output directory",
        root,
        directory=True,
    )
    batch_dir = Path(batch_dir_string)
    successes = _payload_list(result.get("successes"), "Batch successes")
    failures = _payload_list(result.get("failures"), "Batch failures")
    success_count = result.get("success_count")
    failure_count = result.get("failure_count")
    if type(success_count) is not int or success_count != len(successes):
        raise _ServicePayloadError("Batch success count is invalid.")
    if type(failure_count) is not int or failure_count != len(failures):
        raise _ServicePayloadError("Batch failure count is invalid.")

    success_rows: list[list[Any]] = []
    for position, item_value in enumerate(successes, start=1):
        item = _payload_mapping(item_value, f"Batch success {position}")
        image = item.get("image")
        cells = item.get("cells")
        if not isinstance(image, str) or not image:
            raise _ServicePayloadError("Batch success image name is invalid.")
        if not isinstance(cells, list) or len(cells) != 9:
            raise _ServicePayloadError("Batch success must contain 9 cells.")
        output_dir = _payload_path(
            item.get("output_dir"),
            f"Batch success {position} output directory",
            batch_dir,
            directory=True,
        )
        success_rows.append([image, 9, "Success", "", output_dir])

    failure_rows: list[list[Any]] = []
    for position, item_value in enumerate(failures, start=1):
        item = _payload_mapping(item_value, f"Batch failure {position}")
        image = item.get("image")
        reason = item.get("reason")
        if not isinstance(image, str) or not image:
            raise _ServicePayloadError("Batch failure image name is invalid.")
        if not isinstance(reason, str) or not reason:
            raise _ServicePayloadError("Batch failure reason is invalid.")
        failure_rows.append([image, "", "Failed", reason, ""])

    return BatchWebResponse(
        ok=True,
        status=f"{success_count} succeeded, {failure_count} failed",
        success_count=success_count,
        failure_count=failure_count,
        rows=success_rows + failure_rows,
        summary_csv=_payload_path(
            result.get("summary_csv"), "Batch summary CSV", batch_dir
        ),
        failures_csv=_payload_path(
            result.get("failures_csv"), "Batch failures CSV", batch_dir
        ),
        zip_path=_payload_path(result.get("zip_path"), "Batch ZIP", batch_dir),
        batch_dir=batch_dir_string,
    )


def _batch_failure(exception: Exception) -> BatchWebResponse:
    if isinstance(exception, DetectionError):
        return BatchWebResponse(ok=False, status=str(exception))
    if isinstance(exception, ValueError):
        return BatchWebResponse(ok=False, status=str(exception))
    if isinstance(exception, OSError):
        LOGGER.exception("Batch analysis filesystem failure")
        return BatchWebResponse(ok=False, status=RESULTS_FOLDER_ERROR)
    LOGGER.exception("Unexpected batch analysis controller failure")
    return BatchWebResponse(ok=False, status=BATCH_UNEXPECTED_ERROR)


def run_batch_analysis(
    paths: Iterable[str | Path] | None,
    inset: float | str,
    threshold: float | str,
    results_root: str | Path,
) -> BatchWebResponse:
    try:
        if paths is None:
            raise ValueError("Select at least one image for batch processing.")
        if isinstance(paths, (str, os.PathLike)):
            raise ValueError("Select images as a batch, not as a single path.")
        sources = list(paths)
        if not sources:
            raise ValueError("Select at least one image for batch processing.")
        if not all(isinstance(source, (str, os.PathLike)) for source in sources):
            raise ValueError("Each batch item must be an image path.")
        results_path = _results_path(results_root)
        result = analyze_batch(
            sources,
            AnalysisSettings(
                _number(inset, "Inner crop inset"),
                _number(threshold, "No-clot threshold"),
                results_path,
            ),
        )
        return _build_batch_response(result, results_path)
    except Exception as exception:
        return _batch_failure(exception)


def _path_text(value: object) -> str:
    try:
        text = os.fspath(value)
    except TypeError as exception:
        raise ValueError("Path input is invalid.") from exception
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Path input is invalid.")
    return text


def _lexically_contained(
    candidate_value: object,
    root_value: object,
    platform: str,
) -> bool:
    """Check containment without resolving or inspecting the candidate."""
    try:
        candidate_text = _path_text(candidate_value)
        root_text = _path_text(root_value)
        path_module = ntpath if platform == "win32" else os.path
        candidate = path_module.normcase(
            path_module.normpath(path_module.abspath(path_module.expanduser(candidate_text)))
        )
        root = path_module.normcase(
            path_module.normpath(path_module.abspath(path_module.expanduser(root_text)))
        )
        if platform == "win32":
            candidate_drive = path_module.normcase(path_module.splitdrive(candidate)[0])
            root_drive = path_module.normcase(path_module.splitdrive(root)[0])
            if candidate_drive != root_drive:
                return False
        return path_module.commonpath((candidate, root)) == root
    except (TypeError, ValueError, OSError):
        return False


def _resolve_root_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _resolve_candidate_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _display_folder_name(value: str | Path, platform: str) -> str:
    text = _path_text(value)
    path_module = ntpath if platform == "win32" else os.path
    return path_module.basename(path_module.normpath(text))


@contextmanager
def _open_posix_result_directory(root: Path, candidate: Path) -> Iterator[int]:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, flag) for flag in required_flags):
        raise OSError("Secure directory handles are unavailable.")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    relative = candidate.relative_to(root)
    if any(part in ("", ".", "..") for part in relative.parts):
        raise OSError("Invalid result directory path.")

    handles: list[int] = []
    try:
        current = os.open(os.fspath(root), flags)
        handles.append(current)
        if not stat.S_ISDIR(os.fstat(current).st_mode):
            raise OSError("Result root is not a directory.")
        for component in relative.parts:
            current = os.open(component, flags, dir_fd=current)
            handles.append(current)
            if not stat.S_ISDIR(os.fstat(current).st_mode):
                raise OSError("Result path is not a directory.")
        yield current
    finally:
        for handle in reversed(handles):
            try:
                os.close(handle)
            except OSError:
                pass


class _WindowsDirectoryHandleApi:
    FILE_ATTRIBUTE_DIRECTORY = 0x00000010
    FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
    FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    FILE_READ_ATTRIBUTES = 0x00000080
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_TAG_INFO_CLASS = 9

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._wintypes = wintypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class FileAttributeTagInfo(ctypes.Structure):
            _fields_ = [
                ("FileAttributes", wintypes.DWORD),
                ("ReparseTag", wintypes.DWORD),
            ]

        self._attribute_info_type = FileAttributeTagInfo
        self._kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        self._kernel32.CreateFileW.restype = wintypes.HANDLE
        self._kernel32.GetFileInformationByHandleEx.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        self._kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
        self._kernel32.GetFinalPathNameByHandleW.argtypes = [
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        self._kernel32.GetFinalPathNameByHandleW.restype = wintypes.DWORD
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL

    def open_directory(self, path: str) -> object:
        handle = self._kernel32.CreateFileW(
            path,
            self.FILE_READ_ATTRIBUTES,
            self.FILE_SHARE_READ | self.FILE_SHARE_WRITE,
            None,
            self.OPEN_EXISTING,
            self.FILE_FLAG_BACKUP_SEMANTICS | self.FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        invalid_handle = self._ctypes.c_void_p(-1).value
        if handle in (None, invalid_handle):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        return handle

    def attributes(self, handle: object) -> int:
        information = self._attribute_info_type()
        if not self._kernel32.GetFileInformationByHandleEx(
            handle,
            self.FILE_ATTRIBUTE_TAG_INFO_CLASS,
            self._ctypes.byref(information),
            self._ctypes.sizeof(information),
        ):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        return int(information.FileAttributes)

    def final_path(self, handle: object) -> str:
        capacity = 32768
        buffer = self._ctypes.create_unicode_buffer(capacity)
        length = self._kernel32.GetFinalPathNameByHandleW(
            handle,
            buffer,
            capacity,
            0,
        )
        if length == 0 or length >= capacity:
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        path = buffer.value
        if path.startswith("\\\\?\\UNC\\"):
            return "\\\\" + path[8:]
        if path.startswith("\\\\?\\"):
            return path[4:]
        return path

    def close(self, handle: object) -> None:
        self._kernel32.CloseHandle(handle)


def _windows_handle_api() -> _WindowsDirectoryHandleApi:
    return _WindowsDirectoryHandleApi()


@contextmanager
def _open_windows_result_directory(
    root: Path,
    candidate: Path,
) -> Iterator[str]:
    api = _windows_handle_api()
    relative = candidate.relative_to(root)
    paths = [root]
    current = root
    for component in relative.parts:
        if component in ("", ".", ".."):
            raise OSError("Invalid result directory path.")
        current = current / component
        paths.append(current)

    handles: list[object] = []
    root_final: str | None = None
    final_path: str | None = None
    try:
        for path in paths:
            handle = api.open_directory(str(path))
            handles.append(handle)
            attributes = api.attributes(handle)
            if not attributes & api.FILE_ATTRIBUTE_DIRECTORY:
                raise OSError("Result path is not a directory.")
            if attributes & api.FILE_ATTRIBUTE_REPARSE_POINT:
                raise OSError("Result path cannot be a reparse point.")
            final_path = api.final_path(handle)
            if root_final is None:
                root_final = final_path
            elif not _lexically_contained(final_path, root_final, "win32"):
                raise OSError("Result path is outside the result root.")
        if final_path is None:
            raise OSError("Result path is unavailable.")
        yield final_path
    finally:
        for handle in reversed(handles):
            api.close(handle)


@contextmanager
def _open_stable_result_directory(
    root: Path,
    candidate: Path,
    platform: str,
) -> Iterator[tuple[str, dict[str, Any]]]:
    if platform == "win32":
        with _open_windows_result_directory(root, candidate) as final_path:
            yield final_path, {}
        return

    with _open_posix_result_directory(root, candidate) as handle:
        handle_root = "/dev/fd" if platform == "darwin" else "/proc/self/fd"
        handle_path = f"{handle_root}/{handle}"
        if not os.path.exists(handle_path):
            raise OSError("Stable directory handle path is unavailable.")
        yield handle_path, {"pass_fds": (handle,)}


def open_result_folder(path: str, results_root: str | Path) -> str:
    """Open a published result directory only when it is inside the result root."""
    platform = sys.platform
    if not _lexically_contained(path, results_root, platform):
        return "Result folder is unavailable."
    try:
        display_name = _display_folder_name(path, platform)
        root = _resolve_root_path(results_root)
        candidate = _resolve_candidate_path(path)
    except (TypeError, ValueError, OSError, RuntimeError):
        return "Result folder is unavailable."
    if not display_name or not candidate.is_relative_to(root):
        return "Result folder is unavailable."

    try:
        with _open_stable_result_directory(
            root,
            candidate,
            platform,
        ) as (launch_path, popen_kwargs):
            if platform == "win32":
                command = ["explorer", launch_path]
            elif platform == "darwin":
                command = ["open", launch_path]
            else:
                command = ["xdg-open", launch_path]
            try:
                subprocess.Popen(command, **popen_kwargs)
            except (OSError, ValueError) as exception:
                return f"Could not open result folder: {exception}"
    except (TypeError, ValueError, OSError, RuntimeError):
        return "Result folder is unavailable."
    return f"Opened result folder: {display_name}"
