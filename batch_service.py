"""Resilient batch orchestration for the shared image-analysis service."""

from __future__ import annotations

import csv
import ctypes
import errno
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from analysis_service import PALETTE_VERSION, AnalysisSettings, analyze_image


Progress = Callable[[int, int, str], None]
_BATCH_PUBLICATION_LOCKS: dict[Path, threading.Lock] = {}
_BATCH_PUBLICATION_LOCKS_GUARD = threading.Lock()
_DARWIN_RENAME_EXCL = 0x00000004
_LINUX_AT_FDCWD = -100
_LINUX_RENAME_NOREPLACE = 1
_OWNED_NESTED_STAGING_COMPONENT = re.compile(
    r"(\.[/\\])\.[\w-]+_staging_[A-Za-z0-9_-]+(?:\.zip)?"
    r"(?=[/\\\s'\":),;\]]|$)"
)


def _new_batch_name() -> str:
    return (
        datetime.now().strftime("batch_%Y%m%d_%H%M%S_")
        + uuid.uuid4().hex[:8]
    )


def _batch_publication_lock(root: Path) -> threading.Lock:
    with _BATCH_PUBLICATION_LOCKS_GUARD:
        return _BATCH_PUBLICATION_LOCKS.setdefault(root, threading.Lock())


def analyze_batch(
    paths: Iterable[str | Path],
    settings: AnalysisSettings,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Analyze each input independently and publish one complete batch archive."""
    settings.validate()
    sources = [Path(path) for path in paths]
    if not sources:
        raise ValueError("Select at least one image for batch processing.")

    root = (
        Path(settings.results_root).expanduser().resolve()
        if settings.results_root is not None
        else (Path.cwd() / "results").resolve()
    )
    batch_name = _new_batch_name()
    batch_dir = root / batch_name
    root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{batch_name}_staging_", dir=root)
    )

    try:
        successes: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        total = len(sources)
        for index, source in enumerate(sources, start=1):
            if progress is not None:
                progress(index, total, source.name)
            per_image_settings = AnalysisSettings(
                inset_percent=settings.inset_percent,
                no_clot_threshold=settings.no_clot_threshold,
                results_root=staging_dir,
            )
            try:
                successes.append(analyze_image(source, per_image_settings))
            except Exception as exception:
                failures.append(
                    {
                        "image": source.name,
                        "reason": _portable_failure_reason(
                            exception,
                            staging_dir,
                        ),
                    }
                )

        staged_result = _publish_batch(
            staging_dir,
            settings,
            successes,
            failures,
        )
        return _commit_batch_without_overwrite(
            staged_result,
            staging_dir,
            root,
            batch_dir,
        )
    except Exception as original_exception:
        try:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
        except Exception as cleanup_exception:
            original_exception.add_note(
                f"Could not remove batch staging directory: {cleanup_exception}"
            )
        raise


def _portable_failure_reason(exception: Exception, staging_dir: Path) -> str:
    """Keep failure details while removing references to transient storage."""
    reason = str(exception)
    for staging_prefix in sorted(
        {str(staging_dir), staging_dir.as_posix()},
        key=len,
        reverse=True,
    ):
        reason = reason.replace(staging_prefix, ".")
    reason = reason.replace(staging_dir.name, "<batch>")
    return _OWNED_NESTED_STAGING_COMPONENT.sub(
        lambda match: f"{match.group(1)}<image>",
        reason,
    )


def _commit_batch_without_overwrite(
    staged_result: dict[str, Any],
    staging_dir: Path,
    root: Path,
    initial_batch_dir: Path,
) -> dict[str, Any]:
    """Atomically publish to a fresh name without replacing existing entries."""
    batch_dir = initial_batch_dir
    with _batch_publication_lock(root):
        while True:
            if os.path.lexists(batch_dir):
                batch_dir = root / _new_batch_name()
                continue
            published_result = _rebase_result_paths(
                staged_result,
                staging_dir,
                batch_dir,
            )
            try:
                _rename_no_replace(staging_dir, batch_dir)
            except FileExistsError:
                batch_dir = root / _new_batch_name()
                continue
            return published_result


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Rename a directory atomically, failing if the destination exists."""
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        rename_exclusive = getattr(libc, "renamex_np", None)
        if rename_exclusive is not None:
            rename_exclusive.argtypes = [
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            rename_exclusive.restype = ctypes.c_int
            if rename_exclusive(
                os.fsencode(source),
                os.fsencode(destination),
                _DARWIN_RENAME_EXCL,
            ) == 0:
                return
            error = ctypes.get_errno()
            if error not in (errno.EINVAL, errno.ENOSYS, errno.ENOTSUP):
                _raise_rename_error(error, destination)

    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        rename_at2 = getattr(libc, "renameat2", None)
        if rename_at2 is not None:
            rename_at2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            rename_at2.restype = ctypes.c_int
            if rename_at2(
                _LINUX_AT_FDCWD,
                os.fsencode(source),
                _LINUX_AT_FDCWD,
                os.fsencode(destination),
                _LINUX_RENAME_NOREPLACE,
            ) == 0:
                return
            error = ctypes.get_errno()
            if error not in (errno.EINVAL, errno.ENOSYS, errno.ENOTSUP):
                _raise_rename_error(error, destination)

    if os.path.lexists(destination):
        raise FileExistsError(
            errno.EEXIST,
            os.strerror(errno.EEXIST),
            destination,
        )
    os.rename(source, destination)


def _raise_rename_error(error: int, destination: Path) -> None:
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), destination)
    raise OSError(error, os.strerror(error), destination)


def _publish_batch(
    batch_dir: Path,
    settings: AnalysisSettings,
    successes: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    """Write batch reports and a portable ZIP containing every batch artifact."""
    summary_path, failures_path = _write_batch_reports(
        batch_dir,
        settings,
        successes,
        failures,
    )
    zip_path = _create_batch_zip(batch_dir)

    return {
        "batch_dir": str(batch_dir),
        "success_count": len(successes),
        "failure_count": len(failures),
        "summary_csv": str(summary_path),
        "failures_csv": str(failures_path),
        "zip_path": str(zip_path),
        "successes": successes,
        "failures": failures,
    }


def _write_batch_reports(
    batch_dir: Path,
    settings: AnalysisSettings,
    successes: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> tuple[Path, Path]:
    summary_path = batch_dir / "batch-summary.csv"
    failures_path = batch_dir / "failures.csv"
    metadata_path = batch_dir / "batch-metadata.json"

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "status", "cells", "result"],
        )
        writer.writeheader()
        for result in successes:
            writer.writerow(
                {
                    "image": result["image"],
                    "status": "Success",
                    "cells": len(result["cells"]),
                    "result": Path(result["output_dir"]).name,
                }
            )
        for failure in failures:
            writer.writerow(
                {
                    "image": failure["image"],
                    "status": "Failed",
                    "cells": "",
                    "result": failure["reason"],
                }
            )

    with failures_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "reason"])
        writer.writeheader()
        writer.writerows(failures)

    metadata_path.write_text(
        json.dumps(
            {
                "settings": {
                    "inset_percent": settings.inset_percent,
                    "no_clot_threshold": settings.no_clot_threshold,
                    "palette_version": PALETTE_VERSION,
                },
                "palette_version": PALETTE_VERSION,
                "success_count": len(successes),
                "failure_count": len(failures),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_path, failures_path


def _create_batch_zip(batch_dir: Path) -> Path:
    zip_path = batch_dir / "batch-results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(batch_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                archive.write(path, path.relative_to(batch_dir))
    return zip_path


def _rebase_result_paths(value: Any, staging_dir: Path, batch_dir: Path) -> Any:
    """Replace absolute staging paths in a result tree with published paths."""
    if isinstance(value, dict):
        return {
            key: _rebase_result_paths(item, staging_dir, batch_dir)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rebase_result_paths(item, staging_dir, batch_dir) for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _rebase_result_paths(item, staging_dir, batch_dir) for item in value
        )
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                relative_path = candidate.relative_to(staging_dir)
            except ValueError:
                pass
            else:
                return str(batch_dir / relative_path)
    return value
