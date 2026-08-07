"""Start the offline analysis website on the local computer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import os
from pathlib import Path
import socket
import stat
import sys
import tempfile
import traceback
import webbrowser

from app import create_app


_PORT_ERROR = "Port must be between 1 and 65535."
_LINKED_RESULTS_ERROR = (
    "Results folder must not be a symlink, junction, or reparse point."
)
_STARTUP_LOG_NAME = "website-startup.log"


class _LauncherArgumentError(ValueError):
    """Raised when launcher command-line arguments are invalid."""


class _LauncherArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _LauncherArgumentError(message)


def _launcher_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def default_results_root() -> Path:
    """Return the results folder beside the executable or source module."""
    return _launcher_directory() / "results"


def available_loopback_port() -> int:
    """Ask the operating system for an available IPv4 loopback port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _print_console(message: str) -> None:
    """Print a message safely even when the console cannot encode Unicode."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_message = str(message).encode(
        encoding,
        errors="backslashreplace",
    ).decode(encoding)
    print(safe_message)


def _validated_port(port: int | None) -> int | None:
    if port is None:
        return None
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError(_PORT_ERROR)
    return port


def _path_is_linked(path: Path) -> bool:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False

    if stat.S_ISLNK(status.st_mode):
        return True
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None and is_junction():
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(status, "st_file_attributes", 0) & reparse_flag)


def _reject_linked_results_leaf(path: Path) -> None:
    if _path_is_linked(path):
        raise ValueError(_LINKED_RESULTS_ERROR)


def _prepare_results_root(results_root: str | Path) -> Path:
    requested_root = Path(
        os.path.abspath(Path(results_root).expanduser())
    )
    _reject_linked_results_leaf(requested_root)
    root = requested_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    descriptor, probe_name = tempfile.mkstemp(
        prefix=".coagulation-write-probe-",
        dir=root,
    )
    try:
        os.close(descriptor)
    finally:
        Path(probe_name).unlink(missing_ok=True)

    _reject_linked_results_leaf(requested_root)
    if requested_root.resolve() != root:
        raise ValueError(_LINKED_RESULTS_ERROR)
    return root


def _audit_log_directories() -> list[Path]:
    candidates: list[Path] = []
    try:
        candidates.append(_launcher_directory() / "logs")
    except (OSError, RuntimeError, ValueError):
        pass
    for environment_name in ("LOCALAPPDATA", "APPDATA"):
        environment_path = os.environ.get(environment_name)
        if environment_path:
            candidates.append(
                Path(environment_path) / "CoagulationAnalysis" / "logs"
            )
    try:
        candidates.append(Path.home() / ".coagulation-analysis" / "logs")
    except (OSError, RuntimeError):
        pass
    try:
        candidates.append(
            Path(tempfile.gettempdir()) / "coagulation-analysis" / "logs"
        )
    except (OSError, RuntimeError):
        pass

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(candidate))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    return unique_candidates


def _outside_served_root(directory: Path, results_root: str | Path | None) -> bool:
    try:
        raw_audit_directory = Path(
            os.path.abspath(directory.expanduser())
        )
        audit_directory = raw_audit_directory.resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    if results_root is None:
        return True

    try:
        raw_served_root = Path(
            os.path.abspath(Path(results_root).expanduser())
        )
    except (OSError, RuntimeError, ValueError):
        return False
    if raw_audit_directory.is_relative_to(
        raw_served_root
    ) or audit_directory.is_relative_to(raw_served_root):
        return False

    try:
        served_root = raw_served_root.resolve()
    except (OSError, RuntimeError, ValueError):
        return True
    return not (
        raw_audit_directory.is_relative_to(served_root)
        or audit_directory.is_relative_to(served_root)
    )


def _write_startup_audit(
    exception: Exception,
    results_root: str | Path | None = None,
) -> Path | None:
    timestamp = datetime.now(timezone.utc).isoformat()
    formatted_traceback = traceback.format_exc()
    entry = (
        f"[{timestamp}] Website startup failed: {exception}\n"
        f"{formatted_traceback.rstrip()}\n\n"
    )
    for directory in _audit_log_directories():
        if not _outside_served_root(directory, results_root):
            continue
        try:
            directory.mkdir(parents=True, exist_ok=True)
            log_path = directory / _STARTUP_LOG_NAME
            if _path_is_linked(log_path) or not _outside_served_root(
                log_path,
                results_root,
            ):
                continue
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
            flags |= getattr(os, "O_NOFOLLOW", 0)
            flags |= getattr(os, "O_NOINHERIT", 0)
            descriptor = os.open(log_path, flags, 0o600)
            try:
                audit_log = os.fdopen(
                    descriptor,
                    "a",
                    encoding="utf-8",
                    newline="",
                )
            except Exception:
                os.close(descriptor)
                raise
            with audit_log:
                audit_log.write(entry)
            return log_path
        except Exception:
            continue
    return None


def launch_site(
    results_root: str | Path,
    port: int | None,
    open_browser: bool,
) -> None:
    """Build and start the private loopback-only Gradio application."""
    selected_port = _validated_port(port)
    root = _prepare_results_root(results_root)
    application = create_app(root)
    try:
        launch_result = application.launch(
            server_name="127.0.0.1",
            server_port=selected_port,
            share=False,
            inbrowser=False,
            prevent_thread_lock=True,
            show_error=True,
            allowed_paths=[str(root)],
            footer_links=[],
        )
        try:
            local_url = launch_result[1]
        except (IndexError, TypeError) as exception:
            raise RuntimeError(
                "The local website did not provide a browser address."
            ) from exception
        if not isinstance(local_url, str) or not local_url:
            raise RuntimeError(
                "The local website did not provide a browser address."
            )
        if open_browser and not webbrowser.open(local_url):
            raise RuntimeError("Could not open the default browser.")
        application.block_thread()
    except Exception:
        try:
            application.close()
        except Exception:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Parse launcher options, start the site, and contain startup failures."""
    audit_results_root: str | Path | None = None
    try:
        default_root = default_results_root()
        audit_results_root = default_root
        parser = _LauncherArgumentParser(
            description="Start the offline coagulation website."
        )
        parser.add_argument("--port", type=int)
        parser.add_argument("--no-browser", action="store_true")
        parser.add_argument(
            "--results-root",
            type=Path,
            default=default_root,
        )
        arguments = parser.parse_args(argv)
        audit_results_root = arguments.results_root
        launch_site(
            arguments.results_root,
            arguments.port,
            not arguments.no_browser,
        )
    except Exception as exception:
        try:
            _write_startup_audit(exception, audit_results_root)
        except Exception:
            pass
        try:
            _print_console(f"Website startup failed: {exception}")
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
