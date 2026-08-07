"""Start the offline analysis website on the local computer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import sys
import tempfile
from collections.abc import Sequence

from app import create_app


_PORT_ERROR = "Port must be between 1 and 65535."


def default_results_root() -> Path:
    """Return the results folder beside the executable or source module."""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parent
    return base / "results"


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


def _validated_port(port: int | None) -> int:
    if port is None:
        return available_loopback_port()
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError(_PORT_ERROR)
    return port


def _prepare_results_root(results_root: str | Path) -> Path:
    root = Path(results_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    descriptor, probe_name = tempfile.mkstemp(
        prefix=".coagulation-write-probe-",
        dir=root,
    )
    try:
        os.close(descriptor)
    finally:
        Path(probe_name).unlink(missing_ok=True)
    return root


def launch_site(
    results_root: str | Path,
    port: int | None,
    open_browser: bool,
) -> None:
    """Build and start the private loopback-only Gradio application."""
    selected_port = _validated_port(port)
    root = _prepare_results_root(results_root)
    application = create_app(root)
    application.launch(
        server_name="127.0.0.1",
        server_port=selected_port,
        share=False,
        inbrowser=open_browser,
        show_error=True,
        allowed_paths=[str(root)],
        footer_links=[],
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse launcher options, start the site, and contain startup failures."""
    parser = argparse.ArgumentParser(
        description="Start the offline coagulation website."
    )
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_results_root(),
    )
    arguments = parser.parse_args(argv)

    try:
        launch_site(
            arguments.results_root,
            arguments.port,
            not arguments.no_browser,
        )
    except (OSError, RuntimeError, ValueError) as exception:
        _print_console(f"Website startup failed: {exception}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
