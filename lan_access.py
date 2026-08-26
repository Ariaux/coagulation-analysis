"""Private-LAN discovery for the packaged offline website."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import ipaddress
import socket


_PRIVATE_NETWORKS = tuple(
    ipaddress.IPv4Network(value)
    for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


@dataclass(frozen=True)
class LanAccessInfo:
    loopback_url: str
    phone_urls: tuple[str, ...]
    preferred_url: str | None


def _usable_private_ipv4(value: str) -> bool:
    try:
        address = ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError:
        return False
    return any(address in network for network in _PRIVATE_NETWORKS)


def _candidate_ipv4_addresses() -> list[str]:
    records = socket.getaddrinfo(
        socket.gethostname(),
        None,
        socket.AF_INET,
        socket.SOCK_STREAM,
    )
    return [record[4][0] for record in records]


def _preferred_route_ipv4() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            return str(probe.getsockname()[0])
    except OSError:
        return None


def discover_lan_access(
    port: int,
    *,
    candidate_provider: Callable[[], Iterable[str]] = _candidate_ipv4_addresses,
    preferred_provider: Callable[[], str | None] = _preferred_route_ipv4,
) -> LanAccessInfo:
    try:
        candidates = set(candidate_provider())
    except (OSError, ValueError, TypeError):
        candidates = set()
    try:
        preferred = preferred_provider()
    except (OSError, ValueError, TypeError):
        preferred = None
    if preferred:
        candidates.add(preferred)
    usable = sorted(value for value in candidates if _usable_private_ipv4(value))
    if preferred in usable:
        usable.remove(preferred)
        usable.insert(0, preferred)
    urls = tuple(f"http://{address}:{port}" for address in usable)
    return LanAccessInfo(
        loopback_url=f"http://127.0.0.1:{port}",
        phone_urls=urls,
        preferred_url=urls[0] if urls else None,
    )
