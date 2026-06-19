# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""Host-machine resource sampling (CPU / RAM) for the status bar.

The sampling and formatting logic lives here, free of any Qt dependency, so it
can be exercised in isolation. The GUI layer (a status-bar label) only owns a
timer that periodically calls :func:`sample_host_metrics` and renders the result
with :func:`format_host_metrics`.

``psutil`` is imported lazily inside :func:`sample_host_metrics` so importing
this module never hard-fails if the dependency is missing, and so callers can
inject a fake implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_UNITS = ("B", "KB", "MB", "GB", "TB")


def _fmt_size(n: float) -> str:
    """Render a byte count as a compact human-readable string."""
    size = float(n)
    for unit in _UNITS:
        if size < 1024.0 or unit == _UNITS[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} {_UNITS[-1]}"  # pragma: no cover - unreachable


@dataclass(frozen=True)
class HostMetrics:
    """A point-in-time snapshot of host resource usage.

    ``cpu_percent`` and ``ram_percent`` are host-wide 0..100 figures spanning
    all cores / total physical memory. ``process_cpu_percent`` and
    ``process_rss_bytes`` describe *this* application process, so the user can
    see the tool's own footprint. ``process_cpu_percent`` is normalised to the
    whole machine (divided by the logical-CPU count), so 100 % would mean the
    app is saturating every core, keeping it comparable with ``cpu_percent``.
    """

    cpu_percent: float
    ram_percent: float
    ram_used_bytes: int
    ram_total_bytes: int
    process_cpu_percent: float
    process_rss_bytes: int


def _app_memory_bytes(proc: Any) -> int:
    """Application memory that matches what a system monitor reports.

    GNOME System Monitor (via libgtop) shows a process's memory as its
    *resident* set minus *shared* pages — i.e. the memory private to this
    process, excluding shared library/framework pages mapped into every
    process. We mirror that exact formula (``rss - shared``) so the figure
    lines up with what the user sees in the system monitor, rather than the
    raw RSS (which over-counts shared pages) or smaps-derived USS (which is
    computed differently and diverges by tens of MB).
    """
    mi = proc.memory_info()
    rss = int(mi.rss)
    shared = int(getattr(mi, "shared", 0) or 0)
    return max(rss - shared, 0)


def sample_host_metrics(*, _psutil: Any = None, _proc: Any = None) -> HostMetrics:
    """Sample current host and application CPU and memory usage.

    Both the host (``cpu_percent``) and the application (``process_cpu_percent``)
    CPU figures are measured non-blockingly (``interval=None``): they report
    utilisation since the previous call, so the very first call after process
    start returns ``0.0``. Callers should prime them once and then poll on a
    timer.

    The application :class:`psutil.Process` handle is cached at module scope so
    successive calls share the same object; ``Process.cpu_percent`` only yields
    a real figure when called repeatedly on the *same* handle (a fresh handle
    always reports 0 %). The ``_psutil`` / ``_proc`` parameters exist only for
    dependency injection.
    """
    if _psutil is None:
        import psutil  # noqa: PLC0415 - lazy so the module imports without psutil

        _psutil = psutil
    if _proc is not None:
        proc = _proc
    else:
        proc = _get_default_process(_psutil)
    vm = _psutil.virtual_memory()
    ncpu = _psutil.cpu_count() or 1
    return HostMetrics(
        cpu_percent=float(_psutil.cpu_percent(interval=None)),
        ram_percent=float(vm.percent),
        ram_used_bytes=int(vm.used),
        ram_total_bytes=int(vm.total),
        process_cpu_percent=float(proc.cpu_percent(interval=None)) / ncpu,
        process_rss_bytes=_app_memory_bytes(proc),
    )


_DEFAULT_PROCESS: Any = None


def _get_default_process(psutil_mod: Any) -> Any:
    """Return a cached :class:`psutil.Process` for this application."""
    global _DEFAULT_PROCESS
    if _DEFAULT_PROCESS is None:
        _DEFAULT_PROCESS = psutil_mod.Process()
        # Prime per-process CPU accounting so the first real sample is non-zero.
        try:
            _DEFAULT_PROCESS.cpu_percent(interval=None)
        except Exception:  # noqa: BLE001 - priming must never raise
            pass
    return _DEFAULT_PROCESS



def _app_ram_percent(m: "HostMetrics") -> float:
    """Application memory as a percentage of total physical RAM."""
    if m.ram_total_bytes <= 0:
        return 0.0
    return m.process_rss_bytes / m.ram_total_bytes * 100.0


def format_host_metrics(m: HostMetrics) -> str:
    """Render a one-line status-bar summary of host + application usage."""
    return (
        f"Host: CPU {m.cpu_percent:.2f}%  RAM {m.ram_percent:.2f}%   "
        f"App: CPU {m.process_cpu_percent:.2f}%  "
        f"RAM {_fmt_size(m.process_rss_bytes)} ({_app_ram_percent(m):.2f}%)"
    )


def format_host_metrics_tooltip(m: HostMetrics) -> str:
    """Render a multi-line tooltip with the same figures, spelled out."""
    return (
        "Resource usage\n"
        f"Host CPU (all cores): {m.cpu_percent:.2f}%\n"
        f"Host RAM: {m.ram_percent:.2f}%  "
        f"({_fmt_size(m.ram_used_bytes)} used of {_fmt_size(m.ram_total_bytes)})\n"
        f"This application CPU: {m.process_cpu_percent:.2f}% of total machine\n"
        f"This application RAM: {_fmt_size(m.process_rss_bytes)} resident "
        f"({_app_ram_percent(m):.2f}% of total RAM)"
    )
