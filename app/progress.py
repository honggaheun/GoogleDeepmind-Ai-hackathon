"""Thread-local progress message sink for streaming stage updates."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Callable

_local = threading.local()


def set_sink(fn: Callable[[str], None] | None) -> None:
    """Attach or clear a progress callback on the current thread."""
    _local.sink = fn


def get_sink() -> Callable[[str], None] | None:
    """Return the active progress callback for the current thread, if any."""
    return getattr(_local, "sink", None)


@contextmanager
def scope(fn: Callable[[str], None] | None):
    """Context manager that sets the thread-local progress sink for a block."""
    prev = get_sink()
    set_sink(fn)
    try:
        yield
    finally:
        set_sink(prev)


def note(text: str) -> None:
    """Send a progress message to the current thread's sink if registered."""
    fn = get_sink()
    if fn is not None:
        try:
            fn(text)
        except Exception:
            pass
