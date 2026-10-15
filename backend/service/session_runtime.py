"""Process-local exclusion for chat, compression and deletion of one session."""
from contextlib import contextmanager
from threading import Lock


class SessionBusyError(RuntimeError):
    pass


_guard = Lock()
_active: set[str] = set()


@contextmanager
def claim_session(session_id: str):
    with _guard:
        if session_id in _active:
            raise SessionBusyError("Session has an active operation")
        _active.add(session_id)
    try:
        yield
    finally:
        with _guard:
            _active.discard(session_id)
