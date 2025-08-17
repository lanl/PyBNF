# pybnf/dask_preload/silence_kbi.py
# Quietly exit Dask worker/nanny processes on SIGINT to avoid ugly tracebacks.

import logging
import os
import signal
import sys

# ignore Ctrl-Z
try:
    signal.signal(signal.SIGTSTP, signal.SIG_IGN)
except Exception:
    pass

_LOG = logging.getLogger("pybnf.preload")

def _quiet_sigint(signum, frame):
    try:
        _LOG.debug("Worker/Nanny received SIGINT; exiting quietly.")
    except Exception:
        pass
    # Reset to default to avoid recursion, then exit immediately with success
    try:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    except Exception:
        pass
    os._exit(0)  # bypass normal exception machinery (no traceback)

def dask_setup(worker=None):
    # Install SIGINT handler early in worker/nanny processes
    try:
        signal.signal(signal.SIGINT, _quiet_sigint)
    except Exception:
        pass

    # Belt-and-suspenders: if a KeyboardInterrupt still bubbles up, exit quietly.
    def _excepthook(exc_type, exc, tb):
        if exc_type is KeyboardInterrupt:
            os._exit(0)
        return sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook

    # Also reduce Dask logging noise from these processes
    logging.getLogger("distributed").setLevel(logging.ERROR)
    _LOG.debug("Installed quiet SIGINT handler and excepthook.")
