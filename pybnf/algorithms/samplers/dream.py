"""
DREAM sampler (disabled stub).

This placeholder exists to provide a clear, user-facing error instead of a
broken or incorrect implementation. It will be replaced when a correct
implementation is ready.
"""

from ...base import BayesianAlgorithm
from ...printing import PybnfError

__all__ = ["DreamAlgorithm"]

DISABLED_MESSAGE = (
    "The DREAM sampler is not available in this version of PyBNF. "
    "It was removed because it produced incorrect posterior distributions. "
    "Please use 'am' (Adaptive Metropolis)."
)

class DreamAlgorithm(BayesianAlgorithm):
    """Disabled placeholder for the DREAM sampler."""
    def __init__(self, config):
        # Fail fast with a clear message and non-zero exit via PybnfError handling.
        raise PybnfError(
            log_message="DREAM sampler selected; aborting because DREAM is disabled / not implemented.",
            user_message=DISABLED_MESSAGE,
        )
