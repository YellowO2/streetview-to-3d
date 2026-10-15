"""Time each stage of a run, and say which stage was skipped and why."""
import logging
import time

logger = logging.getLogger(__name__)


class Stopwatch:
    """Logs "timing: <stage> <s>s" since the last lap, and warns "<stage> skipped: <why>"."""

    def __init__(self):
        self.start = time.monotonic()

    def lap(self, stage):
        logger.info(f"timing: {stage} {time.monotonic() - self.start:.1f}s")
        self.start = time.monotonic()

    def skipped(self, stage, failure):
        """Warns why; returns None, the skipped stage's result."""
        logger.warning(f"{stage} skipped: {failure!r}")
