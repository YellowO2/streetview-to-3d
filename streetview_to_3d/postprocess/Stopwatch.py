"""Time each stage of a run, and say which stage was skipped and why."""
import time


class Stopwatch:
    """Lines to log: "timing: <stage> <s>s" since the last lap, "<stage> skipped: <why>"."""

    def __init__(self, log=print):
        self.log, self.start = log, time.monotonic()

    def lap(self, stage):
        self.log(f"timing: {stage} {time.monotonic() - self.start:.1f}s")
        self.start = time.monotonic()

    def skipped(self, stage, failure):
        """Logs why; returns None, the skipped stage's result."""
        self.log(f"{stage} skipped: {failure!r}")
