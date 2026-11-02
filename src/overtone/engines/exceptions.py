"""Exceptions raised by engines that run the single points asynchronously."""

from pathlib import Path


class CalculationsPending(RuntimeError):
    """Some single points are still queued or running.

    Raised by engines that submit the single points to a queueing system
    instead of waiting for them. Rerun the same script once the jobs have
    finished: finished single points are not repeated, so the workflow
    continues where it stopped.
    """

    def __init__(self, directory: Path, total: int, statuses: dict[str, int]):
        self.directory = Path(directory)
        self.total = total
        self.statuses = dict(statuses)
        unfinished = ", ".join(f"{n} {status}" for status, n in self.statuses.items())
        super().__init__(
            f"{self.completed}/{total} single points in {self.directory} are finished "
            f"({unfinished}). Rerun when the jobs have finished; finished single points "
            f"are not repeated. If no jobs are queued (squeue), they may have been "
            f"stopped by the scheduler, see `hex report {self.directory}`."
        )

    @property
    def completed(self) -> int:
        return self.total - sum(self.statuses.values())


class CalculationsFailed(RuntimeError):
    """Some single points failed.

    ``failed`` maps the index of each failed structure in the trajectory to
    the error it raised.
    """

    def __init__(self, directory: Path, total: int, failed: dict[int, str]):
        self.directory = Path(directory)
        self.total = total
        self.failed = dict(sorted(failed.items()))
        indices = ", ".join(str(i) for i in self.failed)
        first_index, first_error = next(iter(self.failed.items()))
        super().__init__(
            f"{len(self.failed)}/{total} single points in {self.directory} failed "
            f"(indices {indices}). The first error (index {first_index}): {first_error}\n"
            f"See `hex report {self.directory}`. Failed single points are not rerun "
            f"unless the engine is created with retry_failed=True."
        )
