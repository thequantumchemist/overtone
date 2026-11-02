"""A stand-in for a queueing system, for testing ShephexEngine without Slurm.

``submit_only_executor`` returns an executor class that marks experiments as
submitted without running them, as a queue would; ``finish_submitted`` then
runs them, as the queue eventually does. Both need shephex.
"""
from pathlib import Path


def submit_only_executor():
    from shephex.executor import LocalExecutor

    class SubmitOnlyExecutor(LocalExecutor):
        def __init__(self, *args, **kwargs):  # accepts SlurmExecutor's arguments
            super().__init__()

        def _sequence_execute(self, experiments, *args, **kwargs):
            for experiment in experiments:
                experiment.update_status("submitted")
            return []

    return SubmitOnlyExecutor


def finish_submitted(directory: Path | str) -> int:
    """Run the submitted experiments in ``directory``; returns how many."""
    from shephex.executor import LocalExecutor
    from shephex.experiment.status import Submitted
    from shephex.study import Study

    experiments = Study(directory).get_experiments(status="submitted")
    LocalExecutor().execute(experiments, valid_statuses=[Submitted()])
    return len(experiments)
