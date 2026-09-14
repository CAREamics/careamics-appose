from appose.python_worker import Task
from careamics.careamist import CAREamist
from careamics.lightning.callbacks import ProgressBarCallback

from .progressbar import ApposeProgressBarCallback


def update_callbacks(careamist: CAREamist, task: Task) -> None:
    """Update the callbacks of the CAREamist instance
    to use the ApposeProgressBarCallback.

    Parameters
    ----------
    careamist : CAREamist
        The CAREamist instance whose callbacks are to be updated.
    task : Task
        The Appose task to be used with the ApposeProgressBarCallback.
    """
    progress_callback = ApposeProgressBarCallback(task)
    # lightning trainer can have only one progress bar callback,
    # remove existing progress bar callbacks before adding the new one.
    callbacks = [
        cb for cb in careamist.callbacks if not isinstance(cb, ProgressBarCallback)
    ]
    callbacks.append(progress_callback)
    careamist.callbacks = callbacks
    # update the trainer's callbacks
    careamist.trainer.callbacks = [  # type: ignore
        careamist.prediction_writer,
        *callbacks,
    ]
