from collections.abc import Callable
from typing import Any, Self

from appose.python_worker import Task
from careamics.lightning.callbacks import PredictionStoppedException
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ProgressBar
from lightning.pytorch.trainer import call


class ApposeProgressBarCallback(ProgressBar):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task

    @staticmethod
    def _check_task_cancellation(
        stage: str,
    ):
        def decorator(function: Callable):
            def wrap(
                self: Self,
                trainer: Trainer,
                pl_module: LightningModule,
                *args,
                **kwargs,
            ):
                if self.task.cancel_requested:
                    self.task.update("Cancellation requested. Stopping...")
                    # Stop the training loop
                    trainer.should_stop = True
                    trainer.limit_val_batches = 0  # skip validation
                    pl_module.teardown(stage)
                    # trainer._teardown()
                    return None
                else:
                    return function(self, trainer, pl_module, *args, **kwargs)

            return wrap

        return decorator

    @_check_task_cancellation(stage="fit")
    def on_sanity_check_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        super().on_sanity_check_start(trainer, pl_module)
        self.task.update("Data Sanity Checking...")

    @_check_task_cancellation(stage="fit")
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        super().on_fit_start(trainer, pl_module)

    @_check_task_cancellation(stage="fit")
    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ):
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        self.task.update(
            f"Training: Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}",
            current=batch_idx + 1,
            maximum=int(self.total_train_batches),
        )

    @_check_task_cancellation(stage="validate")
    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        super().on_validation_batch_start(
            trainer, pl_module, batch, batch_idx, dataloader_idx
        )
        self.task.update(
            "Validation:", current=batch_idx + 1, maximum=int(self.total_val_batches)
        )

    def on_fit_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        super().on_fit_end(trainer, pl_module)
        self.task.update(
            "Training: Done!",
            current=int(trainer.num_training_batches),
            maximum=int(trainer.num_training_batches),
        )

    @_check_task_cancellation(stage="predict")
    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        super().on_predict_batch_start(
            trainer, pl_module, batch, batch_idx, dataloader_idx
        )
        self.task.update(
            "Predicting:",
            current=batch_idx + 1,
            maximum=int(trainer.num_predict_batches[0]),
        )

    def on_predict_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        super().on_predict_end(trainer, pl_module)
        _total = int(trainer.num_predict_batches[0])
        self.task.update(
            "Predicting: Done!",
            current=_total,
            maximum=_total,
        )
