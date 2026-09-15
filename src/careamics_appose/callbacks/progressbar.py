from typing import Any

from appose.python_worker import Task
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ProgressBar


class ApposeProgressBarCallback(ProgressBar):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task

    def on_sanity_check_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        super().on_sanity_check_start(trainer, pl_module)
        self.task.update("Data Sanity Check...")

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        super().on_fit_start(trainer, pl_module)

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
            current=batch_idx,
            maximum=int(self.total_train_batches),
        )

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
            "Validation:", current=batch_idx, maximum=int(self.total_val_batches)
        )

    def on_fit_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        super().on_fit_end(trainer, pl_module)
        self.task.update(
            "Training: Done!", current=trainer.max_epochs, maximum=trainer.max_epochs
        )

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
            current=batch_idx,
            maximum=int(self.total_predict_batches_current_dataloader),
        )

    def on_predict_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        super().on_predict_end(trainer, pl_module)
        _total = self.total_predict_batches_current_dataloader
        self.task.update(
            "Predicting: Done!",
            current=int(_total),
            maximum=int(_total),
        )
