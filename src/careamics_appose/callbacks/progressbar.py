from typing import Any

from appose.python_worker import Task
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ProgressBar


class ApposeProgressBarCallback(ProgressBar):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.num_epochs = 0
        self.curr_epoch = 0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        super().on_fit_start(trainer, pl_module)

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx
    ):
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        self.task.update(
            f"Training Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}",
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

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule):
        super().on_fit_end(trainer, pl_module)
        self.task.update(
            "Training finished", current=trainer.max_epochs, maximum=trainer.max_epochs
        )
