import tqdm
from appose.python_worker import Task


class TaskTQDM(tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        self.task: Task | None = globals().get("task")
        super().__init__(*args, **kwargs)

    def update(self, *args, **kwargs):
        if super().update(*args, **kwargs) and self.task is not None:
            self.task.update(self.desc, current=self.n, maximum=self.total)

    def close(self, *args, **kwargs):
        super().close(*args, **kwargs)
        if self.task is not None:
            self.task.update(self.desc, current=self.n, maximum=self.total)


# monkey patch tqdm to use TaskTQDM
tqdm.tqdm = TaskTQDM
