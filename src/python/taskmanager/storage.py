from typing import Protocol
from .models import Task


class TaskStorage(Protocol):
    def __init__(self):
        self.tasks: list[Task] = []
        pass

    def save(self, task: Task) -> None:
        pass


class MemoryStorage:
    def __init__(self):
        self.tasks: list[Task] = []

    def save(self, task: Task) -> None:
        self.tasks.append(task)
