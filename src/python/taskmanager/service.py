from functools import wraps

from .models import Task
from .storage import TaskStorage

class TaskNotFoundError(Exception):
    pass

class InvalidTaskStateError(Exception):
    pass

class TaskService:
    def __init__(self, storage: TaskStorage):
        self.storage = storage

    def create_task(self, task_id: int, title: str) -> Task:
        task = Task(task_id, title)
        self.storage.save(task)
        return task

    def get_task(self, task_id: int):
        for task in self.storage.tasks:
            if task.id == task_id:
                return task
        raise TaskNotFoundError(f"Task {task_id} not found")

    def complete_task(self, task_id: int):
        task = self.get_task(task_id)
        if task.completed:
            raise InvalidTaskStateError(f"Task {task_id} is already completed")
        task.complete()

    @check_completion
    def execute_task(self, task_id: int):
        task = self.get_task(task_id)
        # Here you would put the logic to execute the task
        print(f"Executing task {task_id}: {task.title}")

    def check_completion(self, execute_task):
        @wraps(execute_task)
        def wrapper(task_id: int, *args, **kwargs):
            task = self.get_task(task_id)
            if not task.completed:
                raise InvalidTaskStateError(f"Task {task_id} is not completed yet")
            return execute_task(self, task_id, *args, **kwargs)
        return wrapper