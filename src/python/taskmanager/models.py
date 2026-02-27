from dataclasses import dataclass, field


@dataclass
class Task:
    id: int
    title: str
    completed: bool = False
    tags: list[str] = field(default_factory=list)

    def complete(self) -> None:
        self.completed = True
