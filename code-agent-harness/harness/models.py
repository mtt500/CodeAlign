from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Task:
    """一条待解决的代码任务；通常从 JSONL 的一行反序列化而来。"""
    task_id: str
    prompt: str  # 给模型看的自然语言题目
    tests: str  # verifier 追加到候选代码后的 Python assert
    entry_point: str = "solve"  # 约定候选代码需要实现的函数名


@dataclass
class ExecutionResult:
    """一次候选代码执行的可序列化结果。"""
    passed: bool
    status: str
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0


@dataclass
class Step:
    """Agent 的一个 turn：输入 prompt、模型回复、提取代码和执行反馈。"""
    turn: int
    prompt: str
    response: str
    code: str
    execution: ExecutionResult

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class Trajectory:
    """同一任务的完整修复轨迹，可供评测或回流构造训练数据。"""
    task_id: str
    steps: list[Step] = field(default_factory=list)

    @property
    def solved(self) -> bool:
        # 只以最后一轮是否通过 verifier 定义任务成功。
        return bool(self.steps) and self.steps[-1].execution.passed

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "solved": self.solved, "steps": [step.to_dict() for step in self.steps]}

