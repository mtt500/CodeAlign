from __future__ import annotations

import json
from pathlib import Path

from .models import Task, Trajectory


def load_tasks(path: str | Path) -> list[Task]:
    """读取 JSONL 任务集；每一行对应一个 Task。"""
    tasks: list[Task] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.append(Task(**json.loads(line)))
    return tasks


def write_trajectories(path: str | Path, trajectories: list[Trajectory]) -> None:
    """将每题独立写成一行 JSON，适合后续流式分析和数据回流。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n")


def load_trajectories(path: str | Path) -> list[Trajectory]:
    """读取既有轨迹，用于断点续跑时跳过已经完成的题目。"""
    source = Path(path)
    if not source.exists():
        return []

    trajectories: list[Trajectory] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL trace at {source}:{line_number}") from error
            # 汇总指标只需 task_id、是否通过和轮数；无需反序列化完整 prompt/code。
            steps = record.get("steps", [])
            trajectory = Trajectory(task_id=record["task_id"])
            for step in steps:
                # 保留轻量对象的目的仅是复用 Trajectory.solved 与 len(steps)。
                from .models import ExecutionResult, Step

                execution = ExecutionResult(**step["execution"])
                trajectory.steps.append(
                    Step(step["turn"], "", "", "", execution)
                )
            trajectories.append(trajectory)
    return trajectories


def append_trajectory(path: str | Path, trajectory: Trajectory) -> None:
    """完成一题立即落盘；进程中断时，已完成题目仍可在下次续跑。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()

