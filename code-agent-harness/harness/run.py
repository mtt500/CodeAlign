from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .agent import CodeRepairAgent
from .backends import MockBackend, OpenAICompatibleBackend
from .executor import PythonExecutor
from .io import append_trajectory, load_tasks, load_trajectories, write_trajectories


def format_duration(seconds: float) -> str:
    """将耗时显示为便于终端阅读的 mm:ss 或 hh:mm:ss。"""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def print_progress(done: int, total: int, started_at: float, task_id: str) -> None:
    """无第三方依赖的任务级进度条，并根据已完成任务估算剩余时间。"""
    elapsed = time.monotonic() - started_at
    ratio = done / total if total else 1.0
    width = 24
    completed = round(width * ratio)
    bar = "#" * completed + "-" * (width - completed)
    eta = elapsed / done * (total - done) if done else 0.0
    print(
        f"[{bar}] {done}/{total} ({ratio:.1%}) | "
        f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)} | {task_id}",
        flush=True,
    )


def main() -> None:
    # run.py 是批量实验入口：读取任务、运行 Agent、写轨迹并汇总指标。
    parser = argparse.ArgumentParser(description="Run generate-verify-repair trajectories for code tasks.")
    parser.add_argument("--dataset", default="data/demo_tasks.jsonl")
    parser.add_argument("--backend", choices=("mock", "openai-compatible"), default="mock")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--trace-file", default="runs/trajectories.jsonl")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from an existing trace file; completed task_ids are skipped and new results are appended.",
    )
    args = parser.parse_args()

    # mock 用于离线自检；真实比较 SFT/GRPO/OPD checkpoint 时使用兼容服务后端。
    backend = MockBackend() if args.backend == "mock" else OpenAICompatibleBackend(args.base_url, args.model)
    agent = CodeRepairAgent(backend, PythonExecutor(args.timeout), args.max_turns)
    tasks = load_tasks(args.dataset)
    task_ids = [task.task_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Dataset contains duplicate task_id values; resume would be ambiguous.")

    # --resume 读取并保留已完成的轨迹；新题逐题追加，Ctrl+C 或连接断开后仍可续跑。
    trajectories = load_trajectories(args.trace_file) if args.resume else []
    completed_ids = {trajectory.task_id for trajectory in trajectories}
    unknown_ids = completed_ids - set(task_ids)
    if unknown_ids:
        raise ValueError("Trace contains task_ids not present in this dataset; use a matching dataset or a new trace file.")
    pending_tasks = [task for task in tasks if task.task_id not in completed_ids]

    if args.resume:
        print(f"Resume: {len(completed_ids)} completed, {len(pending_tasks)} pending.", flush=True)
    else:
        # 保留旧 write_trajectories 的覆盖语义：不加 --resume 即开始一次新实验。
        write_trajectories(args.trace_file, [])

    started_at = time.monotonic()
    print_progress(len(completed_ids), len(tasks), started_at, "starting")
    for task in pending_tasks:
        # 单题可能包含多次模型调用；先打印当前题，避免长生成时终端看起来没有响应。
        print(f"Running {len(completed_ids) + 1}/{len(tasks)}: {task.task_id}", flush=True)
        trajectory = agent.solve(task)
        append_trajectory(args.trace_file, trajectory)
        trajectories.append(trajectory)
        completed_ids.add(task.task_id)
        print_progress(len(completed_ids), len(tasks), started_at, task.task_id)

    solved = sum(item.solved for item in trajectories)
    first_pass = sum(bool(item.steps) and item.steps[0].execution.passed for item in trajectories)
    # pass@1 衡量首轮策略能力；repair_success_rate 衡量给定轮数预算下的 Agent 能力。
    summary = {
        "tasks": len(trajectories),
        "solved": solved,
        "pass_at_1": first_pass / len(trajectories) if trajectories else 0.0,
        "repair_success_rate": solved / len(trajectories) if trajectories else 0.0,
        "average_turns": sum(len(item.steps) for item in trajectories) / len(trajectories) if trajectories else 0.0,
    }
    summary_path = Path(args.trace_file).with_name("summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Traces written to {args.trace_file}")


if __name__ == "__main__":
    main()

