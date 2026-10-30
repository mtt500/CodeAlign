from __future__ import annotations

from .backends import Backend
from .code_utils import extract_python_code, truncate_feedback
from .executor import PythonExecutor
from .models import Step, Task, Trajectory


SYSTEM_RULES = """You are a Python coding agent. Return exactly one Python code block.
Implement the requested entry-point function. Do not repeat tests or add prose."""


def build_prompt(task: Task, previous_code: str | None = None, feedback: str | None = None) -> str:
    """构造首轮生成或后续修复 prompt。"""
    prompt = f"{SYSTEM_RULES}\n\nTASK\n{task.prompt}\n\nEntry point: {task.entry_point}\n"
    if previous_code is not None and feedback is not None:
        # 修复轮显式提供上一版代码及 verifier 反馈，形成可观察的工具调用闭环。
        prompt += (
            "\nPREVIOUS CANDIDATE\n```python\n"
            f"{previous_code}\n```\n\nEXECUTION FEEDBACK\n{truncate_feedback(feedback)}\n"
            "Repair the candidate according to the feedback."
        )
    return prompt


class CodeRepairAgent:
    def __init__(self, backend: Backend, executor: PythonExecutor, max_turns: int = 3):
        self.backend = backend
        self.executor = executor
        self.max_turns = max_turns

    def solve(self, task: Task) -> Trajectory:
        """对单题执行至多 max_turns 次“生成—验证—修复”。"""
        trajectory = Trajectory(task_id=task.task_id)
        prior_code: str | None = None
        feedback: str | None = None
        for turn in range(1, self.max_turns + 1):
            # 首轮只有任务；后续轮次同时带上上轮代码与执行反馈。
            prompt = build_prompt(task, prior_code, feedback) if prior_code else build_prompt(task)
            response = self.backend.complete(prompt)
            # 模型回复可能含解释，执行前只保留代码块。
            code = extract_python_code(response)
            execution = self.executor.run(code, task.tests)
            trajectory.steps.append(Step(turn, prompt, response, code, execution))
            if execution.passed:
                # 一旦通过所有测试，立即停止，避免无意义的额外模型调用。
                break
            prior_code = code
            # 将失败原因转为下一轮模型可读的 observation。
            feedback = f"status={execution.status}\nstdout:\n{execution.stdout}\nstderr:\n{execution.stderr}"
        return trajectory

