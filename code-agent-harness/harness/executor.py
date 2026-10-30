from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .models import ExecutionResult


def _truncate_output(value: str | bytes | None, limit: int) -> str:
    # TimeoutExpired 在不同 Python/平台上可能返回 bytes 或 str，先统一为 str。
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return (value or "")[-limit:]


class PythonExecutor:
    """Execute generated Python in a temporary directory with a wall-clock timeout.

    This is a development verifier, not a security boundary. Run untrusted code in a
    container/VM before exposing the harness to arbitrary users.
    """

    def __init__(self, timeout_seconds: float = 3.0, max_output_chars: int = 6000):
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    def run(self, code: str, tests: str) -> ExecutionResult:
        """在临时目录运行“候选代码 + 测试断言”，返回结构化执行结果。

        输入是从模型回复中提取的代码和任务自带 tests；输出不直接抛异常，
        而是将失败/超时转换为 ExecutionResult，供下一轮 Agent 修复使用。
        """
        # 测试放在候选代码后面，因此 assert 能直接调用候选函数。
        program = f"{code}\n\n# Harness tests\n{tests}\n"
        started = time.perf_counter()
        # 使用最小白名单环境，而非继承父进程环境，避免把 API Key 等敏感变量
        # 暴露给候选代码；Windows 仍需要系统路径才能正常启动 Python。
        system_root = os.environ.get("SystemRoot", r"C:\\Windows")
        child_env = {
            "SystemRoot": system_root,
            "WINDIR": os.environ.get("WINDIR", system_root),
            "ComSpec": os.environ.get("ComSpec", str(Path(system_root) / "System32" / "cmd.exe")),
            "PATH": str(Path(system_root) / "System32"),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
        with tempfile.TemporaryDirectory(prefix="code-agent-harness-") as temp_dir:
            # 每轮执行拥有独立工作目录，执行结束后由 TemporaryDirectory 清理。
            script_path = Path(temp_dir) / "candidate.py"
            script_path.write_text(program, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", str(script_path)],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    env=child_env,
                )
            except subprocess.TimeoutExpired as error:
                # timeout 也是有效观察值，Agent 下一轮可据此修改算法或终止条件。
                return ExecutionResult(
                    passed=False,
                    status="timeout",
                    stdout=_truncate_output(error.stdout, self.max_output_chars),
                    stderr=_truncate_output(error.stderr, self.max_output_chars),
                    duration_seconds=time.perf_counter() - started,
                )
        stdout = _truncate_output(completed.stdout, self.max_output_chars)
        stderr = _truncate_output(completed.stderr, self.max_output_chars)
        # returncode 非 0 时保留 stderr；这会作为下一轮 repair prompt 的反馈。
        return ExecutionResult(
            passed=completed.returncode == 0,
            status="passed" if completed.returncode == 0 else "failed",
            stdout=stdout,
            stderr=stderr,
            duration_seconds=time.perf_counter() - started,
        )

