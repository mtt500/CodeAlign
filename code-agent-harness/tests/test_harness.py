import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from harness.agent import CodeRepairAgent
from harness.backends import MockBackend
from harness.executor import PythonExecutor
from harness.io import append_trajectory, load_trajectories
from harness.models import Task


class HarnessTest(unittest.TestCase):
    def test_mock_agent_repairs_candidate(self):
        # 验证最小闭环：首轮失败、观察反馈、第二轮修复成功。
        task = Task("sum", "Write solve(nums) to return the sum of a list of integers.", "assert solve([2, 3]) == 5")
        trajectory = CodeRepairAgent(MockBackend(), PythonExecutor(), max_turns=2).solve(task)
        self.assertTrue(trajectory.solved)
        self.assertEqual(len(trajectory.steps), 2)
        self.assertEqual(trajectory.steps[0].execution.status, "failed")

    def test_appended_trace_can_be_loaded_for_resume(self):
        task = Task("sum", "Write solve(nums) to return the sum of a list of integers.", "assert solve([2, 3]) == 5")
        trajectory = CodeRepairAgent(MockBackend(), PythonExecutor(), max_turns=2).solve(task)
        with TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.jsonl"
            append_trajectory(trace_path, trajectory)
            restored = load_trajectories(trace_path)
        self.assertEqual([item.task_id for item in restored], ["sum"])
        self.assertTrue(restored[0].solved)
        self.assertEqual(len(restored[0].steps), 2)


if __name__ == "__main__":
    unittest.main()
