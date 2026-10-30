import unittest

from harness.prepare_mbpp import convert_record, infer_entry_point


class PrepareDataTest(unittest.TestCase):
    def test_convert_grpo_record_without_mutating_source(self):
        source = {"prompt": "Implement add.", "test_cases": "assert add(1, 2) == 3"}
        task = convert_record(source, 7)
        self.assertEqual(task["task_id"], "mbpp-0007")
        self.assertEqual(task["entry_point"], "add")
        self.assertEqual(task["tests"], source["test_cases"])
        self.assertEqual(infer_entry_point(source["test_cases"]), "add")


if __name__ == "__main__":
    unittest.main()
