from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    # 从完整轨迹中保留“同题失败回答 vs 最终成功回答”，导出偏好对。
    parser = argparse.ArgumentParser(description="Export failed-to-passed repair pairs as DPO-style examples.")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    examples = []
    for line in Path(args.traces).read_text(encoding="utf-8").splitlines():
        trace = json.loads(line)
        steps = trace["steps"]
        if len(steps) < 2 or not trace["solved"]:
            continue
        # 取第一条失败候选作为 rejected，最后的验证通过候选作为 chosen。
        failed = next((item for item in steps if not item["execution"]["passed"]), None)
        passed = steps[-1]
        if failed is None:
            continue
        examples.append(
            {
                "prompt": failed["prompt"],
                "chosen": f"```python\n{passed['code']}\n```",
                "rejected": f"```python\n{failed['code']}\n```",
                "metadata": {"task_id": trace["task_id"], "source": "repair_trajectory"},
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in examples:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Exported {len(examples)} preference pairs to {output}")


if __name__ == "__main__":
    main()

