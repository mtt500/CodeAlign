"""Convert the existing GRPO/MBPP JSONL into Code Agent Harness task JSONL.

The source file is read-only: this tool always writes a separate output file.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# GRPO preprocessing already embeds the required function name in most prompts.
# Infer it again from the assert statements so Harness has an explicit entry_point.
ENTRY_POINT_PATTERN = re.compile(r"assert\s+([A-Za-z_]\w*)\s*\(")


def infer_entry_point(tests: str) -> str | None:
    match = ENTRY_POINT_PATTERN.search(tests)
    return match.group(1) if match else None


def convert_record(record: dict, index: int) -> dict | None:
    """Map GRPO's prompt/test_cases schema to Harness's prompt/tests schema."""
    prompt = record.get("prompt", "").strip()
    tests = record.get("test_cases", "").strip()
    entry_point = infer_entry_point(tests)
    if not prompt or not tests or entry_point is None:
        return None
    return {
        "task_id": f"mbpp-{index:04d}",
        "prompt": prompt,
        "tests": tests,
        "entry_point": entry_point,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GRPO MBPP data to Code Agent Harness JSONL.")
    parser.add_argument("--input", required=True, help="Path to grpo_data.jsonl; the source is never modified.")
    parser.add_argument("--output", default="data/mbpp_harness.jsonl", help="Separate Harness JSONL output path.")
    parser.add_argument("--limit", type=int, default=0, help="Optional positive limit for smoke tests; 0 keeps all valid rows.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = converted = skipped = 0
    with Path(args.input).open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for index, line in enumerate(source, start=1):
            if not line.strip():
                continue
            total += 1
            task = convert_record(json.loads(line), index)
            if task is None:
                skipped += 1
                continue
            target.write(json.dumps(task, ensure_ascii=False) + "\n")
            converted += 1
            if args.limit > 0 and converted >= args.limit:
                break
    print(f"Converted {converted} tasks from {total} rows; skipped {skipped}. Output: {output_path}")


if __name__ == "__main__":
    main()
