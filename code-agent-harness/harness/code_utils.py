from __future__ import annotations

import re


def extract_python_code(text: str) -> str:
    """提取第一段 Markdown Python 代码块；找不到时退化为整段回复。

    这样模型偶尔没有按格式输出时，执行器仍能给出真实报错，而不会让
    Harness 因解析失败中断。
    """
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def truncate_feedback(text: str, limit: int = 3000) -> str:
    """限制回灌给模型的报错长度，避免单个异常占满下一轮上下文。"""
    text = text.strip()
    return text[-limit:] if len(text) > limit else text

