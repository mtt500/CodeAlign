from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod


class Backend(ABC):
    """模型调用的最小抽象，方便替换本地模型、API 模型或测试用 mock。"""
    @abstractmethod
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAICompatibleBackend(Backend):
    """Minimal Chat Completions client for local vLLM or compatible endpoints."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, temperature: float = 0.2):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.temperature = temperature

    def complete(self, prompt: str) -> str:
        # 只依赖标准库 urllib，方便直接连接 vLLM 或兼容 Chat Completions 的网关。
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        # API key 优先从环境变量读取，不应进入代码仓库或 trajectory 文件。
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise RuntimeError(f"Model endpoint unavailable: {error}") from error
        return body["choices"][0]["message"]["content"]


class MockBackend(Backend):
    """Deterministic backend for validating the harness without a model endpoint."""

    def complete(self, prompt: str) -> str:
        # mock 故意第一轮产出错误实现；收到 EXECUTION FEEDBACK 后修复。
        # 它验证 Harness 流程本身，不代表真实模型能力。
        repairing = "EXECUTION FEEDBACK" in prompt
        if "palindrome" in prompt.lower():
            code = (
                "def solve(text):\n    return text == text[::-1]"
                if repairing
                else "def solve(text):\n    return text == text"
            )
        else:
            code = "def solve(nums):\n    return sum(nums)" if repairing else "def solve(nums):\n    return sum(nums) + 1"
        return f"```python\n{code}\n```"

