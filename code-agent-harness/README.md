# Code Agent Harness

一个面向代码任务的 Agent 运行时与评测框架。它把“模型生成代码”扩展为可复现的闭环：**生成 → 执行验证 → 将错误反馈给模型 → 修复 → 记录轨迹**。项目可接入本地 vLLM / OpenAI 兼容服务，也提供不依赖模型服务的 mock 演示。

## 为什么做这个项目

现有后训练项目已覆盖 SFT、DPO、GRPO、OPD 和代码执行奖励，但训练阶段的单步 reward 不等于推理阶段的 Agent 能力。本项目专注于回答：

- 执行反馈能把一次代码生成的成功率提升多少？
- 哪个后训练 checkpoint 更容易修复失败代码？
- 成功与失败轨迹能否沉淀为 SFT / DPO / GRPO 的训练数据？

这不是一个泛化的网页 Agent，而是围绕代码生成和可执行 verifier 的最小可验证 Harness。

## 整个项目完成什么工作

把一个“只会生成一次代码”的模型包装为可评测的代码 Agent：模型给出候选代码后，Harness 执行测试；若测试失败，就把候选代码和报错作为 observation 交给模型修复，直到通过或达到最大轮数。每一步都会保存下来，因此既可分析 Agent 的实际行为，也可把轨迹回流为后训练数据。

```text
JSONL 任务（题目 + assert）
        ↓
模型后端生成代码（mock / 本地 vLLM / API 网关）
        ↓
提取 Python 代码块 → 临时目录执行测试
        ↓
通过：结束并记录成功
失败：stdout/stderr → 下一轮 repair prompt
        ↓
JSONL trajectory + 汇总指标 + 可选 DPO 偏好对
```

### 输入是什么

批量运行时输入为 JSONL 任务集。每行至少包含 `task_id`、`prompt` 和 `tests`：

```json
{"task_id":"sum_list","prompt":"Write solve(nums) to return the sum of a list of integers.","tests":"assert solve([1, 2, 3]) == 6\nassert solve([]) == 0","entry_point":"solve"}
```

- `prompt`：发送给模型的题目；
- `tests`：追加到生成代码末尾执行的 Python 断言；
- `entry_point`：文档化要求模型实现的函数名，目前由 prompt 约束，不做 AST 强校验。

### 输出是什么

以 `--trace-file runs/trajectories.jsonl` 为例，输出包括：

- `runs/trajectories.jsonl`：每题一行，保存所有 turn 的 prompt、原始回复、提取代码、执行状态、stdout、stderr、耗时；
- `runs/summary.json`：`pass_at_1`、最终修复成功率和平均轮数；
- 可选的 `runs/repair_preferences.jsonl`：从失败→成功轨迹抽取的 DPO 风格 `prompt/chosen/rejected` 样本。

一个成功修复轨迹会类似于：

```text
Turn 1: 模型生成 return sum(nums) + 1
Verifier: assert 失败
Turn 2: prompt 附带断言失败信息，模型改为 return sum(nums)
Verifier: 通过；trajectory.solved = true
```

## 推荐阅读顺序

先读核心链路，再看命令行和数据回流。每个模块都已加入中文注释。

| 顺序 | 文件 | 阅读时要回答的问题 |
|---:|---|---|
| 1 | `harness/models.py` | 一条任务、一轮执行、完整轨迹分别长什么样？ |
| 2 | `harness/code_utils.py` | 怎样从模型自然语言回复中得到可执行代码？ |
| 3 | `harness/executor.py` | 候选代码如何与测试拼接、如何超时、失败后返回什么？ |
| 4 | `harness/backends.py` | 如何替换 mock、本地 vLLM 或 OpenAI 兼容服务？ |
| 5 | `harness/agent.py` | 首轮和修复轮的 prompt 如何不同？状态怎样在各轮流转？ |
| 6 | `harness/io.py` | JSONL 任务和 trajectory 如何读写？ |
| 7 | `harness/run.py` | 如何对一个数据集批量运行并计算 pass@1、修复成功率？ |
| 8 | `harness/export_preferences.py` | 如何将轨迹变为 DPO 的 chosen/rejected 对？ |
| 9 | `tests/test_harness.py` | 项目最小闭环如何被自动验证？ |

建议先执行一次 mock，再打开 `runs/demo_trajectories.jsonl` 对照 `agent.py` 阅读。这样能直接看到每一轮的输入、模型回复和执行反馈。

## 功能

- **Agent loop**：失败时自动携带候选代码、stdout/stderr 和失败状态发起修复，默认最多 3 轮。
- **Python verifier**：提取代码块，在临时目录执行断言，记录通过、失败、超时、标准输出和错误输出。
- **模型后端**：内置 mock 后端；支持 OpenAI Chat Completions 兼容端点，可接 vLLM 或网关服务。
- **轨迹与指标**：以 JSONL 保存完整 turn trace，并输出 pass@1、最终成功率和平均轮数。
- **数据回流**：将“失败 → 修复成功”的轨迹导出为 DPO 风格 chosen/rejected 对。

## 项目结构

```text
code-agent-harness/
├── harness/
│   ├── agent.py               # 多轮状态机与反馈 prompt
│   ├── backends.py            # mock / OpenAI-compatible 后端
│   ├── executor.py            # 临时代码执行器与超时控制
│   ├── prepare_mbpp.py        # GRPO/MBPP 数据转 Harness 格式（不修改源文件）
│   ├── prepare_humaneval.py   # HumanEval 基础测试转 Harness 格式
│   ├── run.py                 # 批量运行、指标与轨迹保存
│   └── export_preferences.py  # 轨迹转 DPO 偏好对
├── data/demo_tasks.jsonl      # 无模型依赖的演示任务
└── tests/test_harness.py
```

## 运行方式

要求：Python 3.10+。核心 Harness 仅使用标准库。

### 1. 运行无需模型服务的演示

```powershell
cd D:\Desktop\rw\sx\project\code-agent-harness
python -m harness.run --backend mock --max-turns 3 --trace-file runs\demo_trajectories.jsonl
python -m unittest tests.test_harness
```

启动本地模型作为 api（以 sft 为例）  
看见 `INFO:     Application startup complete.` 即为启动成功
```powershell
CUDA_VISIBLE_DEVICES=1 vllm serve \
  /data/kcl/myt/guojiaxun_workspace/med/outputs/code/outputs-code-sft-merged \
  --served-model-name qwen25-sft \
  --dtype auto \
  --max-model-len 3072 \
  --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 \
  --port 8000
```


运行时会按题目输出进度条，例如 `[#-----------------------] 1/20 (5.0%) | elapsed 00:18 | ETA 05:42 | mbpp-0001`。这是任务级 ETA：单题时长会受模型生成长度、失败后的修复轮数和代码超时影响，因此只能作为估算。

### 1.0 断点续跑

轨迹在**每完成一题后立即追加写入** `--trace-file`。运行被 `Ctrl+C`、SSH 连接中断或服务异常打断后，使用相同数据集和相同 trace 文件加 `--resume` 即可跳过已完成的 `task_id`：

```bash
python -m harness.run \
  --backend openai-compatible \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen25-grpo \
  --dataset data/mbpp_smoke.jsonl \
  --max-turns 3 \
  --timeout 3 \
  --trace-file runs/grpo_mbpp_smoke.jsonl \
  --resume
```

不加 `--resume` 时仍保持原行为：覆盖该 trace 文件并开始新实验。续跑会检查任务集中的 `task_id` 是否唯一，且 trace 与当前数据集是否一致，避免误把另一组实验的结果混入汇总。

mock 后端故意在第一轮返回错误代码、第二轮依据执行反馈修复，因此可检查 trace、指标和偏好对导出是否正常：

```powershell
python -m harness.export_preferences --traces runs\demo_trajectories.jsonl --output runs\repair_preferences.jsonl
```

预期 mock 演示中两道题均首轮失败、第二轮修复成功：`pass_at_1=0.0`、最终 `repair_success_rate=1.0`、平均轮数为 `2.0`。真实模型不应以这个 mock 指标作为能力结果。

### 1.1 使用现有 GRPO/MBPP 数据做开发测试

你的 `grpo_data.jsonl` 已包含题目和可执行断言，与 Harness 的输入几乎一致；转换脚本只做字段适配，**绝不修改原始 GRPO 文件**：

```bash
python -m harness.prepare_mbpp \
  --input /data/kcl/myt/guojiaxun_workspace/med/data/code/grpo/grpo_data.jsonl \
  --output data/mbpp_harness.jsonl

# 先用 20 题确认模型、执行器和修复逻辑
python -m harness.prepare_mbpp \
  --input /data/kcl/myt/guojiaxun_workspace/med/data/code/grpo/grpo_data.jsonl \
  --output data/mbpp_smoke.jsonl \
  --limit 20
```

随后运行：

```bash
python -m harness.run \
  --backend openai-compatible \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen25-sft \
  --dataset data/mbpp_smoke.jsonl \
  --max-turns 3 \
  --timeout 3 \
  --trace-file runs/grpo_mbpp_smoke.jsonl \
  --summary-file runs/grpo_summary.json
```

注意：GRPO 已使用 MBPP 训练，完整 MBPP 只能用于开发、验证 Agent 流程和分析失败轨迹，不能作为 GRPO 的最终泛化指标。

### 1.2 使用 HumanEval 做独立 Agent 评测

HumanEval 的 `test` 字段通常只定义 `check(candidate)`，没有调用它；`prepare_humaneval.py` 在转换后追加 `check(该题入口函数)`，因此不是“修改评测”，而是让测试真正作用于模型生成的函数：

```bash
python -m harness.prepare_humaneval \
  --input /data/kcl/myt/guojiaxun_workspace/med/test_jpg/data/HumanEval.jsonl.gz \
  --output data/humaneval_harness.jsonl

python -m harness.run \
  --backend openai-compatible \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen25-dpo \
  --dataset data/humaneval_harness.jsonl \
  --max-turns 3 \
  --timeout 3 \
  --trace-file runs/dpo_humaneval_base.jsonl \
  --summary-file runs/dpo_humaneval_base_summary.json \
  --resume
```

上述脚本运行的是 **HumanEval 基础测试**。HumanEval+ 的额外测试应继续通过 EvalPlus 执行，或另行接入 EvalPlus verifier；不要把仅含基础 `test` 的转换结果称为 HumanEval+。

### 2. 接入本地 vLLM / OpenAI 兼容服务

先启动模型服务；例如端点为 `http://localhost:8000/v1`。如服务需要认证，在环境变量中设置 `OPENAI_API_KEY`，不要把密钥写入代码或 trace。

```powershell
$env:OPENAI_API_KEY = "your-key"
python -m harness.run `
  --backend openai-compatible `
  --base-url http://localhost:8000/v1 `
  --model your-model-name `
  --dataset data\demo_tasks.jsonl `
  --max-turns 3 `
  --trace-file runs\model_trajectories.jsonl
```

将 `data/demo_tasks.jsonl` 换成 MBPP、HumanEval 或自己的任务集即可。每行格式为：

```json
{"task_id":"example","prompt":"Write solve(x) ...","tests":"assert solve(2) == 4","entry_point":"solve"}
```

## 输出与评测

每个 JSONL trace 记录每一轮的 prompt、模型原始回复、提取后的代码、执行结果、stdout/stderr 与耗时。`summary.json` 记录：

- `pass_at_1`：首轮直接通过的比例；
- `repair_success_rate`：经过最大轮数后成功的比例；
- `average_turns`：每题的平均生成/修复轮数。

推荐在相同任务集、相同 verifier、相同最大轮数下，比较 SFT、DPO、GRPO、OPD checkpoint 与外部模型。另可对比“完整报错”“仅失败测试”“不提供反馈”三种修复 prompt。

## 与后训练项目的衔接

1. 将既有 SFT / GRPO / OPD checkpoint 部署为 OpenAI 兼容端点；
2. 使用本项目评估它们的首轮正确率、修复率和平均轮数；
3. 从成功轨迹抽取 SFT 样本，或从失败/成功代码构造 DPO 偏好对；
4. 回流训练后在未见任务集重新评测，避免把同一任务同时用于轨迹构建和最终测试。

## 安全边界

当前 `PythonExecutor` 使用临时目录、`python -I` 和 wall-clock timeout，**不是安全沙箱**。不要用它执行来自不可信用户或公网模型的任意代码。生产化时应使用容器或微虚拟机，关闭网络、限制 CPU/内存/进程数与文件系统权限，并将执行服务置于独立主机。

## 后续优化方向

- 引入预算控制：token、调用次数、时间成本与早停策略；
- 将错误类型标准化（语法、断言、超时、运行时异常），研究针对性 repair prompt；
- 增加 trajectory 去重、质量打分和数据集版本管理，形成可审计的后训练数据管线；
- 将 `export_preferences.py` 的输出适配现有 DPO/GRPO 数据格式，并在严格隔离的测试集上验证闭环收益。
