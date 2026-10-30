# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Train R1 model with GRPO rl algo.
"""
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import re
from datasets import load_dataset
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
from peft import LoraConfig, TaskType, get_peft_model
from latex2sympy2_extended import NormalizationConfig
# from math_verify import LatexExtractionConfig, parse, verify

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class ScriptArguments:
    """本脚本自定义的 GRPO 数据与量化参数。

    GRPO 的采样数、KL 系数、最大 completion 长度、优化器等核心超参数不在这里，
    而由 TRL 的 ``GRPOConfig`` 提供并通过 ``TrlParser`` 一同解析。阅读启动脚本时，
    应重点关注 ``num_generations``（每个 prompt 生成几条回答）以及与 KL/长度相关的参数。
    """
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    # --------------------------- 题目、标准答案与 prompt ---------------------------
    dataset_name: Optional[str] = field(
        default="openai/gsm8k",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory containing training files for local datasets."}
    )
    train_samples: Optional[int] = field(default=-1, metadata={"help": "Number of samples to train on, -1 for all"})
    subset_name: Optional[str] = field(default="main",
                                       metadata={"help": "Subset name, e.g., 'default', 'main'. default is 'default'"})
    dataset_splits: Optional[str] = field(default="train", metadata={"help": "Split name"})
    preprocessing_num_workers: Optional[int] = field(default=10,
                                                     metadata={"help": "Number of workers for preprocessing"})
    # 量化影响模型显存；GRPO 训练时会在线生成多条回答，显存同时还受生成数量影响。
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})


def normalize_text(text):
    """Normalize text by removing extra whitespace, converting to lowercase."""
    if text is None:
        return ""
    # Remove extra whitespace and convert to lowercase
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text


# def extract_answer(text):
#     """提取 ``<answer>`` 标签中的最终答案，供 GSM8K 式准确率奖励使用。

#     如果模型未遵守标签格式，则退化为整个输出；格式不合规会由 ``format_reward`` 另外惩罚。
#     """
#     if text is None:
#         return ""
#     match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
#     if match:
#         return match.group(1).strip()
#     return text.strip()


# def accuracy_reward(completions, answer, **kwargs):
#     """按最终答案是否正确返回 0/1 奖励。

#     ``GRPOTrainer`` 对每个 prompt 生成多条 completion，并将数据集的 ``answer`` 列
#     传入此函数。返回列表必须与 ``completions`` 一一对应：第 i 个分数是第 i 条
#     生成结果的奖励。这里使用 ``math_verify`` 进行数学等价判断，而不是简单字符串比较。
#     """
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
#     for content, sol in zip(contents, answer):
#         if '####' in sol:
#             # GSM8K 标准答案的最终数值位于 ``####`` 后；模型答案优先取 <answer> 内容。
#             gold_parsed = parse(sol.split("####", 1)[-1].strip())
#             answer_parsed = parse(extract_answer(content))
#         else:
#             # 通用数学数据集：从标准答案和模型输出中抽取第一段可解析 LaTex 表达式。
#             gold_parsed = parse(
#                 sol,
#                 extraction_mode="first_match",
#                 extraction_config=[LatexExtractionConfig()],
#             )
#             # 对模型输出使用较严格的 LaTex 规范化，减少“文本看似相近但表达式无效”的误奖励。
#             answer_parsed = parse(
#                 content,
#                 extraction_config=[
#                     LatexExtractionConfig(
#                         normalization_config=NormalizationConfig(
#                             nits=False,
#                             malformed_operators=False,
#                             basic_latex=True,
#                             equations=True,
#                             boxed="all",
#                             units=True,
#                         ),
#                         # Ensures that boxed is tried first
#                         boxed_match_priority=0,
#                         try_extract_without_anchor=False,
#                     )
#                 ],
#                 extraction_mode="first_match",
#             )
#         try:
#             # 数学表达式等价则 reward=1.0，否则为 0.0；异常样本不给正奖励，避免中断训练。
#             reward = float(verify(answer_parsed, gold_parsed))
#         except Exception as e:
#             logger.warning(f"Error in verification: {e}")
#             reward = 0.0
#         logger.debug(f"predict_answer: {content}, \nground_truth: {sol}, \n"
#                      f"answer_parsed: {answer_parsed}, gold_parsed: {gold_parsed}, reward: {reward}\n\n")
#         rewards.append(reward)
#     logger.debug(f'accuracy rewards: {rewards}')
#     return rewards


# def format_reward(completions, **kwargs):
#     """按是否遵循 ``<think>...</think><answer>...</answer>`` 格式返回 0/1 奖励。

#     它与 ``accuracy_reward`` 共同组成总奖励：前者鼓励答对，后者鼓励把推理和最终答案
#     放在可解析的位置。两个奖励函数的合成/权重由 GRPOTrainer 与 GRPOConfig 决定。
#     """
#     pattern = r"<think>.*?</think><answer>.*?</answer>$"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, content) for content in completion_contents]

#     rewards = [1.0 if match else 0.0 for match in matches]
#     logger.debug(f'format rewards: {rewards}')
#     return rewards


# added by jpg: 2 function
import re
import subprocess

def extract_python_code(text):
    """从模型的完整回复中精准提取纯 Python 代码"""
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    pattern_fallback = r"```(.*?)```"
    match_fallback = re.search(pattern_fallback, text, re.DOTALL)
    if match_fallback:
        return match_fallback.group(1).strip()
        
    return text

def code_execution_reward(completions, test_cases, **kwargs):
    """提取生成的代码并与 test_cases 拼接，放入沙盒执行打分"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, tests in zip(contents, test_cases):
        code = extract_python_code(content)
        # 拼接生成的代码和测试用例 (MBPP 的 assert 语句)
        full_code = code + "\n\n" + tests
        
        try:
            # 开启子进程运行，限制 2 秒超时防止死循环
            result = subprocess.run(
                ["python3", "-c", full_code],
                capture_output=True,
                timeout=2.0
            )
            if result.returncode == 0:
                rewards.append(1.0)  # 代码无 bug 且通过了所有 assert
            else:
                rewards.append(0.0)  # 语法错误或 assert 失败
        except subprocess.TimeoutExpired:
            rewards.append(-0.5)     # 严重惩罚死循环超时
        except Exception as e:
            rewards.append(0.0)
            
    return rewards

def format_reward(completions, **kwargs):
    """按是否遵循 <think>...</think> 以及 Markdown 代码块格式返回奖励"""
    # 匹配 <think> 标签，并在之后匹配 ```python 代码块
    pattern = r"<think>.*?</think>.*?```python.*?```"
    completion_contents = [completion[0]["content"] for completion in completions]
    
    # re.DOTALL 允许跨行匹配
    matches = [re.search(pattern, content, re.DOTALL) for content in completion_contents]
    rewards = [1.0 if match else 0.0 for match in matches]
    
    return rewards

# 该指令必须与 ``format_reward`` 的正则保持同步：若二者不一致，模型会同时接收到互相冲突的信号。
SYSTEM_PROMPT = (
    "You are an expert Python programmer. The user will ask you to solve a programming task. "
    "You must first think about the solution step-by-step and place your reasoning process inside <think> </think> tags. "
    "After thinking, you MUST provide the final Python code enclosed within a standard Markdown code block, i.e., ```python ... ```."
)


def get_checkpoint(training_args: GRPOConfig):
    """查找输出目录中的最近断点，供 ``trainer.train`` 恢复优化器和训练进度。"""
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def find_all_linear_names(peft_model, int4=False, int8=False):
    """找出 ``target_modules=all`` 时应注入 LoRA 的线性层。

    量化模型使用 bitsandbytes 的 Linear 类型；词表输出头默认排除，避免将它纳入 LoRA。
    """
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def grpo_train(
        model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    # ========================== GRPO 阅读主线（建议按此顺序） ==========================
    # 1. 数据集 map：构造 prompt，并保留每题标准答案 answer；
    # 2. accuracy_reward / format_reward：为同一 prompt 的多条在线生成结果打分；
    # 3. GRPOConfig：控制每题生成数、长度、KL 正则和优化超参数；
    # 4. GRPOTrainer：在组内把奖励标准化为优势，并据此进行策略梯度更新。
    #
    # 设一个 prompt x 一次生成 G 条回答 y_1...y_G，聚合奖励为 r_i。GRPO 的核心是使用
    # 同组相对优势（概念上 A_i = (r_i - mean(r)) / (std(r) + ε)）替代单独训练 value model。
    # 因此高于本组平均水平的回答被强化，低于平均的被削弱；若同组所有回答奖励相同，
    # 则几乎没有区分信号。实际优势归一化、KL 和损失细节由当前安装版本的 TRL 实现及
    # GRPOConfig 参数决定。
    # ==============================================================================
    # Trainer/Accelerate 会接管 DDP、DeepSpeed 等分布式初始化；这里只借 local_rank 判断日志进程。
    is_main_process = training_args.local_rank in [-1, 0]

    # Only log on main process
    if is_main_process:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Script parameters {script_args}")
        logger.info(f"Training parameters {training_args}")

    # ------------------------- 1. tokenizer 与 prompt 格式 -------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=getattr(model_args, "trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        # 批量生成的 completion 长度不同，需要 PAD；此处用 EOS 作为 padding token。
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------- 2. 数据：题目 + 标准答案 -------------------------
    # 预期原始数据至少具有 ``question`` 和 ``answer`` 列。question 用于生成，
    # answer 不送入模型上下文，而是随 batch 传给 accuracy_reward 作外部判分依据。
    if script_args.train_file_dir and os.path.exists(script_args.train_file_dir):
        # 本地 JSON 数据路径。
        dataset = load_dataset("json", data_dir=script_args.train_file_dir, split="train")
    else:
        # Hugging Face Hub 数据路径（默认 GSM8K）。
        dataset = load_dataset(script_args.dataset_name, script_args.subset_name, split=script_args.dataset_splits)

    if script_args.train_samples > 0:
        # 调试时先固定随机种子打乱、再截取，避免只取原数据集开头的一小段。
        dataset = dataset.shuffle(seed=42).select(range(script_args.train_samples))

    # --------------------- 3. 题目 -> GRPOTrainer 输入样本 ---------------------
    with training_args.main_process_first(desc="Dataset preparation"):
        dataset = dataset.map(
            lambda x: {
                'prompt': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': x['prompt']}  # 改为读取新数据的 'prompt'
                ],
                'test_cases': x['test_cases']  # 改为保留新数据的 'test_cases'，传递给沙盒阅卷机
            },
            num_proc=script_args.preprocessing_num_workers,
            desc="Processing dataset" if is_main_process else None,
        )

    # 固定比例划分训练/验证。这里未提供 seed，因此不同运行的切分可能不同；若需严格可复现，
    # 应显式传入 seed（本次仅加注释，不改变原有行为）。
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    if is_main_process:
        logger.info("*** Initializing model kwargs ***")

    # -------------------------- 4. policy 模型加载 --------------------------
    # GRPOTrainer 在该实现中只接收一个待训练 policy 模型；参考策略/KL 相关处理由 TRL
    # 按当前 GRPOConfig 和模型/PEFT 状态在内部完成，而不像本项目 DPO 脚本显式传 ref_model。
    torch_dtype = (
        model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    )

    # DDP 时每个进程独占本地设备，不能再用 device_map="auto" 将同一模型跨卡分片。
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1

    # QLoRA 的量化权重与 ZeRO-3 参数切分在此组合下不兼容。
    if script_args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 are both currently incompatible with QLoRA.")

    # 4-bit 和 8-bit 是互斥加载方式。
    if model_args.load_in_4bit and model_args.load_in_8bit:
        raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")

    # BitsAndBytesConfig 决定基础模型以何种低比特格式加载；QLoRA 通常在该量化底座上训练 LoRA adapter。
    quantization_config = None
    if script_args.qlora and (model_args.load_in_4bit or model_args.load_in_8bit):
        if is_main_process:
            logger.info(
                f"Quantizing model, load_in_4bit: {model_args.load_in_4bit}, load_in_8bit: {model_args.load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif model_args.load_in_4bit or model_args.load_in_8bit:
        # Support quantization even without qlora flag
        if is_main_process:
            logger.info(
                f"Quantizing model, load_in_4bit: {model_args.load_in_4bit}, load_in_8bit: {model_args.load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    # 这些参数决定模型加载精度、注意力实现、CPU 内存占用和量化方式。
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=getattr(model_args, "trust_remote_code", True),
        attn_implementation=model_args.attn_implementation,
        dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        quantization_config=quantization_config,
    )

    num_gpus = torch.cuda.device_count()
    if ddp:
        # DDP 的设备放置由 Accelerate/Trainer 管理。
        model_kwargs["device_map"] = None
    elif num_gpus > 1:
        # 非 DDP 的单进程多卡情况下，让 Transformers 自动把层分布到多张 GPU；
        # 同时每张卡预留约 20% 显存给生成时的 KV cache、激活值和优化器状态。
        max_memory = {}
        for i in range(num_gpus):
            gpu_props = torch.cuda.get_device_properties(i)
            total_mem = gpu_props.total_memory
            # 预留20%内存给训练时的梯度、优化器状态等
            usable_mem = int(total_mem * 0.8)
            max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"
        model_kwargs["max_memory"] = max_memory
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"

    if is_main_process:
        logger.info(f"Using {num_gpus} GPUs")
        logger.info(f"model_kwargs={model_kwargs}")

    config = AutoModelForCausalLM.config_class if hasattr(AutoModelForCausalLM, 'config_class') else None
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=getattr(model_args, "trust_remote_code", True),
            revision=model_args.model_revision,
        )
    except Exception:
        config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    # ZeRO-3 对 MoE 模块进行特殊标记，避免专家路由相关模块被不正确地细粒度切分。
    model_type = getattr(config, "model_type", None) if config else getattr(model.config, "model_type", None)
    if model_type == "mixtral" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
        set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    if model_type == "deepseek_v3" and is_deepspeed_zero3_enabled():
        for layer in model.model.layers:
            if 'DeepseekV3MoE' in str(type(layer.mlp)):
                layer.mlp._z3_leaf = True

    if model_type == "qwen3_moe" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
        set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])

    if model_type == "qwen3_5_moe" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
        set_z3_leaf_modules(model, [Qwen3_5MoeSparseMoeBlock])

    if is_main_process and hasattr(model, 'hf_device_map'):
        logger.info(f"Model Device Map: {model.hf_device_map.items()}")
    elif is_main_process and num_gpus > 1:
        logger.info("Model Device Map:")
        for name, param in model.named_parameters():
            if hasattr(param, 'device'):
                logger.info(f"  {name}: {param.device}")
                break

    # ----------------------- 5. LoRA/QLoRA 的可训练参数 -----------------------
    if model_args.use_peft:
        # 使用 LoRA 时基础模型保持冻结，RL 梯度仅更新低秩 adapter，显著降低 GRPO 的显存成本。
        if is_main_process:
            logger.info("Fine-tuning method: LoRA(PEFT)")
        if training_args.gradient_checkpointing:
            # 此实现为 LoRA 主动关闭 gradient checkpointing；要变更该策略需同时验证
            # PEFT、生成和反向传播的兼容性。
            logger.warning("Gradient checkpointing is enabled. It may cause issues with LoRA, setting it to False.")
            training_args.gradient_checkpointing = False
        target_modules = model_args.lora_target_modules if model_args.lora_target_modules else None
        if target_modules == 'all' or (target_modules and 'all' in target_modules):
            # 自动选择各线性层作为 LoRA 目标；量化模型使用对应的 bitsandbytes 线性层类型。
            target_modules = find_all_linear_names(model, int4=model_args.load_in_4bit, int8=model_args.load_in_8bit)
        if is_main_process:
            logger.info(f"Peft target_modules: {target_modules}, lora rank: {model_args.lora_r}, ")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        # 将 adapter 参数转换到 fp32，规避量化/半精度训练时部分更新出现的类型问题。
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
    else:
        if is_main_process:
            logger.info("Fine-tuning method: Full parameters training")

    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        # 重算激活值换显存；开启时关闭 KV cache，因为 cache 与重计算训练图不兼容。
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")

    # ----------------- 6. GRPOTrainer：在线采样、打分与组相对更新 -----------------
    trainer = GRPOTrainer(
        # 每次更新中 Trainer 会针对同一 prompt 生成一组候选 completion，
        # 再对每条 completion 计算 reward 与组内相对优势。
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            code_execution_reward,  # <-- 替换为新的代码沙盒打分器
            format_reward           # <-- 替换为新的格式打分器
        ],
        args=training_args,
        # 数据集保留 prompt 与 answer；answer 会作为 reward function 的关键字参数传递。
        train_dataset=train_dataset,
        eval_dataset=test_dataset if training_args.eval_strategy != "no" else None,
    )
    logger.info("*** GRPO Trainer initialized ***")
    logger.debug(f"Trainer: {trainer}")

    # ---------------------------- 7. 执行 GRPO 训练 ----------------------------
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        if is_main_process:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if is_main_process:
        logger.info(
            f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for '
            f'{training_args.num_train_epochs} epochs ***'
        )

    # trainer.train 内部循环的核心顺序：
    # prompt -> 生成 G 个 completion -> 逐条调用 reward_funcs -> 组内计算相对优势 ->
    # 计算带 KL 约束的策略损失 -> 反向传播更新 policy（或 LoRA adapter）。
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Log and save metrics on main process
    if is_main_process:
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info("*** Training complete ***")
        logger.info("*** Save model ***")

    # 训练阶段为兼容 gradient checkpointing 可能关闭 cache；保存前恢复，便于后续生成推理。
    trainer.model.config.use_cache = True
    if is_main_process:
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    training_args.distributed_state.wait_for_everyone()

    if is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Tokenizer saved to {training_args.output_dir}")

        # Create model card and save config
        kwargs = {
            "dataset_name": script_args.dataset_name,
            "tags": ["r1", "grpo"],
        }
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if is_main_process:
        logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
