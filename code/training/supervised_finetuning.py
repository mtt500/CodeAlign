# -*- coding: utf-8 -*-
# Copyright 2023 XuMing(xuming624@qq.com) and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, LLaMA, Bloom, ...) on a json file or a dataset.

part of code is modified from https://github.com/shibing624/textgen
"""

import math
import os
import sys
import json
from dataclasses import dataclass, field
from glob import glob
from types import MethodType
from typing import Literal, Optional, Tuple

import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    Seq2SeqTrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils.versions import require_version

from transformers.integrations import is_deepspeed_zero3_enabled
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.tool_utils import get_tool_utils, FunctionCall, load_local_json_datasets
from training.template import get_conv_template
try:
    import flash_attn  # noqa: F401

    is_flash_attn_2_available = True
except ImportError:
    is_flash_attn_2_available = False



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the model in 4bit mode or not."})
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "Auth token to log in with Hugging Face Hub."})
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "Adopt scaled rotary positional embeddings."}
    )
    flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."}
    )
    shift_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable shifted sparse attention (S^2-Attn) proposed by LongLoRA."}
    )
    neft_alpha: Optional[float] = field(
        default=0,
        metadata={"help": "The alpha parameter to control the noise magnitude in NEFTune. value can be 5."}
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The train jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.max_train_samples is not None and 0 < self.max_train_samples <= 1000:
            logger.warning("You may set max_train_samples = -1 to run all samples in production.")


@dataclass
class ScriptArguments:
    """SFT 特有的训练开关。

    阅读本脚本时应优先核对这里的参数与启动脚本是否一致：它们直接决定
    全参/LoRA、是否只学习回答、最大上下文长度，以及工具调用数据的序列化格式。
    """
    # True：仅训练 LoRA 等 PEFT 参数；False：所有基础模型参数均参与训练。
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    # False 是常规指令微调设置：用户输入只作为条件，不对其 token 计算 loss。
    train_on_inputs: bool = field(default=False, metadata={"help": "Whether to train on inputs"})
    # ``all`` 会自动查找所有线性层；也可传入逗号分隔的层名，精细控制 LoRA 的注入位置。
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None, metadata={"help": "The path to the peft model"})
    # QLoRA 依赖 4-bit/8-bit 量化加载，并在量化基础模型上训练 LoRA 适配器。
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum model context length. suggest: 8192 * 4, 8192 * 2, 8192, 4096, 2048, 1024, 512"}
    )
    template_name: Optional[str] = field(
        default=None,
        metadata={"help": "The prompt template name. If not set, use tokenizer's built-in chat_template."}
    )
    tool_format: Optional[str] = field(
        default=None,
        metadata={"help": "Tool format to use for agent training. Options: default, glm4, llama3, mistral, qwen."}
    )

    def __post_init__(self):
        if self.model_max_length < 60:
            raise ValueError("You must specify a valid model_max_length >= 60 to run training")


class SavePeftModelTrainer(Trainer):
    """为 PEFT/LoRA 覆盖保存行为的 Trainer。

    ``PeftModel.save_pretrained`` 默认只保存适配器权重和配置，而不是完整基础模型。
    因此训练后的目录需要与原始 ``model_name_or_path`` 配合，才能重新加载推理。
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """保存 LoRA adapter，并一并保存 Trainer 参数供断点续训/追溯使用。"""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


def save_model(model, tokenizer, args):
    """保存非 ZeRO-3 场景的最终模型（全参模型或 LoRA adapter）与 tokenizer。"""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # DDP/并行包装后，真实模型通常位于 ``model.module``；单卡时则直接保存 model。
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_model_zero3(model, tokenizer, args, trainer):
    """Save the model for deepspeed zero3.
    refer https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_lora.py#L209
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # ZeRO-3 会把参数切分到各个进程，不能直接 save_pretrained；先由 DeepSpeed 聚合为完整 16-bit state dict。
    state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir, state_dict=state_dict_zero3)
    tokenizer.save_pretrained(output_dir)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(peft_model, int4=False, int8=False):
    """找出可注入 LoRA 的线性层名称。

    普通模型使用 ``torch.nn.Linear``；量化模型会替换为 bitsandbytes 的线性层类型。
    输出层 ``lm_head`` / ``output_layer`` 被刻意排除，避免将词表输出头默认纳入 LoRA。
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


def check_and_optimize_memory():
    """检查并优化GPU内存使用"""
    if not torch.cuda.is_available():
        return

    logger.info("🔍 检查GPU内存状态...")

    # 清理缓存
    torch.cuda.empty_cache()

    # 检查每个GPU的内存状态
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024 ** 3
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        cached = torch.cuda.memory_reserved(i) / 1024 ** 3
        free = total_memory - allocated - cached

        logger.info(f"GPU {i} ({props.name}):")
        logger.info(f"  总内存: {total_memory:.1f}GB")
        logger.info(f"  已分配: {allocated:.1f}GB")
        logger.info(f"  已缓存: {cached:.1f}GB")
        logger.info(f"  可用: {free:.1f}GB")

    # 设置内存优化选项
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        logger.info("✅ 启用Flash Attention优化")

    # 启用内存高效的注意力机制
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logger.info("✅ 启用内存高效注意力机制")


def main():
    # ============================ 阅读主线（建议按此顺序） ============================
    # 1) 参数与 tokenizer：决定上下文格式、特殊 token 与最大长度；
    # 2) 数据集 + preprocess_function：决定 prompt/answer 如何转为 labels，即模型实际学习的目标；
    # 3) 模型加载 + 量化 + LoRA：决定可训练参数及显存占用；
    # 4) data collator + Trainer：决定 batch padding、反向传播、保存和评估。
    # ==============================================================================
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))

    # 使用 parse_args_into_dataclasses 时忽略未知参数
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果我们传递了一个 JSON 文件，让我们用它来配置参数
        model_args, data_args, training_args, script_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        # 否则解析命令行参数，忽略未知参数
        model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses(look_for_args_file=False)

    # DeepSpeed 配置由 TrainingArguments 在 Trainer 初始化阶段接管；此处清空已缓存的 plugin，
    # 防止提前创建的分布式状态使用了过期配置。
    if training_args.deepspeed is not None:
        training_args.distributed_state.deepspeed_plugin = None

    # Trainer 后续负责初始化 DDP/DeepSpeed。这里只用 rank 判断哪些进程可以输出日志或保存最终文件。
    is_main_process = training_args.local_rank in [-1, 0]

    # Only log on main process
    if is_main_process:
        logger.info(f"Model args: {model_args}")
        logger.info(f"Data args: {data_args}")
        logger.info(f"Training args: {training_args}")
        logger.info(f"Script args: {script_args}")
        logger.info(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )

    # 在 shuffle、数据切分及模型初始化之前设置种子，保证可复现实验（分布式仍需注意各进程随机状态）。
    set_seed(training_args.seed)

    # ---------------------------- 1. tokenizer 与提示词格式 ----------------------------
    # tokenizer 的特殊 token、template 与数据预处理必须匹配底座模型；不匹配会造成
    # 训练文本格式与推理格式不同，即使 loss 正常下降也可能无法正确对话。
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    prompt_template = None
    if script_args.template_name:
        # 显式模板优先；未指定时会在 preprocess_function 中使用 tokenizer.chat_template。
        prompt_template = get_conv_template(script_args.template_name)
    if tokenizer.eos_token_id is None:
        # SFT 会为每个回答追加 EOS，因此没有 EOS 的 tokenizer 必须先补齐该定义。
        if prompt_template:
            tokenizer.eos_token = prompt_template.stop_str
        else:
            tokenizer.eos_token = "</s>"
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"Add eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    if tokenizer.bos_token_id is None:
        # 某些模型无独立 BOS 时复用 EOS，保证首轮编码可使用统一的 special token 逻辑。
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"Add bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    if tokenizer.pad_token_id is None:
        # batch 内长度不同需要 padding；优先复用 unk，缺失时退化为 eos。
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    logger.debug(f"Tokenizer: {tokenizer}")

    # labels 中等于 IGNORE_INDEX 的位置会被交叉熵忽略。它既用于屏蔽 prompt，
    # 也会被 data collator 用于补齐 label；该取值必须和后续 collator 保持一致。
    IGNORE_INDEX = LabelSmoother.ignore_index if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    # ----------------------------- 2. 原始数据集与切分 ------------------------------
    # 两种输入来源都应最终包含 ``train``；若未提供 validation，则固定随机种子后从 train 切出验证集。
    if data_args.dataset_name is not None:
        # Hugging Face Hub 数据集路径。
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            # 先 shuffle 再切分，避免按原文件顺序切分引入分布偏差；seed=42 使切分稳定。
            shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)
            # Split the shuffled train dataset into training and validation sets
            split = shuffled_train_dataset.train_test_split(
                test_size=data_args.validation_split_percentage / 100,
                seed=42
            )
            # Assign the split datasets back to raw_datasets
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]
    else:
        # 本地路径会递归收集所有 .jsonl 文件；训练/验证文件可以分别存放在不同目录。
        data_files = {}
        if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
            train_data_files = glob(f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {train_data_files}")
            data_files["train"] = train_data_files
        if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
            eval_data_files = glob(f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {eval_data_files}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_local_json_datasets(data_files, cache_dir=model_args.cache_dir)
        # 未给出验证文件时，从本地训练集按配置比例切分，逻辑与 Hub 数据集路径保持一致。
        if "validation" not in raw_datasets.keys():
            shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)
            split = shuffled_train_dataset.train_test_split(
                test_size=float(data_args.validation_split_percentage / 100),
                seed=42
            )
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]
    logger.info(f"Raw datasets: {raw_datasets}")

    # ``max_length`` 是单条多轮样本最终 token 序列的上限；它同时影响显存、吞吐和长上下文保留程度。
    # -------------------------- 3. 核心：对话转监督训练样本 ---------------------------
    max_length = script_args.model_max_length

    def preprocess_function(examples):
        """将一批原始对话转换为监督微调所需的模型输入与训练标签。

        处理流程：
        1. 将数据集中的 ``conversations`` 规范化为“用户问题 / 助手回答”轮次；
        2. 根据提示词模板（或 tokenizer 自带的 chat template）拼接每个轮次；
        3. 对文本进行分词、按比例截断，并为每轮回答补充 EOS；
        4. 构造 attention mask 与 labels。默认只在助手回答位置计算 loss，
           用户输入、系统提示等上下文位置会以 ``IGNORE_INDEX`` 屏蔽。

        本实现的对话处理部分参考了 FastChat：https://github.com/lm-sys/FastChat
        """
        # ``datasets.Dataset.map(..., batched=True)`` 会传入“列名 -> 一个 batch 的值列表”。
        # 下面三个列表中的每个元素对应一个有效对话样本。
        input_ids_list = []
        attention_mask_list = []
        targets_list = []
        # 原始数据中常见的标准角色名称；实际兼容的别名会在下方分别处理。
        roles = ["human", "gpt"]

        def get_dialog(examples):
            """逐条解析 batch 内的对话，并产出交替排列的 prompt / response 文本。

            产出的 ``dialog`` 形式为 ``[prompt_1, answer_1, prompt_2, answer_2, ...]``。
            其中 prompt 已经包含该轮之前的必要对话历史，以便模型学习多轮上下文。
            无法构成至少一组“用户—助手”问答的原始样本会被跳过。
            """
            # 可选的批量级系统提示：当单条对话未显式提供 system 消息时作为兜底。
            system_prompts = examples.get("system_prompt", "")
            for i, source in enumerate(examples['conversations']):
                # 每条样本独立维护系统提示与工具描述，避免不同样本间相互污染。
                system_prompt = ""
                tools_text = ""
                if "tools" in examples and examples["tools"][i]:
                    tools_json = examples["tools"][i]
                    if isinstance(tools_json, str):
                        # 工具定义通常以 JSON 字符串存储；按当前模型约定格式化后放入系统提示。
                        tools_parsed = json.loads(tools_json)
                        if tools_parsed and script_args.tool_format:
                            tu = get_tool_utils(script_args.tool_format)
                            tools_text = tu.tool_formatter(tools_parsed)
                
                # 先统一为 chat-template 可识别的 user / assistant 角色与纯文本内容。
                messages = []
                for sentence in source:
                    role = sentence.get("from", "")
                    value = sentence.get("value", "")
                    
                    if role == "system":
                        # 原始对话中的 system 消息优先级高于字段 ``system_prompt``。
                        system_prompt = value
                        continue
                    
                    if role in ["human", "user", "observation"]:
                        if role == "observation":
                            # 工具执行结果仍作为下一轮用户侧上下文，但需使用各模型约定的包裹格式。
                            if script_args.tool_format == "qwen":
                                value = f"<tool_response>\n{value}\n</tool_response>"
                            elif script_args.tool_format == "glm4":
                                value = f"<|observation|>\n{value}"
                            elif script_args.tool_format == "mistral":
                                value = f"[TOOL_RESULTS] {{\"content\": {value}}}[/TOOL_RESULTS]"
                            else:
                                value = f"Observation: {value}"
                        messages.append({"role": "user", "content": value})
                    elif role in ["gpt", "assistant", "function_call"]:
                        if role == "function_call":
                            # 函数调用在原始数据中为 JSON；将其转成模型训练时应生成的调用文本。
                            fc_dict = json.loads(value)
                            if "name" in fc_dict and "arguments" in fc_dict:
                                if script_args.tool_format:
                                    tu = get_tool_utils(script_args.tool_format)
                                    value = tu.function_formatter([FunctionCall(fc_dict["name"], json.dumps(fc_dict["arguments"], ensure_ascii=False))])
                                else:
                                    value = f"Action: {fc_dict['name']}\nAction Input: {json.dumps(fc_dict['arguments'], ensure_ascii=False)}"
                        messages.append({"role": "assistant", "content": value})

                if tools_text:
                    # 让模型在整段对话开始前即可获知可调用工具；保留已有系统提示内容。
                    system_prompt = system_prompt + ("\n\n" if system_prompt else "") + tools_text

                # 将可能不严格交替的消息修复为若干 [user_content, assistant_content] 对。
                # 连续 user 消息（例如工具 observation）会并入当前问题；孤立 assistant 消息会忽略。
                history_messages = []
                temp_history = []
                for msg in messages:
                    if not temp_history and msg["role"] == "user":
                        temp_history.append(msg["content"])
                    elif len(temp_history) == 1 and msg["role"] == "assistant":
                        temp_history.append(msg["content"])
                        history_messages.append(temp_history)
                        temp_history = []
                    elif msg["role"] == "user" and len(temp_history) == 1:
                        temp_history[0] += "\n" + msg["content"]
                    elif msg["role"] == "assistant" and len(temp_history) == 0:
                        pass
                    elif msg["role"] == "assistant" and len(temp_history) == 2:
                        history_messages[-1][1] += "\n" + msg["content"]
                        
                if not history_messages:
                    # 仅有系统消息、仅有用户消息等样本不能提供监督信号，直接丢弃。
                    continue

                if not system_prompt:
                    # 仅在对话内部没有 system 消息时，才采用同 batch 对应位置的默认系统提示。
                    system_prompt = system_prompts[i] if system_prompts else ""
                if prompt_template:
                    # 自定义模板负责生成每轮的“上下文 prompt + 当前回答”文本对。
                    yield prompt_template.get_dialog(history_messages, system_prompt=system_prompt)
                else:
                    # 没有自定义模板时，使用 tokenizer 的官方 chat template。
                    convs = []
                    accumulated = []
                    if system_prompt:
                        accumulated.append({"role": "system", "content": system_prompt})
                    # ``prev_text`` 用于只截取新增的 prompt 片段，避免把完整历史重复写入序列。
                    prev_text = ""
                    for uq, br in history_messages:
                        accumulated.append({"role": "user", "content": uq})
                        # 加入 generation prompt，得到“截至本轮问题、等待助手回答”的模板文本。
                        cur_text = tokenizer.apply_chat_template(
                            accumulated, tokenize=False, add_generation_prompt=True
                        )
                        convs.append(cur_text[len(prev_text):])
                        convs.append(br)
                        accumulated.append({"role": "assistant", "content": br})
                        # 记录完整已完成历史，供下一轮从中切分出增量 prompt。
                        prev_text = tokenizer.apply_chat_template(
                            accumulated, tokenize=False, add_generation_prompt=False
                        )
                    yield convs

        for dialog in get_dialog(examples):
            # 一个多轮对话最终拼为一个 token 序列；labels 与 input_ids 必须一一对齐。
            input_ids, labels = [], []

            for i in range(len(dialog) // 2):
                # 偶数下标是当前轮 prompt，奇数下标是对应的助手回答。
                # 仅第一轮加入特殊 token，避免在对话中间反复插入 BOS 等标记。
                source_ids = tokenizer.encode(text=dialog[2 * i], add_special_tokens=(i == 0))
                target_ids = tokenizer.encode(text=dialog[2 * i + 1], add_special_tokens=False)

                total_len = len(source_ids) + len(target_ids)
                # 在本轮内部按 prompt / response 原始长度比例分配最大长度，
                # 防止任一侧过长而完全挤占另一侧；EOS 会额外预留一个位置。
                max_source_len = int(max_length * (len(source_ids) / total_len))
                max_target_len = int(max_length * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len - 1:  # eos token
                    target_ids = target_ids[:max_target_len - 1]
                # 某些 tokenizer 可能把 EOS 编进片段中；统一移除，稍后为每轮显式添加一个 EOS。
                if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:
                    source_ids = source_ids[1:]
                if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:
                    target_ids = target_ids[:-1]
                # 当前轮若无法完整放入总长度限制，停止拼接后续轮次，保留已构造的历史。
                if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:
                    break

                # 每轮回答后追加 EOS，明确该轮 assistant 输出的结束位置。
                input_ids += source_ids + target_ids + [tokenizer.eos_token_id]  # add eos token for each turn
                if script_args.train_on_inputs:
                    # 开启后，prompt 与回答均参与交叉熵损失计算。
                    labels += source_ids + target_ids + [tokenizer.eos_token_id]
                else:
                    # 默认 SFT 只监督助手输出；prompt 位置填 IGNORE_INDEX，使 loss 函数忽略它们。
                    labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

            # attention mask 中 1 表示真实 token；后续的数据整理器会负责 padding。
            input_ids_list.append(input_ids)
            attention_mask_list.append([1] * len(input_ids))
            targets_list.append(labels)

        # 返回的字典键与 Hugging Face Trainer / data collator 使用的字段名称保持一致。
        return dict(
            input_ids=input_ids_list,
            attention_mask=attention_mask_list,
            labels=targets_list,
        )

    def filter_empty_labels(example):
        """移除没有任何可监督 token 的样本。

        例如只有用户消息，或长度截断后只剩 prompt 的样本，labels 会全部为 IGNORE_INDEX。
        保留它们既不能提供梯度，又会浪费训练 batch。
        """
        return not all(label == IGNORE_INDEX for label in example["labels"])

    # 构建训练集
    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        # 每次训练先稳定地打乱样本；``max_train_samples`` 在打乱后截取，适合快速调试。
        train_dataset = raw_datasets['train'].shuffle(seed=42)
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        if is_main_process:
            logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")

        with training_args.main_process_first(desc="Train dataset tokenization"):
            # 多进程/多卡时，主进程优先构建 datasets cache，其余进程随后复用，避免同时写缓存。
            # ``remove_columns`` 很关键：原始 conversations/tools 等字段不再传入 Trainer。
            tokenized_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset" if is_main_process else None,
            )
            # 只保留至少有一个非 IGNORE_INDEX label 的样本，确保每个训练样本能够贡献 loss。
            train_dataset = tokenized_dataset.filter(
                filter_empty_labels,
                num_proc=data_args.preprocessing_num_workers
            )

            if is_main_process:
                logger.debug(f"Num train_samples: {len(train_dataset)}")
                logger.debug("Tokenized training example:")
                logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(train_dataset[0]['input_ids'])}")
                replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id
                                   for label in list(train_dataset[0]['labels'])]
                logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")

    # 构建验证集
    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        with training_args.main_process_first(desc="Eval dataset tokenization"):
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            # 按配置限制评估样本数
            if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            eval_size = len(eval_dataset)
            logger.debug(f"Num eval_samples: {eval_size}")
            if eval_size > 500:
                logger.warning(f"Num eval_samples is large: {eval_size}, "
                               f"training slow, consider reduce it by `--max_eval_samples=50`")
            logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
            # 验证集使用与训练集完全相同的模板、截断和 label masking 规则，
            # 否则 eval_loss 与训练目标并不可比。
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            eval_dataset = eval_dataset.filter(filter_empty_labels, num_proc=data_args.preprocessing_num_workers)
            logger.debug(f"Num eval_samples: {len(eval_dataset)}")
            logger.debug("Tokenized eval example:")
            logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))

    # --------------------- 4. 模型加载：精度、并行、长上下文和量化 ----------------------
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        # DDP 由每个进程独占一张卡；此时不能再让 device_map="auto" 把同一模型跨卡切分，
        # 否则会与 DDP 的设备管理冲突。
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        ddp = world_size != 1
        if ddp:
            model_args.device_map = None
        if model_args.device_map in ["None", "none", ""]:
            model_args.device_map = None
        if script_args.qlora and (len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled()):
            logger.warning("FSDP and DeepSpeed ZeRO-3 are both currently incompatible with QLoRA.")

        config_kwargs = {
            "trust_remote_code": model_args.trust_remote_code,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }
        # 先读取 config，再按命令行修改 RoPE/注意力等结构配置，最后据此加载权重。
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

        # RoPE 扩展用于让模型接受比预训练上限更长的 ``model_max_length``；
        # 它只调整位置编码尺度，不能保证超出原上下文窗口后的模型效果。
        if model_args.rope_scaling is not None:
            if hasattr(config, "rope_scaling"):
                if model_args.rope_scaling == "dynamic":
                    logger.warning(
                        "Dynamic NTK may not work well with fine-tuning. "
                        "See: https://github.com/huggingface/transformers/pull/24653"
                    )
                current_max_length = getattr(config, "max_position_embeddings", None)
                if current_max_length and script_args.model_max_length > current_max_length:
                    scaling_factor = float(math.ceil(script_args.model_max_length / current_max_length))
                else:
                    logger.warning(f"The model_max_length({script_args.model_max_length}) is smaller than max "
                                   f"length({current_max_length}). Consider increase model_max_length.")
                    scaling_factor = 1.0

                setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})
                logger.info("Using {} scaling strategy and setting scaling factor to {}".format(
                    model_args.rope_scaling, scaling_factor
                ))
            else:
                logger.warning("Current model does not support RoPE scaling.")

        # FlashAttention-2 主要降低注意力的显存和时间开销；需同时满足安装、GPU 和模型实现要求。
        if model_args.flash_attn:
            if is_flash_attn_2_available:
                config_kwargs["use_flash_attention_2"] = True
                logger.info("Using FlashAttention-2 for faster training and inference.")
            else:
                logger.warning("FlashAttention-2 is not installed.")
        elif model_args.shift_attn and getattr(config, "model_type", None) == "llama":
            logger.warning("Using `--flash_attn` for faster training in large context length, enable if your GPU"
                           " is RTX3090, RTX4090, A100 or H100.")

        # S^2-Attn 仅对实现了该接口的 LLaMA 模型生效，用稀疏注意力降低长序列开销。
        if model_args.shift_attn:
            if getattr(config, "model_type", None) == "llama":
                setattr(config, "group_size_ratio", 0.25)
                logger.info("Using shifted sparse attention with group_size_ratio=1/4.")
            else:
                logger.warning("Current model does not support shifted sparse attention.")

        # 量化仅改变基础权重的存储/计算精度；4-bit 和 8-bit 是互斥选项。
        load_in_8bit = model_args.load_in_8bit
        load_in_4bit = model_args.load_in_4bit
        quantization_config = None
        if load_in_4bit and load_in_8bit:
            raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")
        elif load_in_8bit or load_in_4bit:
            logger.info(f"Quantizing model, load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}")
            if is_deepspeed_zero3_enabled():
                raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")
            if load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif load_in_4bit:
                if script_args.qlora:
                    # QLoRA 的典型设置：NF4 权重 + 双重量化，计算仍使用 torch_dtype。
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                    )

        # 这些参数共同决定模型权重以何精度、在哪些设备上、以何种量化方式加载。
        model_kwargs = {
            "config": config,
            "torch_dtype": torch_dtype,
            "trust_remote_code": model_args.trust_remote_code,
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True,  # 减少CPU内存使用
            "device_map": model_args.device_map,
        }

        # 单进程多卡时，device_map="auto" 会把模型模块分散到多张 GPU；
        # max_memory 预留 20% 空间，避免权重加载即占满显存，给激活值/梯度/优化器留出余量。
        num_gpus = torch.cuda.device_count()
        if model_args.device_map == 'auto':
            if num_gpus > 1 and not ddp:
                # 大模型多GPU：使用auto进行张量并行
                model_kwargs["device_map"] = "auto"
                # 设置最大内存使用
                max_memory = {}
                for i in range(num_gpus):
                    # 为每个GPU预留一些内存给梯度和优化器
                    gpu_props = torch.cuda.get_device_properties(i)
                    total_mem = gpu_props.total_memory
                    # 预留20%内存给训练时的梯度、优化器状态等
                    usable_mem = int(total_mem * 0.8)
                    max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"

                model_kwargs["max_memory"] = max_memory

        logger.info(f"🔧 大模型训练配置:")
        logger.info(f"  model_kwargs: {model_kwargs}")

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs
        )

        logger.info("✅ 模型加载完成")

        # 显示模型分布信息
        logger.info("📊 模型分布情况:")
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            logger.info("🔧 使用HuggingFace设备映射:")
            for module_name, device in model.hf_device_map.items():
                logger.info(f"  {module_name}: {device}")

            # 统计每个GPU上的模块数量
            device_count = {}
            for device in model.hf_device_map.values():
                device_str = str(device)
                device_count[device_str] = device_count.get(device_str, 0) + 1

            logger.info("📈 设备使用统计:")
            for device, count in device_count.items():
                logger.info(f"  {device}: {count} 个模块")
        else:
            # 检查模型参数的设备分布
            device_params = {}
            total_params = 0
            for name, param in model.named_parameters():
                device = str(param.device)
                if device not in device_params:
                    device_params[device] = {'count': 0, 'size': 0}
                device_params[device]['count'] += 1
                device_params[device]['size'] += param.numel()
                total_params += param.numel()

            logger.info("📈 参数设备分布:")
            for device, info in device_params.items():
                param_size_gb = info['size'] * 4 / 1024 ** 3  # 假设float32
                percentage = info['size'] / total_params * 100
                logger.info(f"  {device}: {info['count']} 个参数组, {param_size_gb:.2f}GB ({percentage:.1f}%)")

        # 显示GPU内存使用情况
        if torch.cuda.is_available():
            logger.info("💾 GPU内存使用情况:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                cached = torch.cuda.memory_reserved(i) / 1024 ** 3
                total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                logger.info(f"  GPU {i}: 已分配={allocated:.1f}GB, 缓存={cached:.1f}GB, 总计={total:.1f}GB")

        # 部分模型的输出头命名不为标准 ``lm_head``；统一别名后，LoRA 和保存逻辑才能一致处理。
        if getattr(config, "model_type", None) == "chatglm" or getattr(config, "model_type", None) == "internlm2":
            setattr(model, "lm_head", model.transformer.output_layer)
            setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

        # NEFTune 在训练时给词嵌入添加均匀噪声，是一种正则化技巧；只在 alpha > 0 时开启。
        if model_args.neft_alpha > 0:
            input_embed = model.get_input_embeddings()
            if isinstance(input_embed, torch.nn.Embedding):
                def noisy_forward(self: torch.nn.Embedding, x: torch.Tensor) -> torch.Tensor:
                    embeddings = input_embed.__class__.forward(self, x)
                    dims = self.num_embeddings * self.embedding_dim
                    mag_norm = model_args.neft_alpha / (dims ** 0.5)
                    embeddings += torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)
                    return embeddings

                input_embed.forward = MethodType(noisy_forward, input_embed)
                logger.info("Using noisy embedding with alpha={:.2f}".format(model_args.neft_alpha))
            else:
                logger.warning("Input embeddings are not normal nn.Embedding, cannot transform into noisy embedding.")

        # Patch Mixtral MOE model
        if getattr(config, "model_type", None) == "mixtral" and is_deepspeed_zero3_enabled():
            require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
            from deepspeed.utils import set_z3_leaf_modules  # type: ignore
            from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock  # type: ignore

            set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

        # Patch DeepSeek-V3 MoE module 防止 ZeRO-3 优化器把 Mixtral 的核心专家模块切得太碎导致训练崩溃，把 MixtralSparseMoeBlock 视为一个不可分割的“原子节点
        if getattr(config, "model_type", None) == "deepseek_v3" and is_deepspeed_zero3_enabled():
            require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
            for layer in model.model.layers:
                if 'DeepseekV3MoE' in str(type(layer.mlp)):
                    layer.mlp._z3_leaf = True

        # Patch Qwen3 MoE module
        if getattr(config, "model_type", None) == "qwen3_moe" and is_deepspeed_zero3_enabled():
            require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
            from deepspeed.utils import set_z3_leaf_modules
            from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
            set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])

        # Patch Qwen3.5 MoE module
        if getattr(config, "model_type", None) == "qwen3_5_moe" and is_deepspeed_zero3_enabled():
            require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
            from deepspeed.utils import set_z3_leaf_modules
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
            set_z3_leaf_modules(model, [Qwen3_5MoeSparseMoeBlock])
    else:
        raise ValueError(f"Error, model_name_or_path is None, SFT must be loaded from a pre-trained model")

    if script_args.use_peft:
        # ------------------------ 5. LoRA / QLoRA 可训练参数 ------------------------
        # 基础模型权重保持冻结，梯度只流向低秩适配器；显存和可保存参数量都显著降低。
        logger.info("Fine-tuning method: LoRA(PEFT)")

        # 输出 logits 时转为 fp32，提升损失计算的数值稳定性，尤其适用于半精度/量化底座。
        output_layer = getattr(model, "lm_head")
        if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
            def fp32_forward_post_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
                return output.to(torch.float32)

            output_layer.register_forward_hook(fp32_forward_post_hook)

        # 若给出 peft_path，则在基础模型上继续训练已有 adapter；否则新建 LoRA adapter。
        if script_args.peft_path is not None:
            logger.info(f"Peft from pre-trained model: {script_args.peft_path}")
            model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
        else:
            logger.info("Init new peft model")
            if load_in_8bit or load_in_4bit:
                # 为量化模型启用梯度检查点兼容设置、输入层梯度等，是 QLoRA 正常反传的前置条件。
                model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)
            target_modules = script_args.target_modules.split(',') if script_args.target_modules else None
            if target_modules and 'all' in target_modules:
                # 依模型实际层类型自动选取目标线性层，量化模型会识别 bitsandbytes 层。
                target_modules = find_all_linear_names(model, int4=load_in_4bit, int8=load_in_8bit)
            modules_to_save = script_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            logger.info(f"Peft target_modules: {target_modules}")
            logger.info(f"Peft lora_rank: {script_args.lora_rank}")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                modules_to_save=modules_to_save)
            model = get_peft_model(model, peft_config)  # 把原模型改为LoRA训练的轻量级模型
        # LoRA 参数以 fp32 保存/训练以避免低精度累积误差；基础量化权重不受影响。
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
    else:
        # 全参微调会将所有权重转为 fp32，显存需求远高于 LoRA，需关注 batch size 与 GPU 容量。
        logger.info("Fine-tuning method: Full parameters training")
        model = model.float()
        print_trainable_parameters(model)

    # ----------------------- 6. 训练运行时与 batch 组装 ------------------------
    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        # 以重算前向激活换取显存。开启时必须关掉 KV cache，否则二者的计算图不兼容。
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")
    model.enable_input_require_grads()
    if not ddp and torch.cuda.device_count() > 1:
        # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        # label padding 也填 IGNORE_INDEX，保证补齐位置不会被计入 loss。
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,  # for shifted sparse attention
    )
    # Trainer 接管优化器、学习率调度、梯度累积、混合精度、日志与分布式同步；
    # 这些具体超参数来自 Seq2SeqTrainingArguments 和启动脚本，而非本文件硬编码。
    trainer = SavePeftModelTrainer(  # 只保存LoRA微调部分参数的trainer
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # ------------------------------ 7. 训练与最终保存 ------------------------------
    if training_args.do_train:
        if trainer.is_world_process_zero():
            logger.info("*** Train ***")
            sample = next(iter(trainer.get_train_dataloader()))
            logger.debug(f"Train dataloader example: {sample}")
            logger.debug(f"input_ids:\n{list(sample['input_ids'])[:3]}, \nlabels:\n{list(sample['labels'])[:3]}")
            logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(sample['input_ids'][0])}")
            replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id for label in
                               sample['labels'][0]]
            logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")
        # 指定 resume_from_checkpoint 时，Trainer 会恢复模型、优化器、调度器和训练步数。
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # 训练时因 gradient checkpointing 关闭的 cache 在结束后恢复，便于加载该模型后直接生成。
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"

        if trainer.is_world_process_zero():
            logger.debug(f"Training metrics: {metrics}")
            logger.info(f"Saving model checkpoint to {training_args.output_dir}")
            if is_deepspeed_zero3_enabled():
                # 只有 ZeRO-3 需要先聚合被切分的权重；其他模式直接调用 save_pretrained。
                save_model_zero3(model, tokenizer, training_args, trainer)
            else:
                save_model(model, tokenizer, training_args)

    # --------------------------------- 8. 验证指标 ---------------------------------
    if training_args.do_eval:
        if trainer.is_world_process_zero():
            logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")

        metrics["eval_samples"] = max_eval_samples
        # 困惑度是 token 级交叉熵的指数形式；越低通常表示模型对验证集回答 token 的预测越好。
        # 注意它受 template、截断策略和是否屏蔽输入 token 的影响，只适合在相同设置间横向比较。
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if trainer.is_world_process_zero():
            logger.debug(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()
