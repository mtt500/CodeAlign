# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Train a model from SFT using DPO
"""
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, Optional
import sys

import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import DPOTrainer, DPOConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.tool_utils import get_tool_utils, FunctionCall, load_local_json_datasets
from training.template import get_conv_template

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class ScriptArguments:
    """DPO 训练所需的模型、偏好数据与优化参数。

    DPO 的一个样本必须由同一上下文下的两条候选回答构成：``chosen`` 是偏好回答，
    ``rejected`` 是非偏好回答。训练不会像 SFT 一样直接拟合单一标准答案，而是学习
    让模型相对更倾向 chosen、相对更不倾向 rejected。
    """
    # Model arguments
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The model checkpoint for weights initialization."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the model in 4bit mode or not."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
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
    # ----------------------------- 偏好数据与文本长度 -----------------------------
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The input jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."}, )
    template_name: Optional[str] = field(
        default=None,
        metadata={"help": "The prompt template name. If not set, use tokenizer's built-in chat_template."}
    )
    tool_format: Optional[str] = field(
        default=None,
        metadata={"help": "Tool format to use for agent training. Options: default, glm4, llama3, mistral, qwen."}
    )
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "Train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "Eval batch size per device"})
    # prompt 是 chosen/rejected 共用的上下文（包括系统提示、多轮历史和当前用户问题）。
    max_source_length: Optional[int] = field(default=2048, metadata={"help": "Max length of prompt input text"})
    # target 是一条候选回答；每个样本需同时容纳 chosen 和 rejected 两个“prompt + target”序列。
    max_target_length: Optional[int] = field(default=512, metadata={"help": "Max length of output text"})
    # 注意：当前脚本未在任何地方使用该参数；它不会实际过滤过短回答。
    min_target_length: Optional[int] = field(default=4, metadata={"help": "Min length of output text"})
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
        default=4, metadata={"help": "The number of processes to use for the preprocessing."},
    )
    # ------------------------------- DPO 优化与 LoRA -------------------------------
    # 使用 PEFT 时，policy 仅训练 LoRA adapter；DPOTrainer 会使用其 PEFT 参考策略处理 reference。
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})
    target_modules: Optional[str] = field(default=None)
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=16.0)
    peft_path: Optional[str] = field(default=None)
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the validation set."})
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "Learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "The lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "The number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "The weight decay"})
    optim: Optional[str] = field(default="adamw_torch", metadata={"help": "The optimizer type"})
    fp16: Optional[bool] = field(default=True, metadata={"help": "Whether to use fp16"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "Whether to use bf16"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use gradient checkpointing"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "The number of gradient accumulation steps"}
    )
    save_steps: Optional[int] = field(default=50, metadata={"help": "X steps to save the model"})
    eval_steps: Optional[int] = field(default=50, metadata={"help": "X steps to evaluate the model"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "X steps to log the model"})
    output_dir: Optional[str] = field(default="outputs-dpo", metadata={"help": "The output directory"})
    max_steps: Optional[int] = field(default=200, metadata={"help": "Number of steps to train"})
    eval_strategy: Optional[str] = field(default="steps", metadata={"help": "Evaluation strategy"})
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={"help": "Remove unused columns from the dataset if `datasets.Dataset` is used"},
    )
    report_to: Optional[str] = field(default="tensorboard", metadata={"help": "Report to wandb or tensorboard"})

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


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
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(peft_model, int4=False, int8=False):
    """找出可被 LoRA 注入的线性层名称。

    当 ``target_modules=all`` 时使用。量化模型的线性层类型由 bitsandbytes 提供，
    因而需要与普通 ``torch.nn.Linear`` 分开识别；输出头默认不注入 LoRA。
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


def main():
    # =========================== DPO 阅读主线（建议按此顺序） ===========================
    # 1. return_prompt_and_responses：将原始偏好数据变为 prompt/chosen/rejected 三元组；
    # 2. train_dataset.filter：限制一对候选回答的上下文长度；
    # 3. DPOConfig + DPOTrainer：定义序列上限、policy/reference 模型及真正的 DPO 损失；
    # 4. trainer.train：由 TRL 完成 tokenization、前向、对数概率比较和参数更新。
    #
    # DPO 的优化目标可写作：
    #   -log sigmoid(beta * ((log πθ(y_w|x) - log πref(y_w|x))
    #                        - (log πθ(y_l|x) - log πref(y_l|x))))
    # 其中 x 为 prompt，y_w/y_l 分别是 chosen/rejected，πθ 是待训练 policy，
    # πref 是冻结参考策略。也就是说，优化的是“chosen 相对 rejected 的优势”，
    # 并用参考模型约束 policy 不要无边界地偏离 SFT 初始化模型。
    # ==============================================================================
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    logger.info(f"Parse args: {args}")

    # -------------------------- 1. tokenizer 与对话模板 --------------------------
    # DPO 的 prompt 必须和 SFT/推理时使用同一种 chat template。若 prompt 格式不一致，
    # chosen/rejected 的概率比较仍可执行，但训练出的偏好很难在真实对话中复现。
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer,
        "trust_remote_code": args.trust_remote_code,
    }
    tokenizer_name_or_path = args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    prompt_template = None
    if args.template_name:
        # 指定模板时使用项目内模板；未指定时在下方使用 tokenizer 自带的 chat_template。
        prompt_template = get_conv_template(args.template_name)
    if tokenizer.eos_token_id is None:
        # 候选回答的边界需要 EOS；没有时补齐，保证两个回答都能以一致方式结束。
        if prompt_template:
            tokenizer.eos_token = prompt_template.stop_str
        else:
            tokenizer.eos_token = "</s>"
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"Add eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"Add bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    if tokenizer.pad_token_id is None:
        # 同一个 DPO batch 中 chosen/rejected 的长度不同，因而必须定义 padding token。
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    logger.debug(f"Tokenizer: {tokenizer}")

    # ----------------------------- 2. 加载原始偏好数据集 -----------------------------
    # 期望每条记录拥有 conversations（共享上下文）、chosen（偏好回答）和 rejected（非偏好回答）。
    if args.dataset_name is not None:
        # Hugging Face Hub 数据集路径；若没有 validation 切分，则从 train 的头部切出一部分。
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
            )
    else:
        data_files = {}
        if args.train_file_dir is not None and os.path.exists(args.train_file_dir):
            train_data_files = glob(f'{args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {', '.join(train_data_files)}")
            data_files["train"] = train_data_files
        if args.validation_file_dir is not None and os.path.exists(args.validation_file_dir):
            eval_data_files = glob(f'{args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {', '.join(eval_data_files)}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_local_json_datasets(data_files, cache_dir=args.cache_dir)
        # 本地 JSONL 未提供 validation 时，同样从 train 切出前若干条作为验证集。
        # 注意：这里没有 shuffle；若原始文件按领域/难度排序，验证集分布可能与训练集不同。
        if "validation" not in raw_datasets.keys():
            train_dataset = raw_datasets["train"]
            validation_size = int(len(train_dataset) * args.validation_split_percentage / 100)
            raw_datasets["validation"] = train_dataset.select(range(validation_size))
            raw_datasets["train"] = train_dataset.select(range(validation_size, len(train_dataset)))
    logger.info(f"Raw datasets: {raw_datasets}")

    # ------------------- 3. 核心：原始对话 -> DPO 偏好三元组 --------------------
    # ``full_max_length`` 是一条“prompt + 一个候选回答”的上限，而非两条回答拼接后的长度。
    # DPOTrainer 会分别计算 prompt+chosen 与 prompt+rejected 的序列对数概率。
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length
    full_max_length = max_source_length + max_target_length

    def _format_tool_call_value(value, tool_fmt):
        """将结构化 function_call 转为模型模板规定的文本。

        工具调用出现在共享的对话历史中时，它也是 prompt 的一部分；因此 chosen 与
        rejected 会看到完全相同的工具调用上下文，二者只应在最终候选回答上产生差异。
        """
        fc_dict = json.loads(value)
        if "name" in fc_dict and "arguments" in fc_dict:
            if tool_fmt:
                tu = get_tool_utils(tool_fmt)
                return tu.function_formatter(
                    [FunctionCall(fc_dict["name"], json.dumps(fc_dict["arguments"], ensure_ascii=False))]
                )
            else:
                return f"Action: {fc_dict['name']}\nAction Input: {json.dumps(fc_dict['arguments'], ensure_ascii=False)}"
        return value

    def _format_observation_value(value, tool_fmt):
        """将工具返回结果包装为指定模型可识别的 user 侧上下文格式。"""
        if tool_fmt == "qwen":
            return f"<tool_response>\n{value}\n</tool_response>"
        elif tool_fmt == "glm4":
            return f"<|observation|>\n{value}"
        elif tool_fmt == "mistral":
            return f'[TOOL_RESULTS] {{"content": {value}}}[/TOOL_RESULTS]'
        else:
            return f"Observation: {value}"

    def _build_prompt_from_conversations(source, tools_json, system_prompt_override, tool_fmt):
        """把 ShareGPT 风格的共享历史构造成“等待生成”的 DPO prompt。

        返回值只包含共同上下文，不包含当前样本的 chosen/rejected。本函数最后总是
        添加 assistant generation prompt，随后由 DPOTrainer 分别接上 chosen 与
        rejected，得到两条只在候选答案部分不同的完整序列。
        """
        system_prompt = system_prompt_override or ""
        tools_text = ""

        if tools_json:
            # tools 元数据通常是 JSON 字符串，格式化后与 system prompt 合并，供两条分支共享。
            tools_parsed = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
            if tools_parsed and tool_fmt:
                tu = get_tool_utils(tool_fmt)
                tools_text = tu.tool_formatter(tools_parsed)

        # 先将多种原始角色名归一为 chat template 所需的 user / assistant / system。
        chat_messages = []
        for sentence in source:
            role = sentence.get("from", "")
            value = sentence.get("value", "")

            if role == "system":
                # 对话内的 system 消息优先于外部传入的兜底 system_prompt。
                system_prompt = value
                continue

            if role in ["human", "user", "observation"]:
                if role == "observation":
                    # observation 是工具结果，语义上仍作为下一轮 assistant 的条件输入。
                    value = _format_observation_value(value, tool_fmt)
                chat_messages.append({"role": "user", "content": value})
            elif role in ["gpt", "assistant", "function_call"]:
                if role == "function_call":
                    value = _format_tool_call_value(value, tool_fmt)
                chat_messages.append({"role": "assistant", "content": value})

        if tools_text:
            # 在不覆盖原有 system 指令的前提下拼接工具 schema/描述。
            system_prompt = system_prompt + ("\n\n" if system_prompt else "") + tools_text

        # 规整为 [user, assistant] 历史轮次。连续 user 消息合并，例如“用户问题 + 工具 observation”；
        # 连续 assistant 消息合并，避免模板收到不合法的角色序列。
        temp = []
        history_messages = []
        for msg in chat_messages:
            if not temp and msg["role"] == "user":
                temp.append(msg["content"])
            elif len(temp) == 1 and msg["role"] == "assistant":
                temp.append(msg["content"])
                history_messages.append(temp)
                temp = []
            elif msg["role"] == "user" and len(temp) == 1:
                temp[0] += "\n" + msg["content"]
            elif msg["role"] == "assistant" and len(temp) == 2:
                history_messages[-1][1] += "\n" + msg["content"]
        if len(temp) == 1:
            # 未得到助手回答的最后一个用户消息，就是当前待比较候选答案对应的问题。
            history_messages.append([temp[0], ""])

        if prompt_template:
            # 项目模板负责将历史转换为文本，并在末尾留下 assistant 起始标记。
            return prompt_template.get_prompt(messages=history_messages, system_prompt=system_prompt)
        else:
            # tokenizer 原生模板路径：仅放入已有历史，add_generation_prompt=True 负责追加生成起始标记。
            accumulated = []
            if system_prompt:
                accumulated.append({"role": "system", "content": system_prompt})
            for uq, br in history_messages:
                accumulated.append({"role": "user", "content": uq})
                if br:
                    accumulated.append({"role": "assistant", "content": br})
            return tokenizer.apply_chat_template(
                accumulated, tokenize=False, add_generation_prompt=True
            )

    def return_prompt_and_responses(examples) -> Dict[str, str]:
        """将批量原始样本转为 DPOTrainer 所需的偏好三元组。

        Data format (ShareGPT):
            {conversations: [{from, value}, ...], chosen: str|dict, rejected: str|dict, tools?: str}

        The dataset is converted to a dictionary with the following structure:
        {
            'prompt': List[str],
            'chosen': List[str],
            'rejected': List[str],
        }

        关键不变量：同一条数据中 ``prompt`` 对两个分支必须完全一样；``chosen`` 和
        ``rejected`` 只应表示对当前 prompt 的两个备选答案。若二者属于不同问题、
        使用了不同模板，DPO 的偏好差分将没有意义。
        """
        prompts = []
        chosen_list = []
        rejected_list = []

        tools_list = examples.get("tools", [None] * len(examples["conversations"]))
        for i, source in enumerate(examples["conversations"]):
            # map(batched=True) 下每个字段均为列表，必须用相同索引取出同一条偏好样本。
            tools_json = tools_list[i] if tools_list else None

            chosen_text = examples["chosen"][i]
            rejected_text = examples["rejected"][i]

            # 不在这里手动 token 化或拼接回答。保留文本三元组，由 DPOTrainer 统一分词、
            # padding，并分别计算 chosen/rejected 的 completion log-probability。
            prompt = _build_prompt_from_conversations(
                source, tools_json, "", args.tool_format
            )
            prompts.append(prompt)
            chosen_list.append(chosen_text)
            rejected_list.append(rejected_text)

        return {
            "prompt": prompts,
            "chosen": chosen_list,
            "rejected": rejected_list,
        }

    # ---------------------- 4. 三元组长度过滤与训练集构建 ----------------------
    train_dataset = None
    max_train_samples = 0
    if args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train']
        max_train_samples = len(train_dataset)
        if args.max_train_samples is not None and args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        # 打乱只影响样本顺序；随后 map 会删除原始字段，仅留下 prompt/chosen/rejected。
        tokenized_dataset = train_dataset.shuffle().map(
            return_prompt_and_responses,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        # 这里的 ``len(str)`` 是字符数而非 token 数，因而只是低成本的近似预筛选，
        # 不能严格保证序列不超过 DPOConfig.max_length。真正的分词、截断及 batch padding
        # 由 DPOTrainer 在训练时处理。中英文及不同 tokenizer 下字符/token 比例差异很大。
        # 同时检查两个分支，避免只保留其中一个回答可用的“残缺偏好对”。
        train_dataset = tokenized_dataset.filter(
            lambda x: 0 < len(x['prompt'] + x['chosen']) <= full_max_length
                      and 0 < len(x['prompt'] + x['rejected']) <= full_max_length
        )
        logger.debug(f"Num train_samples: {len(train_dataset)}")
        logger.debug("First train example:")
        first_example = train_dataset[0]
        logger.debug(f"prompt:\n{first_example['prompt']}")
        logger.debug(f"chosen:\n{first_example['chosen']}")
        logger.debug(f"rejected:\n{first_example['rejected']}")

    eval_dataset = None
    max_eval_samples = 0
    if args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        max_eval_samples = len(eval_dataset)
        if args.max_eval_samples is not None and args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
        # 验证必须复用训练集相同的 prompt 构造和过滤规则，才可比较偏好准确率与损失。
        eval_dataset = eval_dataset.map(
            return_prompt_and_responses,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        eval_dataset = eval_dataset.filter(
            lambda x: 0 < len(x['prompt'] + x['chosen']) <= full_max_length
                      and 0 < len(x['prompt'] + x['rejected']) <= full_max_length
        )
        logger.debug(f"Num eval_samples: {len(eval_dataset)}")
        logger.debug("First eval example:")
        first_example = eval_dataset[0]
        logger.debug(f"prompt:\n{first_example['prompt']}")
        logger.debug(f"chosen:\n{first_example['chosen']}")
        logger.debug(f"rejected:\n{first_example['rejected']}")

    # ------------------- 5. policy 模型与 DPO reference 的准备 -------------------
    # 当前加载的 ``model`` 是待优化的 policy πθ，初始化通常来自已经 SFT 的检查点。
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )
    # DDP 中每个进程绑定本地设备；关闭 device_map 跨卡分配，避免与 DDP 管理冲突。
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1
    if ddp:
        args.device_map = None
    if args.device_map in ["None", "none", ""]:
        args.device_map = None
    logger.info(f"Device map: {args.device_map}")
    # QLoRA 与 ZeRO-3 在此组合下不兼容：前者依赖量化权重，后者会切分模型参数。
    if args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 are both currently incompatible with QLoRA.")
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        dtype=torch_dtype,
        cache_dir=args.cache_dir
    )
    # 只有 ``qlora=True`` 时才会把 BitsAndBytesConfig 传入模型加载器；单独设置
    # load_in_4bit/load_in_8bit 在当前代码中只会打印日志，不会启用量化加载。
    if args.load_in_4bit or args.load_in_8bit:
        logger.info(f"Quantizing model, load_in_4bit: {args.load_in_4bit}, load_in_8bit: {args.load_in_8bit}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        ) if args.qlora else None,
    )

    # ZeRO-3 需要把 MoE block 标记为不可分割的 leaf module，避免专家路由相关参数被错误切分。
    if getattr(config, "model_type", None) == "mixtral" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
        set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    if getattr(config, "model_type", None) == "deepseek_v3" and is_deepspeed_zero3_enabled():
        for layer in model.model.layers:
            if 'DeepseekV3MoE' in str(type(layer.mlp)):
                layer.mlp._z3_leaf = True

    if getattr(config, "model_type", None) == "qwen3_moe" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
        set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])

    if getattr(config, "model_type", None) == "qwen3_5_moe" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
        set_z3_leaf_modules(model, [Qwen3_5MoeSparseMoeBlock])

    # 将可训练参数升为 fp32，规避部分 fp16/LoRA 组合在反向传播或优化器更新中的数值/类型错误。
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)

    # 梯度检查点通过在反向阶段重算激活降低显存；开启时必须关闭 KV cache，二者计算图不兼容。
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True

    training_args = DPOConfig(
        # DPOTrainer 使用此上限处理每条 prompt+completion 序列；与前述字符级 filter 不同，
        # 这里的限制最终以 tokenizer token 为单位生效。
        max_length=full_max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        report_to=args.report_to,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=args.remove_unused_columns,
        run_name=f"dpo_v1",
    )

    # --------------------- 6. DPOTrainer：真正的偏好损失在此执行 ---------------------
    peft_config = None
    if args.use_peft:
        # LoRA 只训练低秩适配器，基础模型权重冻结；在 DPO 中常用以降低“policy + reference”开销。
        logger.info("Fine-tuning method: LoRA(PEFT)")
        target_modules = args.target_modules.split(',') if args.target_modules else None
        if target_modules and 'all' in target_modules:
            target_modules = find_all_linear_names(model, int4=args.load_in_4bit, int8=args.load_in_8bit)
        logger.info(f"Peft target_modules: {target_modules}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    else:
        logger.info("Fine-tuning method: Full parameters training")
    trainer = DPOTrainer(
        # ``model`` 是 policy πθ：它对 chosen/rejected 的相对概率会被梯度更新。
        model,
        # 全参训练时显式复制一份冻结模型作为 πref；它应保持 SFT 初始分布，用来计算 KL 风格的
        # 相对锚点。PEFT 模式传 None，让 TRL 按其 PEFT reference 机制处理参考对数概率，
        # 通常通过临时禁用 adapter、以基础模型作为参考，避免复制整个模型造成双倍显存。
        # 具体参考 adapter 的处理细节取决于已安装的 TRL 版本。
        ref_model=None if args.use_peft else deepcopy(model),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # DPOTrainer 使用 tokenizer 将每个文本三元组组装为 chosen/rejected 两个 token 序列，
        # 并仅对 completion token 汇总 log-probability，而非把 prompt 当成偏好目标。
        processing_class=tokenizer,
        peft_config=peft_config if args.use_peft else None,
    )
    print_trainable_parameters(trainer.model)

    # ------------------------------- 7. 执行 DPO 更新 -------------------------------
    if args.do_train:
        if trainer.is_world_process_zero():
            logger.info("*** Train ***")
        # 每一步会同时前向 chosen/rejected（以及 reference），计算二者相对优势的 DPO loss，
        # 反向传播仅更新 policy/LoRA 参数；reference 不接收梯度。
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        if trainer.is_world_process_zero():
            logger.debug(f"Training metrics: {metrics}")
            logger.info(f"Saving model checkpoint to {args.output_dir}")
            trainer.save_model(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            trainer.model.save_pretrained(args.output_dir)

    # ------------------------------- 8. 验证偏好效果 -------------------------------
    if args.do_eval:
        if trainer.is_world_process_zero():
            logger.info("*** Evaluate ***")
        # DPOTrainer 返回偏好目标相关的 eval 指标；其含义依赖当前 TRL 版本和 DPOConfig 设置。
        metrics = trainer.evaluate()
        metrics["eval_samples"] = max_eval_samples
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if trainer.is_world_process_zero():
            logger.debug(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()
