#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 training/grpo_training.py \
    --model_name_or_path ./outputs/code/outputs-code-sft-merged \
    --train_file_dir ./data/code/grpo \
    --train_samples -1 \
    --max_steps -1 --num_train_epochs 1 \
    --save_steps 100 \
    --save_strategy steps \
    --save_total_limit 13 \
    --output_dir outputs/code/outputs-code-grpo-qwen-2.5-7b \
    --dtype float16 \
    --bf16 False \
    --fp16 True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --beta 0.001 \
    --learning_rate 5.0e-7 \
    --lr_scheduler_type cosine \
    --warmup_steps 10 \
    --use_vllm False \
    --logging_steps 10 \
    \
    `# QLoRA配置` \
    --use_peft True \
    --qlora False \
    --load_in_4bit False \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    \
    `# 显存优化配置` \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --num_generations 4 \
    --gradient_accumulation_steps 1 \
    --max_completion_length 1024

echo "训练完成!"
