CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 training/dpo_training.py \
    --model_name_or_path ./outputs/code/outputs-code-sft-merged \
    --train_file_dir ./data/code/dpo \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --max_steps -1 \
    --num_train_epochs 2 \
    --eval_steps 20 \
    --save_steps 50 \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --output_dir outputs/code/outputs-code-dpo-qwen-2.5-7b \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --bf16 False \
    --fp16 True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --tool_format default \
    --cache_dir ./cache \
    --logging_steps 2

    # --validation_file_dir ./data/reward \
# scripts/run_dpo.sh 2>&1 | tee outputs/code/outputs-code-dpo-qwen-2.5-7b/dpo.log