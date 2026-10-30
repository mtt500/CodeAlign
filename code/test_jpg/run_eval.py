"""
python test_jpg/run_eval.py --model_path ./outputs/code/outputs-code-sft-merged --output_file ./test_jpg/output_code/sft.jsonl
"""

import argparse
import json
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def extract_python_code(text):
    """剔除 <think> 过程，精准提取纯代码给阅卷机"""
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--output_file", type=str, required=True, help="保存的 jsonl 文件名")
    parser.add_argument("--max_samples", type=int, default=-1, help="仅测试前 N 道题，用于快速调试")
    args = parser.parse_args()

    print(f"🚀 正在加载模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        device_map="auto", 
        torch_dtype=torch.float16
    )

    # 加载 HumanEval 考试题库 (164道题)
    # 从本地加载 HumanEval 考试题库
    print("📚 正在从本地读取 HumanEval 题库...")
    # 注意：本地读取 JSON 文件时，默认的 split 名称是 "train"
    ds = load_dataset("json", data_files="./test_jpg/data/HumanEval.jsonl.gz", split="train")

    # 👉 增加这几行：如果设置了 max_samples 且大于 0，则进行切片截取
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        print(f"⚡ 调试模式开启：当前仅测试前 {len(ds)} 道题")

    results = []
    # 遍历每道题
    for item in tqdm(ds, desc="模型做题中"):
        # 组装 Prompt：提示模型这是续写任务
        prompt = item["prompt"]
        messages = [
            {"role": "system", "content": "You are an expert Python programmer. Write code to complete the function."},
            {"role": "user", "content": f"Please complete the following Python code:\n```python\n{prompt}\n```"}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # 贪婪解码 (Temperature=0.0) 是代码评测的行业标准
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.0, do_sample=False)
            
        # 截取新生成的部分
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 提取纯代码并与题目给定的前缀拼接 (HumanEval 要求提交完整代码)
        solution = extract_python_code(response)
        full_code = prompt + "\n" + solution
        
        results.append({"task_id": item["task_id"], "solution": full_code})

    # 保存答卷
    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"✅ 考试完成！答卷已保存至 {args.output_file}")

if __name__ == "__main__":
    main()