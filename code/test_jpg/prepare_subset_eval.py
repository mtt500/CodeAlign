import gzip
import json

input_path = "./data/HumanEvalPlus.jsonl.gz"  # 你的 Plus 数据集路径
output_path = "./data/HumanEvalPlus_100.jsonl.gz"     # 裁剪后的新文件路径

subset_data = []
# 读取原版压缩包
with gzip.open(input_path, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 100:  # 限制只取前 100 道题
            break
        subset_data.append(json.loads(line))

# 保存为新的压缩包
with gzip.open(output_path, "wt", encoding="utf-8") as f:
    for item in subset_data:
        f.write(json.dumps(item) + "\n")

print(f"✅ 成功截取前 {len(subset_data)} 道题的标准答案，已保存至 {output_path}")