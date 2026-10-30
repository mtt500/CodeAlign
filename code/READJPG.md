# 训练路线（code 背景）
- Qwen2.5-7B  
- SFT  
- SFT + DPO  
- SFT + GRPO  
- SFT + OPD  

<br><br>

# 数据
### 1. SFT 有监督微调
* **原始数据集**：`Magicoder` (`data-evol_instruct-decontaminated.jsonl`)
* **数据量**：1000 条
* **核心目标**：教会模型基本的代码语法，并强制让其养成在回答前使用 `<think>` 标签输出思维链、随后输出标准 Markdown 代码块的规范。
* **最终转换格式** (ShareGPT 对话格式)：
  ```json
  {
    "conversations": [
      {
        "from": "human",
        "value": "Please amend the subsequent Python script..."
      },
      {
        "from": "assistant",
        "value": "<think>\nThis is a programming task. I will think step-by-step to design the solution before writing the code.\n</think>\n\n```python\n# Establish an integer list\narr = [1, 2, 3, 4]\n...```"
      }
    ]
  }

### 2. DPO (直接偏好优化) 阶段：逻辑纠偏与好坏对比
* **原始数据集**：UltraInteract_pair (UltraInteract_pair_0000_pair.parquet)

* **数据量**：500 条

* **核心目标**：提供高质量的优雅代码（Chosen）与包含缩进错误、死循环或逻辑漏洞的劣质代码（Rejected）。

* **最终转换格式** (包含偏好对的三元组)：

```JSON
{
  "conversations": [
    {
      "from": "human",
      "value": "Write Python code to solve the task:\nMike has a string s..."
    }
  ],
  "chosen": "Step 1: To solve this problem...\n```python\ns = input()...```",
  "rejected": "Step 1: We need to determine...\n```python\ndef is_palindrome(s):...```"
}
```

### 3. GRPO (组相对策略优化) 阶段：沙盒试错与自我进化
* **原始数据集**：MBPP (mbpp.jsonl)

* **数据量**：974 条（全量数据）

* **核心目标**：摒弃传统标准答案，仅向模型下发考题。模型需在线生成多条代码，由执行沙盒运行附带的测试用例进行打分，通过组内相对优势（Advantage）的强化学习突破推理天花板。

* **最终转换格式** (仅包含考题与断言)：

```JSON
{
  "prompt": "Write a function to find the minimum cost path to reach (m, n) from (0, 0)...",
  "test_cases": "assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8\nassert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12"
}
```

### 4. OPD (在线偏好蒸馏) 阶段：名师指点与在线纠偏
- **原始数据集**：`Magicoder` (`data-evol_instruct-decontaminated.jsonl`)
- **数据量**：1000 条（严格跳过 SFT 阶段使用过的前 1000 条数据，保证模型面对的是全新考题）
- **核心目标**：放弃离线静态的错题本，让 7B 学生模型（Student）针对纯题目当场生成多份代码，由强大的 27B/32B 教师模型（Teacher）在线充当裁判（LLM-as-a-Judge）进行打分。系统动态提取最好和最差的答案构造偏好对，实时纠正模型当下真实暴露的逻辑短板。
- **最终转换格式** (为通过底层框架的严格校验，将纯题目包装成了包含空回答占位符的对话格式)：
  ```json
  {
    "conversations": [
      {
        "from": "human",
        "value": "Write a Python function to check whether a given number is an Armstrong number or not."
      },
      {
        "from": "assistant",
        "value": ""
      }
    ]
  }

<br><br>

# 训练过程
## sft
一共 1000 条数据  
`--num_train_epochs 3` 表示所有的数据看3遍  
`--model_max_length 2048` 保证所有数据都能看到不被截断  
`--per_device_train_batch_size 2` 单卡批次大小，1 张显卡在一次计算中同时处理几条数据  
`--gradient_accumulation_steps 8` 梯度累积步数，所以我们让显卡计算完 2 条数据的梯度后，先不更新参数，而是把它存在一个临时的池子里，连续计算 8 次，然后把这 8 次的梯度加在一起，执行一次统一的参数更新。 
`--logging_steps 5` 这样框架每走 5 步，就会往 TensorBoard 里画一个点。

处于 `med` 目录下
```bash
scripts/run_sft.sh 2>&1 | tee outputs/code/outputs-code-sft-qwen-2.5-7b/sft.log
```

q：batch是多少？  
a：真实全局 Batch Size = 2（单卡）× 8（累积）× 2（双卡）= 32。模型每看完 32 条数据，才会真正迈出训练的 1 步。

q：100%|██████████| 93/93 中的 93 是什么意思？  
a：整个训练任务从头到尾要执行的总步数，每轮 31 步，所有数据看 3 轮。

q：训练配置和总用时？  
a：两张 NVIDIA A800 80GB显卡，总用时 7:40

## 合并 LoRA
用 `tools/merge_peft_adapter.py` 合并 LoRA 微调的部分权重和原始冻结权重。
```bash
python tools/merge_peft_adapter.py \
    --base_model /data/kcl/myt/Qwen2.5-7B-Instruct \
    --tokenizer_path /data/kcl/myt/Qwen2.5-7B-Instruct \
    --lora_model ./outputs/code/outputs-code-sft-qwen-2.5-7b \
    --output_dir ./outputs/code/outputs-code-sft-merged
```

## DPO
`--max_steps -1` 解除绝对步数锁定，将控制权交还给 Epoch。-1 是一个特殊的占位符，相当于告诉框架：“不要用固定的步数来强行终止训练”。  
`--num_train_epochs 2` 所有的数据一共看 2 遍

```bash
scripts/run_dpo.sh 2>&1 | tee outputs/code/outputs-code-dpo-qwen-2.5-7b/dpo.log
```

指标
```
***** train metrics *****
  epoch                    =     66.8889
  total_flos               = 588715982GF
  train_loss               =      0.0298
  train_runtime            =  0:52:08.65
  train_samples            =         495
  train_samples_per_second =       4.091
  train_steps_per_second   =       0.064
  
***** eval metrics *****
  epoch                    =    66.8889
  eval_entropy             =     0.2784
  eval_logits/chosen       =    -0.0296
  eval_logits/rejected     =    -0.2112
  eval_logps/chosen        =  -342.9422
  eval_logps/rejected      =  -486.7789
  eval_loss                =        0.0
  eval_mean_token_accuracy =     0.8149
  eval_num_tokens          = 12161320.0
  eval_rewards/accuracies  =        1.0
  eval_rewards/chosen      =   -14.3786
  eval_rewards/margins     =    19.6888  √ 好坏答案的差距
  eval_rewards/rejected    =   -34.0674
  eval_runtime             = 0:00:00.88
  eval_samples             =          5
  eval_samples_per_second  =      5.666
  eval_steps_per_second    =        3.4
```

在 tensorboard 中查看
```bash
tensorboard --logdir outputs/code/outputs-code-dpo-qwen-2.5-7b --bind_all --port 6006
```

q：总用时？  
a： 11%|█         | 1/9 [00:45<06:03, 45.46s/it]


merge
```bash
python tools/merge_peft_adapter.py \
    --base_model ./outputs/code/outputs-code-sft-merged \
    --tokenizer_path ./outputs/code/outputs-code-sft-merged \
    --lora_model ./outputs/code/outputs-code-dpo-qwen-2.5-7b \
    --output_dir ./outputs/code/outputs-code-dpo-merged
```

## GRPO
修改：
1. 替换 SYSTEM_PROMPT
2. 重新编写了两个 reward_funcs：extract_python_code，format_reward，分别用于提取代码块和放入子进程中打分，在 GRPOTrainer 传入

新的奖励：  
1. 格式奖励：利用正则检查是否包含 \<think>...\</think> 以及 Markdown 标准的 python ...  代码块。  
2. 准确率奖励：彻底弃用数学公式解析，转为沙盒真实执行。利用正则提取出纯 Python 代码体，将其与 test_cases（各种 assert 断言）拼接在一起，通过 subprocess 放入独立的 Python 进程运行。如果退出码为 0（无 Bug 且通过断言）给 1 分，语法报错给 0 分，死循环超时（超过 2 秒）给 -0.5 分的严厉惩罚。

训练 1 轮（所有数据看 1 遍）

NOTE: GRPO 训练集中没有标准答案

```bash
export TMPDIR="/data/kcl/myt/guojiaxun_workspace/cache"
export TEMP="/data/kcl/myt/guojiaxun_workspace/cache"
export TMP="/data/kcl/myt/guojiaxun_workspace/cache"
export HF_DATASETS_CACHE="/data/kcl/myt/guojiaxun_workspace/cache/huggingface/datasets"

scripts/run_grpo.sh 2>&1 | tee outputs/code/outputs-code-grpo-qwen-2.5-7b/grpo.log
```

q：训练用时？  
a：100%|██████████| 438/438 [2:33:48<00:00, 21.07s/it]

q：为什么是 438 步？  
a：MBPP 数据集总共 974 条，拿走 10% 做验证，剩下 867 条训练数据。每张卡 batch_size=1，gradient_accumulation_steps=1,2 张卡。总步数：876 / 2 = 438.

指标
```
***** train metrics *****
  clip_ratio/high_max                =        0.0
  clip_ratio/high_mean               =        0.0
  clip_ratio/low_mean                =        0.0
  clip_ratio/low_min                 =        0.0
  clip_ratio/region_mean             =        0.0
  completions/clipped_ratio          =        0.0
  completions/max_length             =     432.25
  completions/max_terminated_length  =     432.25
  completions/mean_length            =   278.5781
  completions/mean_terminated_length =   278.5781
  completions/min_length             =     167.75
  completions/min_terminated_length  =     167.75
  entropy                            =     0.7916
  epoch                              =        1.0
  frac_reward_zero_std               =      0.625
  kl                                 =     0.0002
  num_tokens                         =  1210938.0
  reward                             =     1.2344
  reward_std                         =     0.3472
  rewards/code_execution_reward/mean =       0.25
  rewards/code_execution_reward/std  =      0.354
  rewards/format_reward/mean         =     0.9844
  rewards/format_reward/std          =     0.0442
  step_time                          =    24.7844
  total_flos                         =        0GF
  train_loss                         =     0.0177
  train_runtime                      = 2:35:43.81
  train_samples                      =        876
  train_samples_per_second           =      0.094
  train_steps_per_second             =      0.047
```

merge
```bash
python tools/merge_peft_adapter.py \
    --base_model ./outputs/code/outputs-code-sft-merged \
    --tokenizer_path ./outputs/code/outputs-code-sft-merged \
    --lora_model ./outputs/code/outputs-code-grpo-qwen-2.5-7b \
    --output_dir ./outputs/code/outputs-code-grpo-merged
```

## OPD
sft 数据来源的另外 1000 条数据

学习率从 sft的 5e-5 降到 5e-6（超参数中指定的是 Peak Learning Rate）

教师模型：Qwen/Qwen2.5-Coder-32B-Instruct

```bash
scripts/run_opd.sh 2>&1 | tee outputs/code/outputs-code-opd-qwen-2.5-7b/opd.log
```

q：训练总用时？  
a：100%|██████████| 31/31 [05:24<00:00, 10.47s/it]

q：loss 为什么变大了？  
a：因为在偏离原来的基准

merge
```bash
python tools/merge_peft_adapter.py \
    --base_model ./outputs/code/outputs-code-sft-merged \
    --tokenizer_path ./outputs/code/outputs-code-sft-merged \
    --lora_model ./outputs/code/outputs-code-opd-qwen-2.5-7b \
    --output_dir ./outputs/code/outputs-code-opd-merged
```

<br><br>

# 训练结果

## 在线测试
### 测试命令
生成答案并提取纯代码
```bash
export HF_HOME="/data/kcl/myt/guojiaxun_workspace/cache/huggingface"
export HF_DATASETS_CACHE="/data/kcl/myt/guojiaxun_workspace/cache/huggingface/datasets"

测试全部 164 道题
python test_jpg/run_eval.py --model_path ./outputs/code/outputs-code-sft-merged --output_file ./test_jpg/output_code/sft.jsonl

指定测试题目数量（100题）
python test_jpg/run_eval.py --model_path ./outputs/code/outputs-code-sft-merged --output_file ./test_jpg/output_code/sft.jsonl --max_samples 100
```

测试（以 sft 为例）
```bash
export XDG_CACHE_HOME="/data/kcl/myt/guojiaxun_workspace/cache"
export HF_HOME="/data/kcl/myt/guojiaxun_workspace/cache/huggingface"
export HUMANEVAL_OVERRIDE_PATH="/data/kcl/myt/guojiaxun_workspace/med/test_jpg/data/HumanEvalPlus.jsonl.gz"

evalplus.evaluate --dataset humaneval --samples ./output_code/sft.jsonl
```

### 测试结果
base
```
humaneval (base tests)
pass@1: 0.880
humaneval+ (base + extra tests)
pass@1: 0.810
```


sft
```
humaneval (base tests)
pass@1: 0.817
humaneval+ (base + extra tests)
pass@1: 0.750
```

dpo
```
humaneval (base tests)
pass@1: 0.840
humaneval+ (base + extra tests)
pass@1: 0.770
```

grpo
```
humaneval (base tests)
pass@1: 0.860
humaneval+ (base + extra tests)
pass@1: 0.810
```


opd
```
humaneval (base tests)
pass@1: 0.860
humaneval+ (base + extra tests)
pass@1: 0.810
```


base-coder
```
humaneval (base tests)
pass@1: 0.909
humaneval+ (base + extra tests)
pass@1: 0.848
```



## 交互测试
### 测试命令
不合并 LoRA，则提供 lora_model
```bash
python3 demo/inference.py --base_model Qwen/Qwen3.5-2B --lora_model outputs-sft-qwen-v1 --interactive
```
合并 LoRA，e.g. 测试 grpo
```bash
python3 demo/inference.py --base_model ./outputs/code/outputs-code-grpo-merged --interactive
```

### 交互测试 - q1
测试目的：  
这是一道结合了 BFS（广度优先搜索）和三维状态管理的极品算法题（最短路径 + 穿墙次数限制）。SFT / DPO 表现：大概率只能写出普通的二维 BFS，忘记把“剩余穿墙次数 $k$”也作为一个状态维度去记录，导致死循环或者找错最短路径。OPD / GRPO 表现：经过 27B 教师指点或沙盒试错的模型，在 <think> 中会明确写出“我们需要一个三维的 visited 集合：(r, c, k)”，并且知道如何提前剪枝（如果剩余步数小于曼哈顿距离直接 return）。

### q:  
You are given an m x n integer matrix `grid` where each cell is either 0 (empty) or 1 (obstacle). You can move up, down, left, or right to an empty cell in one step. You are also given an integer `k`. You can eliminate at most `k` obstacles. Write a Python function `shortest_path(grid, k)` to return the minimum number of steps to walk from the upper left corner (0, 0) to the lower right corner (m-1, n-1). If it is not possible to find such walk return -1. Please handle the state management carefully to avoid infinite loops and optimize for Time Limit Exceeded (TLE).

a:

1) base
````
To solve this problem, we can use a modified version of Dijkstra's algorithm with a priority queue to find the shortest path while considering the number of obstacles that can be eliminated. We will also use a set to keep track of visited states to avoid revisiting them and falling into infinite loops.

Here's the implementation of the `shortest_path` function:

```python
from heapq import heappop, heappush

def shortest_path(grid, k):
    m, n = len(grid), len(grid[0])
    if m == 1 and n == 1:
        return 0  # Already at the destination
    
    # Directions: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Initialize the priority queue with the starting point and the number of steps (0) and obstacles (0)
    queue = [(0, 0, (0, 0))]  # (steps, obstacles, (row, col))
    visited = set([(0, 0, 0)])  # (obstacles, (row, col))
    
    while queue:
        steps, obstacles, (row, col) = heappop(queue)
        
        # If we reach the bottom-right corner, return the number of steps
        if row == m - 1 and col == n - 1:
            return steps
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check if the new position is within bounds
            if 0 <= new_row < m and 0 <= new_col < n:
                new_obstacles = obstacles + grid[new_row][new_col]
                
                # Check if we can visit this cell with the current number of obstacles
                if new_obstacles <= k and (new_obstacles, (new_row, new_col)) not in visited:
                    visited.add((new_obstacles, (new_row, new_col)))
                    heappush(queue, (steps + 1, new_obstacles, (new_row, new_col)))
    
    # If we cannot reach the bottom-right corner, return -1
    return -1

# Example usage:
grid = [
    [0, 0, 0],
    [1, 1, 0],
    [0, 0, 0],
    [0, 1,
````

2) sft
````
To solve this problem, we can use a breadth-first search (BFS) approach to find the shortest path from the start to the end while considering the maximum number of obstacles that can be eliminated. Here's a Python implementation that meets the requirements:

```python
from collections import deque

def shortest_path(grid, k):
    if not grid or not grid[0]:
        return -1
    
    m, n = len(grid), len(grid[0])
    if m == 1 and n == 1:
        return 0
    
    # Directions for moving up, down, left, right
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    
    # Initialize the queue with the starting position and the number of obstacles eliminated
    queue = deque([(0, 0, 0, k)])  # (x, y, steps, k)
    visited = set([(0, 0, k)])
    
    while queue:
        x, y, steps, remaining_k = queue.popleft()
        
        # If we reach the bottom-right corner, return the number of steps
        if x == m - 1 and y == n - 1:
            return steps
        
        # Explore all four possible directions
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check if the new position is within the grid bounds
            if 0 <= nx < m and 0 <= ny < n:
                new_k = remaining_k - grid[nx][ny]
                if new_k >= 0 and (nx, ny, new_k) not in visited:
                    visited.add((nx, ny, new_k))
                    queue.append((nx, ny, steps + 1, new_k))
    
    # If we cannot reach the bottom-right corner, return -1
    return -1
```

### Explanation:
1. **Initialization**: We initialize the queue with the starting position `(0, 0)` and the number of obstacles eliminated `k`. We also initialize a set `visited` to keep track of visited states to avoid revisiting them.
2. **BFS Loop**: We process each state in the queue. For each state, we check if we have reached the bottom-right corner. If so, we return the number of steps taken.
3. **Direction Exploration**: For each state, we explore all four possible directions (up
````

3) sft + dpo
````
To solve this problem, we can use a Breadth-First Search (BFS) approach with a priority queue to keep track of the minimum number of steps and the number of obstacles eliminated. This way, we can efficiently explore the shortest path while considering the constraints of the number of obstacles that can be eliminated.

Here is the Python implementation:

```python
from heapq import heappop, heappush
from collections import deque

def shortest_path(grid, k):
    m, n = len(grid), len(grid[0])
    if m == 1 and n == 1:
        return 0
    
    # Initialize the priority queue with the starting position and the number of obstacles eliminated
    pq = [(0, 0, 0)]  # (steps, obstacles_eliminated, row, col)
    visited = set([(0, 0, 0)])
    
    while pq:
        steps, obstacles_eliminated, row, col = heappop(pq)
        
        # Explore all 4 possible directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            
            if 0 <= new_row < m and 0 <= new_col < n:
                new_obstacles_eliminated = obstacles_eliminated + grid[new_row][new_col]
                
                if new_row == m - 1 and new_col == n - 1:
                    return steps + 1
                
                if (new_row, new_col, new_obstacles_eliminated) not in visited and new_obstacles_eliminated <= k:
                    visited.add((new_row, new_col, new_obstacles_eliminated))
                    heappush(pq, (steps + 1, new_obstacles_eliminated, new_row, new_col))
    
    return -1
```

This solution works as follows:

1. We initialize a priority queue with the starting position (0,0) and the number of obstacles eliminated (0). We also initialize a set to keep track of visited states to avoid revisiting them.
2. We explore all four possible directions (up, down, left, right) from the current position.
3. For each new position, we calculate the new number of obstacles eliminated. If we reach the bottom-right corner, we return the number of steps taken.
4. We add the new state to the priority queue if it
````

4) sft + grpo
````
To solve this problem, we can use a breadth-first search (BFS) approach with a priority queue to keep track of the number of steps taken and the number of obstacles eliminated. This way, we can explore all possible paths efficiently and keep track of the minimum number of steps required to reach the destination.

Here's a Python function to solve the problem:

```python
from collections import deque

def shortest_path(grid, k):
    m, n = len(grid), len(grid[0])
    if m == 1 and n == 1:
        return 0  # No need to move if we are already at the destination
    
    # Directions for moving up, down, left, and right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Initialize the queue with the starting position and the number of steps and obstacles eliminated
    queue = deque([(0, 0, 0, k)])  # (x, y, steps, k_left)
    visited = set([(0, 0, k)])
    
    while queue:
        x, y, steps, k_left = queue.popleft()
        
        # If we reach the destination
        if x == m - 1 and y == n - 1:
            return steps
        
        # Explore all four possible directions
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check if the new position is within the grid bounds
            if 0 <= nx < m and 0 <= ny < n:
                # If the new position is an obstacle and we still have k_left
                if grid[nx][ny] == 1 and k_left > 0 and (nx, ny, k_left - 1) not in visited:
                    visited.add((nx, ny, k_left - 1))
                    queue.append((nx, ny, steps + 1, k_left - 1))
                # If the new position is empty
                elif grid[nx][ny] == 0 and (nx, ny, k_left) not in visited:
                    visited.add((nx, ny, k_left))
                    queue.append((nx, ny, steps + 1, k_left))
    
    # If we can't reach the destination
    return -1
```

### Explanation:
1. **Initialization**: We initialize the queue with the starting position
````

5) sft + opd
````
To solve this problem, we can use a breadth-first search (BFS) approach with a priority queue to handle the state management efficiently. BFS is suitable for finding the shortest path in an unweighted grid, and we can use a priority queue to keep track of the number of obstacles eliminated so far.

Here's the Python implementation of the `shortest_path` function:

```python
import heapq

def shortest_path(grid, k):
    if not grid or not grid[0]:
        return -1
    
    m, n = len(grid), len(grid[0])
    if m == 1 and n == 1:
        return 0
    
    # Directions for moving up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Priority queue: (steps, obstacles_eliminated, current_row, current_col)
    pq = [(0, k, 0, 0)]
    visited = set([(0, 0, k)])
    
    while pq:
        steps, obstacles_left, row, col = heapq.heappop(pq)
        
        # If we reach the bottom-right corner, return the number of steps
        if row == m - 1 and col == n - 1:
            return steps
        
        # Explore all 4 possible directions
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check if the new position is within the grid boundaries
            if 0 <= new_row < m and 0 <= new_col < n:
                # If the new position is an obstacle
                if grid[new_row][new_col] == 1:
                    if obstacles_left > 0 and (new_row, new_col, obstacles_left - 1) not in visited:
                        visited.add((new_row, new_col, obstacles_left - 1))
                        heapq.heappush(pq, (steps + 1, obstacles_left - 1, new_row, new_col))
                # If the new position is empty
                else:
                    if (new_row, new_col, obstacles_left) not in visited:
                        visited.add((new_row, new_col, obstacles_left))
                        heapq.heappush(pq, (steps + 1, obstacles_left, new_row, new_col))
    
    # If we cannot reach the bottom-right corner
    return -1
````

6) Qwen2.5-Coder-32B-Instruct （stream_generate_answer 中最大生成长度调整为 1024）
````
To solve this problem, we can use a Breadth-First Search (BFS) approach. BFS is suitable for finding the shortest path in an unweighted grid. However, since we can eliminate up to `k` obstacles, we need to keep track of the number of obstacles eliminated so far to avoid revisiting the same state unnecessarily.

Here's the implementation of the `shortest_path` function:

```python
from collections import deque

def shortest_path(grid, k):
    m, n = len(grid), len(grid[0])
    if m == 1 and n == 1:
        return 0
    
    # Directions for moving up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Queue for BFS: (x, y, steps, remaining_k)
    queue = deque([(0, 0, 0, k)])
    
    # Visited set to keep track of visited states (x, y, remaining_k)
    visited = set((0, 0, k))
    
    while queue:
        x, y, steps, remaining_k = queue.popleft()
        
        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check if the new position is within bounds
            if 0 <= nx < m and 0 <= ny < n:
                if grid[nx][ny] == 0:
                    # If the cell is empty, move to it
                    if (nx, ny, remaining_k) not in visited:
                        visited.add((nx, ny, remaining_k))
                        queue.append((nx, ny, steps + 1, remaining_k))
                else:
                    # If the cell is an obstacle and we can eliminate it
                    if remaining_k > 0 and (nx, ny, remaining_k - 1) not in visited:
                        visited.add((nx, ny, remaining_k - 1))
                        queue.append((nx, ny, steps + 1, remaining_k - 1))
        
        # Check if we have reached the bottom-right corner
        if (x, y) == (m - 1, n - 1):
            return steps
    
    # If we exhaust the queue without finding a path
    return -1
```

### Explanation:

1. **Initialization**:
   - We initialize a
````
