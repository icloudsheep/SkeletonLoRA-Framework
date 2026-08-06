# SkeletonLoRA-Framework 脚本使用说明

本文档提供 `download.sh`、`run.sh` 和 `evaluate.sh` 的常用命令。所有命令均在项目根目录执行。训练和评估前需激活项目约定的 Conda 环境：

```shell
cd /path/to/SkeletonLoRA-Framework
conda activate skeleton_lora_fe
```

配置文件是模型、数据集、联邦训练超参数和评测参数的唯一真相源。`configs/default.yaml` 与 `configs/loss.yaml` 是基准配置；新实验应复制基准配置并在派生文件中修改，不要仅根据本文档中的示例推断当前参数。

## download.sh 的使用案例

命令格式：

```shell
bash download.sh [TARGET ...]
```

不传目标或传入 `all` 时下载全部资源：

```shell
bash download.sh
bash download.sh all
```

也可以只下载单个资源，或一次下载多个资源：

```shell
# OpenLLaMA 3B v2
bash download.sh llama3bv2

# OpenLLaMA 7B v2
bash download.sh llama7bv2

# Dolly 与 Super-NaturalInstructions 训练集
bash download.sh dolly natural-instructions

# MMLU 与 GSM8K 训练集
bash download.sh mmlu-train gsm8k-train

# MMLU 与 GSM8K 测试集
bash download.sh mmlu gsm8k

# 查看脚本内置帮助
bash download.sh --help
```

可用目标及主要落盘位置：

| TARGET | 内容 | 路径 |
| --- | --- | --- |
| `llama3bv2` | OpenLLaMA 3B v2 | `models/open_llama_3b_v2/` |
| `llama7bv2` | OpenLLaMA 7B v2 | `models/open_llama_7b_v2/` |
| `dolly` | Databricks Dolly 15k | `datasets/databricks-dolly-15k/` |
| `natural-instructions` | Super-NaturalInstructions train | `datasets/natural-instructions/train/` |
| `mmlu-train` | MMLU auxiliary train | `datasets/mmlu/mmlu_auxiliary_train.jsonl` |
| `gsm8k-train` | GSM8K main train | `datasets/gsm8k/train.jsonl` |
| `mmlu` | MMLU test | `evaluation/mmlu/mmlu_test.jsonl` |
| `gsm8k` | GSM8K main test | `evaluation/gsm8k/test.jsonl` |

下载通过项目内的 `hfd.sh` 完成，默认使用 `https://hf-mirror.com`。需要切换 Hugging Face 端点时可临时指定：

```shell
HF_ENDPOINT=https://huggingface.co bash download.sh llama3bv2
```

已完整下载的仓库文件会由下载工具检查并复用，不会重新完整下载；中断后再次运行会续传。MMLU/GSM8K 的 Parquet 转换要求当前 Python 环境已安装 `pyarrow`，转换命令再次运行时会重新生成对应的 JSONL 文件。

## run.sh 的使用案例

命令格式：

```shell
bash run.sh [config]
```

使用默认基准配置训练：

```shell
bash run.sh
# 等价于：bash run.sh configs/default.yaml
```

使用 Natural Instructions 收敛基准配置训练：

```shell
bash run.sh configs/loss.yaml
```

使用从基准配置派生的任务配置训练：

```shell
cp configs/default.yaml configs/my_experiment.yaml
# 编辑 configs/my_experiment.yaml 后运行
bash run.sh configs/my_experiment.yaml
```

脚本会检查当前环境是否为 `skeleton_lora_fe`，设置项目内的 `hf-cache/`，启动 TensorBoard，然后执行 `main.py`。TensorBoard 默认监听 `6006` 端口，可通过环境变量修改：

```shell
TB_PORT=6007 bash run.sh configs/my_experiment.yaml
```

启动日志中的以下字段是本次实验标识：

```text
run_id=YYYY-MM-DD_HH-MM-SS
```

同一 `run_id` 的主要产物位于：

```text
logs/<run_id>/train.log
output/<run_id>/
├── checkpoints/
│   ├── round_01/
│   ├── round_XX/
│   └── final/
├── metrics/
└── tensorboard/
```

训练结束或脚本收到退出信号时，`run.sh` 会关闭由它启动的 TensorBoard。当前仓库的 `main.py` 按配置中的 `encryption` 段执行 CKKS 上传、公钥聚合和客户端解密；聚合仍采用客户端等权平均。

## 修复历史下载流量

历史 `round.csv` 的 `broadcast_size` 记录的是解密和 SVD 后的明文 adapter，
不能作为 CKKS 聚合 payload 的下载量。已有 final adapter 时可以跳过训练，按配置中的
真实 tensor 形状、CKKS 参数和分块规则快速重放下载线格式：

```shell
bash repair_download_metrics.sh configs/ckks-AB-1.yaml 3B-AB-01-MMLU
```

默认模式只对每个块执行一次决定密文层级和序列化大小的代表性运算。需要完整复制为
`K` 个客户端并执行全部同态乘法与聚合时，增加 `--exact`：

```shell
bash repair_download_metrics.sh \
  configs/ckks-AB-1.yaml \
  3B-AB-01-MMLU \
  --exact
```

结果写入：

```text
output/<run-id>/metrics/download_traffic_replay.csv
```

CSV 同时给出每客户端上传和下载的 bytes/MiB，以及按配置中的客户端数和轮数计算的
整次运行 GiB。换算使用二进制单位：`1 MiB = 2^20 bytes`，
`1 GiB = 2^30 bytes`。原 `round.csv` 不会被修改。

## evaluate.sh 的使用案例

命令格式：

```shell
bash evaluate.sh [config] <run-id> [train|mmlu|gsm8k] [adapter|base] [evaluation-config]
```

参数含义：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `config` | `configs/default.yaml` | 训练时使用的任务配置；模型和 LoRA 参数必须与 checkpoint 匹配 |
| `run-id` | 无 | 必填；训练产生的 `output/<run-id>` 目录名 |
| `target` | `train` | 训练分片、MMLU 或 GSM8K |
| `model-mode` | `adapter` | 评测聚合 LoRA，或仅评测原始底座模型 |
| `evaluation-config` | `configs/evaluation.yaml` | MMLU/GSM8K 的数据路径和生成参数 |

评估最终聚合的 LoRA adapter：

```shell
RUN_ID=2026-07-28_12-00-00

# 在训练配置重建出的客户端分片上计算 loss 和 perplexity
bash evaluate.sh configs/default.yaml "$RUN_ID" train

# MMLU 零样本四选一准确率
bash evaluate.sh configs/default.yaml "$RUN_ID" mmlu

# GSM8K 贪心生成的最终数值 exact match
bash evaluate.sh configs/default.yaml "$RUN_ID" gsm8k
```

评估原始底座模型，用于和 LoRA 结果比较：

```shell
bash evaluate.sh configs/default.yaml "$RUN_ID" mmlu base
bash evaluate.sh configs/default.yaml "$RUN_ID" gsm8k base
```

使用自定义专业评测配置：

```shell
cp configs/evaluation.yaml configs/evaluation_long.yaml
# 编辑数据路径、max_length 或 max_new_tokens 后运行
bash evaluate.sh configs/default.yaml "$RUN_ID" gsm8k adapter configs/evaluation_long.yaml
```

为兼容旧调用方式，第四、第五个参数也可以按“评测配置、模型模式”的顺序传入；新命令建议始终使用上面展示的“模型模式、评测配置”顺序。

`adapter` 模式加载：

```text
output/<run-id>/checkpoints/final/adapter_model.safetensors
```

`base` 模式不读取 LoRA checkpoint，只加载任务配置中指定的底座模型；此时 `run-id` 用于确定结果输出目录，便于将基线和 adapter 结果放在同一实验下。结果文件如下：

| target | adapter 模式 | base 模式 |
| --- | --- | --- |
| `train` | `metrics/eval.csv` | `metrics/eval_base.csv` |
| `mmlu` | `metrics/mmlu.csv` | `metrics/mmlu_base.csv` |
| `gsm8k` | `metrics/gsm8k.csv` | `metrics/gsm8k_base.csv` |

以上路径都相对于 `output/<run-id>/`。重复运行相同的 target 和 model-mode 会覆盖对应 CSV，运行前应按需备份。`train` 评估反映训练分布拟合程度，不等同于独立泛化能力；正式比较应至少同时报告 MMLU/GSM8K 的底座和 adapter 结果，并保持模型配置、评测配置及解码方式一致。

## 完整流程示例

下面以默认 MMLU 训练配置为例：

```shell
# 1. 下载默认配置所需的模型、训练集和 MMLU 测试集
bash download.sh llama3bv2 mmlu-train mmlu

# 2. 训练；从控制台的 run_id=... 记录实验标识
bash run.sh configs/default.yaml

# 3. 将这里替换为上一步实际输出的 run_id
RUN_ID=2026-07-28_12-00-00

# 4. 分别评估 LoRA 和原始底座模型
bash evaluate.sh configs/default.yaml "$RUN_ID" mmlu adapter
bash evaluate.sh configs/default.yaml "$RUN_ID" mmlu base
```
