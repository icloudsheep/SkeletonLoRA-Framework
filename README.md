# SkeletonLoRA-Framework

面向联邦 LoRA 实验的多客户端训练框架。框架共享一份基座模型，为每个客户端维护独立 LoRA adapter，并在每轮本地训练后执行加密、服务端聚合、广播和 checkpoint 保存。

## 核心能力

- 多客户端共享基座权重，降低单机联邦实验的显存开销。
- CKKS 客户端上传、服务端公钥聚合和客户端解密采用独立协议阶段。
- 支持 OpenLLaMA 3B/7B v2、本地训练数据和离线评测。
- 支持 Dolly 15k、MMLU auxiliary train、Super-NaturalInstructions 和 dummy 冒烟数据。
- 记录逐步 loss、移动平均 loss、困惑度、监督 token、梯度范数、吞吐量、训练耗时和通信开销。
- 支持训练分片 loss、MMLU 和 GSM8K 三种评估目标。

## 配置基准

`configs/default.yaml` 和 `configs/loss.yaml` 都是后续实验的基准配置，不代表所有任务的最终参数。YAML 文件是参数的唯一真相源，下表仅记录当前基准快照：

| 配置 | 基准用途 | 当前关键参数 |
|---|---|---|
| `configs/default.yaml` | OpenLLaMA 3B + MMLU auxiliary train 的常规联邦训练基准 | 4 clients、3 rounds、300 local steps、batch 4、LoRA rank 4、max length 512 |
| `configs/loss.yaml` | OpenLLaMA 3B + Super-NaturalInstructions 的收敛基准 | 2 clients、20 rounds、300 local steps、batch 4、LoRA rank 4、max length 512、最多 20000 samples |
| `configs/smoke.yaml` | dummy 模型和数据的快速回归测试 | 以 YAML 为准 |

修改 `default.yaml` 或 `loss.yaml` 的基准实验语义时，应同步更新上表；普通派生任务只需维护自己的 YAML，无需反向修改基准说明。新任务应从最接近的基准复制并派生独立配置，明确记录客户端数、round 数、local steps、数据量、序列长度、LoRA rank 和加密参数。运行时向脚本显式传入任务配置：

```bash
bash run.sh configs/<task>.yaml
```

## 环境准备

```bash
conda env create -f environment.yaml
conda activate skeleton_lora_fe
```

`run.sh` 和 `evaluate.sh` 要求 Conda 环境名为 `skeleton_lora_fe`。脚本会把 Hugging Face 缓存设置到项目内的 `hf-cache/`，Linux 训练环境还会优先使用 Conda 中的 `libstdc++.so.6`。

## 下载资源

不带参数会下载全部模型、训练集和测试集：

```bash
bash download.sh
```

也可以按需下载：

```bash
bash download.sh llama3bv2
bash download.sh llama7bv2
bash download.sh mmlu-train
bash download.sh gsm8k-train
bash download.sh natural-instructions
bash download.sh mmlu gsm8k
```

主要落盘路径：

| 资源 | 本地路径 |
|---|---|
| OpenLLaMA 3B v2 | `models/open_llama_3b_v2/` |
| OpenLLaMA 7B v2 | `models/open_llama_7b_v2/` |
| Dolly 15k | `datasets/databricks-dolly-15k/` |
| MMLU auxiliary train | `datasets/mmlu/mmlu_auxiliary_train.jsonl` |
| GSM8K train | `datasets/gsm8k/train.jsonl` |
| Super-NaturalInstructions | `datasets/natural-instructions/train/*.jsonl` |
| MMLU test | `evaluation/mmlu/mmlu_test.jsonl` |
| GSM8K test | `evaluation/gsm8k/test.jsonl` |

MMLU 和 GSM8K 的 Parquet 文件会由 `utils/convert_hf_parquet.py` 转为本地 JSONL，因此执行相关下载前，当前 Python 环境需要安装 `pyarrow`。大文件目录均已加入 `.gitignore`。

## 训练

先基于基准配置生成任务配置，再显式运行：

```bash
bash run.sh configs/<task>.yaml
```

也可以直接调用入口：

```bash
python main.py --config configs/<task>.yaml
```

每轮流程为：

```text
上一轮聚合权重广播（首轮跳过）
  -> 各客户端依次本地训练
  -> 客户端加密上传
  -> 服务端使用公钥聚合并下发 payload
  -> 客户端解密、Skeleton 重构和 SVD
  -> 保存 round checkpoint
```

第二轮及以后均从上一轮聚合结果开始。当前默认聚合对客户端等权，不按客户端样本数加权；每个客户端每轮都会重新创建 AdamW，因此优化器状态不会跨客户端或 round 保留。

Natural Instructions 收敛基准示例：

```bash
bash download.sh natural-instructions
bash run.sh configs/loss.yaml
```

加载器只读取 `train/*.jsonl`，使用 `definition + inputs` 构造提示词，仅让 `targets + EOS` 参与 causal-LM loss。空 `targets` 样本会被跳过且不占用 `dataset.max_samples` 配额；其他字段结构错误仍会终止加载。`dataset.max_samples` 可限制有效样本数，并在任务文件之间做确定性均衡抽样。

## 训练产物

一次运行会生成同一 `run_id` 下的日志与结构化产物：

```text
logs/<run_id>/train.log
output/<run_id>/
├── checkpoints/
│   ├── round_01/
│   │   ├── A.safetensors
│   │   ├── B.safetensors
│   │   └── adapter_model.safetensors
│   └── final -> round_XX
├── metrics/
│   ├── step.csv
│   ├── client_round.csv
│   ├── round.csv
│   └── grad_norm.csv
└── tensorboard/
```

- `step.csv`：逐 step loss、移动平均、困惑度、监督 token、全局梯度范数、学习率、耗时和吞吐量。
- `client_round.csv`：每个客户端每轮的普通平均 loss、token 加权平均 loss、最小/最大/最终 loss、最终移动平均和训练耗时。
- `round.csv`：加密/聚合/解密耗时、上传 payload、聚合 payload 下载量，以及解密后的 adapter 大小。
- `grad_norm.csv`：每个 LoRA 参数张量的逐 step 梯度范数。

`A.safetensors` 和 `B.safetensors` 分别保存聚合后的 LoRA A/B；`adapter_model.safetensors` 保存可直接加载的完整 adapter 状态；`final` 指向最后一轮 checkpoint。

## 评估

评估不会由训练脚本自动触发。完成训练后，使用训练时的任务配置和对应 `run_id`：

```bash
# 在训练分片上计算 loss/perplexity
bash evaluate.sh configs/<task>.yaml <run_id> train

# MMLU 零样本四选一准确率
bash evaluate.sh configs/<task>.yaml <run_id> mmlu

# GSM8K 贪心生成后的最终数值 exact match
bash evaluate.sh configs/<task>.yaml <run_id> gsm8k

# 不加载 LoRA，直接评测原始底座模型
bash evaluate.sh configs/<task>.yaml <run_id> mmlu base
bash evaluate.sh configs/<task>.yaml <run_id> gsm8k base
```

默认 `model-mode` 为 `adapter`，会加载该 run 的 `final/adapter_model.safetensors`。`base` 模式只加载配置中的本地底座模型，不读取 checkpoint；`run_id` 仅用于确定结果目录，从而把同一实验的底座分数和 LoRA 分数放在一起比较。原生结果分别写入 `metrics/eval_base.csv`、`metrics/mmlu_base.csv` 或 `metrics/gsm8k_base.csv`，CSV 的 `model_mode` 列会明确标记模型模式。专业评测的数据路径和生成长度由 `configs/evaluation.yaml` 控制。MMLU CSV 包含总正确率、分学科正确率和逐题结果。

## 加解密接口

CKKS 聚合接口按信任边界拆分：

```python
encrypt(state_dict, client_id, round_id) -> client_payload
aggregate_encrypted(ciphertexts, round_id) -> aggregate_payload
decrypt_aggregate(aggregate_payload, round_id) -> state_dict
```

服务端聚合阶段只使用 public CKKS context。聚合 payload 在下发大小完成统计后，才由客户端侧解密、重构和 SVD。历史实验的下载量可用 `repair_download_metrics.sh` 从 final adapter 免训练重放。具体约束见 [CLAUDE.md](./CLAUDE.md)。

## 验证

```bash
conda run -n skeleton_lora_fe python -m unittest discover -s tests -v
conda run -n skeleton_lora_fe ruff check .
bash -n download.sh run.sh evaluate.sh repair_download_metrics.sh
```

## 文档

- [instruction.md](./instruction.md)：`download.sh`、`run.sh` 和 `evaluate.sh` 的参数、使用案例、输出路径与完整实验流程。执行下载、训练或评估前应先阅读此文档。
- [CLAUDE.md](./CLAUDE.md)：代码边界、配置语义、训练状态、指标定义、评估方法和扩展约束。
- [overview.md](./overview.md)：项目需求与早期设计背景。
