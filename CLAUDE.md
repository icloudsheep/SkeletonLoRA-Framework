# SkeletonLoRA-Framework 开发说明

本文档面向后续代码协作者。描述以当前仓库代码为准，重点记录实验语义、模块边界和不可凭空假设的约束。

## 必读文档

执行或修改资源下载、训练启动、评估流程之前，必须先阅读 [instruction.md](./instruction.md)。该文档记录 `download.sh`、`run.sh` 和 `evaluate.sh` 的当前参数格式、默认值、资源路径、输出文件及完整调用案例。脚本实现和 YAML 配置仍是最终事实来源；文档与代码不一致时，应先核对实际实现并同步更正文档，不得自行补全不存在的命令、配置或运行行为。

## 项目目标

框架用于单机模拟多客户端联邦 LoRA 微调：只加载一份基座模型，为每个客户端注册独立 adapter；客户端完成本地训练后上传 LoRA 状态，服务端执行解密与聚合，再把聚合结果广播给下一轮的所有客户端。

`main.py` 显式保留联邦流程和加解密钩子。数据编码、训练指标、checkpoint、模型构建和评测分别由独立模块承担。

## 配置原则

`configs/default.yaml` 与 `configs/loss.yaml` 都是基准配置，后续任务必须根据实验目的派生，不应把任一文件解释为所有任务的固定配置。YAML 文件是参数的唯一真相源，下表仅记录当前基准快照。

| 文件 | 基准含义 | 当前关键参数 |
|---|---|---|
| `configs/default.yaml` | OpenLLaMA 3B、MMLU auxiliary train、常规联邦训练 | 4 clients、3 rounds、300 local steps、batch 4、LoRA rank 4、max length 512 |
| `configs/loss.yaml` | OpenLLaMA 3B、Super-NaturalInstructions、收敛实验 | 2 clients、20 rounds、300 local steps、batch 4、LoRA rank 4、max length 512、最多 20000 samples |
| `configs/smoke.yaml` | dummy 模型与数据的快速端到端回归 | 以 YAML 为准 |
| `configs/evaluation.yaml` | MMLU/GSM8K 本地路径和评测长度 | 以 YAML 为准 |

修改 `default.yaml` 或 `loss.yaml` 中定义基准实验语义的参数时，必须同步更新上表。普通派生任务只维护自己的配置，不反向修改基准快照。

新增实验配置时，应从最接近的基准派生，并在文件中显式固定：

- 数据集和样本上限；
- `num_clients`、`num_rounds`、`local_steps`；
- batch size、学习率、序列长度和 dtype；
- LoRA rank、alpha 和 target modules；
- 加密方案及其全部参数；
- 影响可复现性的 seed。

任务运行时传入明确的派生配置路径：

```bash
bash run.sh configs/<task>.yaml
```

## 当前目录与职责

```text
SkeletonLoRA-Framework/
├── main.py                       联邦训练编排与业务钩子
├── training_progress.py          当前生效的本地训练与训练指标实现
├── instruction.md                下载、训练与评估脚本使用速查
├── run.sh                        环境检查、TensorBoard 和训练启动
├── evaluate.py                   checkpoint 评估入口
├── evaluate.sh                   评估环境检查与参数转发
├── download.sh                   模型、训练集和测试集下载入口
├── client/                       客户端加密薄壳
├── server/                       服务端解密/聚合薄壳
├── runtime/
│   ├── device.py                 device 选择与随机种子
│   ├── paths.py                  run_id 和输出目录
│   ├── peft_ops.py               多 adapter 构建与参数遍历
│   ├── loss.py                   模型类型对应的 loss 和 batch 搬运
│   ├── broadcast.py              聚合状态广播
│   ├── checkpoint.py             round checkpoint 与 final 软链接
│   └── train_step.py             旧的拆分实现，当前 main.py 未调用
├── models/                       dummy/OpenLLaMA 模型注册与加载
├── datasets/                     数据加载、编码、分片和 DataLoader
├── evaluation/                   MMLU/GSM8K 专业评测
├── utils/                        日志、计时、指标、聚合和 IO
├── configs/                      基准及任务配置
└── tests/                        数据、评测和指标回归测试
```

修改训练行为时以 `main.py` 实际 import 的 `training_progress.py` 为准。不要只修改 `runtime/train_step.py` 后假定训练行为已经变化。

## 联邦状态流转

每个 round 的顺序由 `main.py` 固定：

```text
global_state
  -> broadcast_to_adapters（round 1 的 global_state 为 None，因此跳过）
  -> client_0 本地训练 -> 加密
  -> client_1 本地训练 -> 加密
  -> ...
  -> server.decrypt_aggregate
  -> save_round_checkpoint
  -> global_state = aggregated
```

第二轮的起点是第一轮服务端聚合所得 `global_state`，它会先写入所有客户端 adapter；之后各客户端在同一聚合起点上分别训练。客户端在一轮内按编号顺序执行，但彼此 adapter 独立，不应把前一个客户端的本地结果当成后一个客户端的起点。

当前训练还有以下重要语义：

- 基座模型权重在客户端之间共享，LoRA adapter 相互独立。
- 每个客户端每轮重新创建 AdamW，优化器动量等状态不跨 round 保留。
- DataLoader seed 为 `seed + round_id * num_clients + client_id`，因此客户端和 round 的打乱顺序不同，但在同一配置下可复现。
- `local_steps` 超过一个分片的 batch 数时，会重新创建该 DataLoader 的迭代器并继续取样。
- 默认聚合计算每个客户端的 `B @ A`，做客户端等权平均，再用 SVD 分解回指定 rank 的 A/B。
- 当前默认聚合不是按样本量或监督 token 数加权。比较非均衡分片实验时必须明确这一点。
- checkpoint 只保存聚合后的 adapter 参数，不保存 optimizer、DataLoader iterator 或 RNG 状态，因此不支持从 checkpoint 精确恢复训练现场。

## 加解密与聚合边界

业务钩子位于 `main.py` 顶部：

```python
encrypt_fn = lambda state_dict, client_id, round_id: state_dict
decrypt_fn = lambda ciphertext, client_id, round_id: ciphertext
aggregate_fn = lambda plaintexts, rank: aggregate_lora_products(plaintexts, rank=rank)
secure_aggregate_fn = None
```

接口含义：

| 钩子 | 输入 | 输出 |
|---|---|---|
| `encrypt_fn` | adapter state、client id、round id | 任意可传递的密文对象 |
| `decrypt_fn` | 密文对象、client id、round id | 可聚合的 adapter state |
| `aggregate_fn` | 全部客户端明文 state、目标 rank | 聚合后的 adapter state |
| `secure_aggregate_fn` | `(client_id, ciphertext)` 列表、round id | 聚合后的 adapter state |

`secure_aggregate_fn is None` 时，`Server` 逐客户端调用 `decrypt_fn`，再调用明文 `aggregate_fn`。设置密文聚合函数后，服务端不走逐客户端解密路径。

约束：

- 加密和解密的对象结构必须严格配对，框架不会推断密文格式。
- 解密或密文聚合返回的 key、shape 和 dtype 必须能写回 PEFT adapter。
- `client_id` 和 `round_id` 可用于密钥选择、AAD 或轮次隔离。
- `sizeof` 使用 pickle 序列化后的字节数作为明文、密文和下发大小的统一度量口径。
- 加密计时只覆盖 `client.encrypt(...)`；训练时间和服务端聚合时间分别统计。
- 引入有损近似、量化或同态加密时，应单独验证解密误差、聚合误差和最终 checkpoint 可加载性。

## 模型与数据集

### 模型

`models/__init__.py` 当前支持：

- `dummy`：单层 Linear，仅用于冒烟测试。
- `open_llama`：通过 `AutoModelForCausalLM.from_pretrained(..., local_files_only=True)` 从本地目录加载。

OpenLLaMA dtype 支持 `float32`、`float16` 和 `bfloat16`。模型加载不访问远端；缺失权重或 tokenizer 时应先运行 `download.sh`。

### 数据集注册

`datasets/__init__.py` 当前支持：

| `dataset.kind` | 实现 | 监督目标 |
|---|---|---|
| `dummy` | `datasets/dummy.py` | MSE 冒烟任务 |
| `dolly_15k` | `datasets/dolly.py` | 仅 response + EOS |
| `mmlu_train` | `datasets/mmlu_train.py` | 仅正确选项标签 + EOS |
| `natural_instructions` | `datasets/natural_instructions.py` | 仅 targets + EOS |

三个 causal-LM 数据集都会把 prompt 和 padding 对应的 label 设为 `-100`，因此 prompt 文本不参与 loss。这里不存在把参考答案放进输入后再同时监督 prompt 的行为；答案只作为需要预测的 completion 追加到序列尾部。

### Super-NaturalInstructions

数据来源为 `Muennighoff/natural-instructions`。加载器读取：

```text
datasets/natural-instructions/train/*.jsonl
```

单条样本必须包含：

```json
{"definition": "...", "inputs": "...", "targets": "..."}
```

`inputs` 可以为空；`definition` 和 `targets` 必须是非空字符串。提示模板由 `definition + inputs` 构成，仅 `targets + EOS` 参与 loss。

设置 `dataset.max_samples` 后，加载器先按 seed 打乱任务文件，再限制每个任务文件的最大样本数，避免截取结果集中于少数任务。加载阶段会一次性完成分词并把固定长度 tensor 保存在内存中；提高 `max_samples` 或 `max_length` 会直接增加主机内存占用。

当前所有真实训练集只实现 `iid_uniform` 分片。分片在训练开始时构建一次，各 round 复用同一分片，但 DataLoader 的 shuffle seed 随 round 变化。

## 下载入口

```bash
bash download.sh [TARGET ...]
```

| TARGET | 资源 | 输出位置 |
|---|---|---|
| `llama3bv2` | OpenLLaMA 3B v2 | `models/open_llama_3b_v2/` |
| `llama7bv2` | OpenLLaMA 7B v2 | `models/open_llama_7b_v2/` |
| `dolly` | Dolly 15k | `datasets/databricks-dolly-15k/` |
| `natural-instructions` | Super-NaturalInstructions train | `datasets/natural-instructions/train/` |
| `mmlu-train` | MMLU auxiliary train | `datasets/mmlu/mmlu_auxiliary_train.jsonl` |
| `gsm8k-train` | GSM8K main train | `datasets/gsm8k/train.jsonl` |
| `mmlu` | MMLU test | `evaluation/mmlu/mmlu_test.jsonl` |
| `gsm8k` | GSM8K main test | `evaluation/gsm8k/test.jsonl` |
| `all` | 以上全部资源 | 对应目录 |

无参数等价于 `all`。MMLU/GSM8K 下载包含 Parquet 转换，依赖当前 Python 环境中的 `pyarrow`。下载器支持重新执行和断点续传，不应通过删除已有目录来处理普通的重复下载。

## 训练指标定义

训练终端日志、CSV 和 TensorBoard 由 `training_progress.py` 与 `utils/metrics.py` 写入。

### `step.csv`

每个客户端每个本地 step 一行：

| 字段 | 定义 |
|---|---|
| `loss` | 当前 batch 的模型 loss；OpenLLaMA 为监督 token 上的 causal-LM cross entropy |
| `loss_moving_avg` | 当前客户端本轮最近 N 步 loss 的算术平均，N 由 `train.loss_moving_average_window` 控制，默认 20 |
| `perplexity` | OpenLLaMA 下为 `exp(min(loss, 20))`；dummy 为 NaN |
| `supervised_tokens` | 当前 batch 中 label 不等于 `-100` 的 token 数 |
| `loss_sum` | `loss * supervised_tokens`，用于构造 token 加权均值 |
| `global_grad_norm` | optimizer step 前，当前 adapter 全部参数梯度 L2 范数的合成值 |
| `learning_rate` | 当前 optimizer 参数组学习率 |
| `step_time` | 含前向、反向、optimizer step 和 CUDA 同步的耗时 |
| `supervised_tokens_per_second` | `supervised_tokens / step_time` |

### `client_round.csv`

每个客户端每轮一行：

- `mean_loss`：本轮 step loss 的算术平均。
- `token_weighted_mean_loss`：`sum(loss_sum) / sum(supervised_tokens)`，跨不同监督长度的 batch 更可比。
- `min_loss`、`max_loss`、`final_loss`：本轮离散 step 的极值与末步值。
- `final_moving_avg_loss`：本轮结束时的移动平均，比较收敛趋势时通常比 `final_loss` 稳定。
- `perplexity`：由 token 加权平均 loss 计算。
- `supervised_tokens`：本轮实际参与 loss 的 token 总数。
- `mean_step_time`、`train_time`：平均 step 耗时和本地训练总耗时。

### 其他 CSV

- `round.csv`：每轮每客户端的 `encrypt_time`、`plaintext_size`、`ciphertext_size`，以及本轮共享的 `aggregate_time`、`broadcast_size`。
- `grad_norm.csv`：每层 LoRA 参数的逐 step 梯度范数长表。

学术对比收敛速度时，应优先使用相同 token 预算下的 `token_weighted_mean_loss` 或 `final_moving_avg_loss`，同时报告数据集、有效监督 token、客户端数、round、local steps、batch size、学习率和 seed。不同任务的绝对 loss 不可直接比较。

## 输出与 checkpoint

`runtime/paths.py` 使用启动时间生成 `YYYY-MM-DD_HH-MM-SS` 格式的 `run_id`：

```text
logs/<run_id>/train.log
output/<run_id>/metrics/
output/<run_id>/tensorboard/
output/<run_id>/checkpoints/round_XX/
```

每轮 checkpoint 包含：

- `A.safetensors`：所有 LoRA A 参数；
- `B.safetensors`：所有 LoRA B 参数；
- `adapter_model.safetensors`：聚合后的完整 adapter state；
- `final`：指向最后一轮目录的 POSIX 软链接。

评估加载 `final/adapter_model.safetensors`。复制运行产物到不保留软链接的文件系统时，应确认 `final` 仍可解析，或直接指定/恢复最后一轮目录结构。

## 评估语义

```bash
bash evaluate.sh configs/<task>.yaml <run_id> <train|mmlu|gsm8k> [adapter|base]
```

默认 `adapter` 模式使用 `final/adapter_model.safetensors`，必须保证模型和 LoRA 配置与 checkpoint 匹配。`base` 模式直接评测配置中的底座模型，不创建 PEFT adapter，也不读取 checkpoint；此时 `run_id` 只负责把基线结果归入待对比实验的 `metrics/` 目录。

adapter 结果沿用 `eval.csv`、`mmlu.csv`、`gsm8k.csv`，base 结果写入对应的 `*_base.csv`，避免彼此覆盖。所有评测 CSV 都包含 `model_mode` 字段。

### `train`

重新构建训练配置中的客户端分片，在每个分片上计算平均 batch loss 和 perplexity，写入 `metrics/eval.csv`。这是训练分布上的拟合指标，不是独立泛化测试，也不能替代 MMLU/GSM8K。

### `mmlu`

使用零样本四选一提示。分别计算 `A/B/C/D` completion 的 log probability，选择分数最高者。输出 `metrics/mmlu.csv`，其中包含：

- `row_type=overall`：总正确数、总题数和准确率；
- `row_type=subject`：分学科正确率；
- `row_type=question`：逐题预测和正确性。

### `gsm8k`

使用确定性贪心生成，要求模型把最终数值放在 `####` 后。评分会规范化逗号、小数和负零，并对最终数值做 exact match。结果写入 `metrics/gsm8k.csv`。

MMLU 和 GSM8K 的路径、输入长度、生成长度位于 `configs/evaluation.yaml`。评测过程均显示进度和当前累计准确率。

## 修改与验证要求

修改代码前先确认实际调用路径，尤其避免混淆 `training_progress.py` 与未被 `main.py` 调用的 `runtime/train_step.py`。

数据集扩展至少需要：

1. 在 `datasets/` 新增加载和编码实现。
2. 在 `datasets/__init__.py` 注册新的 `dataset.kind`。
3. 明确 prompt 与监督 labels 的边界。
4. 添加字段校验、截断、mask、分片和确定性测试。
5. 更新 `download.sh`、基准派生配置和文档。

评测扩展至少需要：

1. 在 `evaluation/` 实现 `BenchmarkResult`。
2. 在 `evaluation/__init__.py` 注册 target。
3. 在 `evaluate.py` 和 `evaluate.sh` 增加可选值。
4. 在 `configs/evaluation.yaml` 增加路径和参数。
5. 添加数据格式、评分和汇总行测试。

提交前执行：

```bash
conda run -n skeleton_lora_fe python -m unittest discover -s tests -v
conda run -n skeleton_lora_fe ruff check .
bash -n download.sh run.sh evaluate.sh
git diff --check
```

对于真实模型训练，还需至少完成一次对应任务配置的小规模运行，并确认：

- 训练日志包含有限 loss 和非零监督 token；
- `step.csv`、`client_round.csv`、`round.csv` 和 `grad_norm.csv` 正常写入；
- checkpoint 能由 `evaluate.py` 加载；
- 加密模式下密文大小、耗时和解密后数值符合预期。

## 当前限制

- 真实数据集只实现 IID 均分，没有 Dirichlet 或按任务/学科的 non-IID 分片。
- 客户端顺序执行，不是并行训练。
- 默认聚合为客户端等权，不按数据量加权。
- optimizer 状态不跨 round 保存。
- checkpoint 不包含完整恢复训练所需的状态。
- `train` 评估复用训练分布；专业泛化结果应使用独立测试集。
- MMLU 当前为 zero-shot，GSM8K 当前为 greedy generation，不包含 few-shot 或多次采样设置。

## 协作约束

- 不猜测第三方数据字段、模型 API、加密格式或服务器目录；先读本地代码、样本或官方资料。
- 不改动与当前任务无关的文件，不覆盖用户未提交的修改。
- 配置是实验记录的一部分；新增任务应派生配置，不应反复改写两个基准配置来代表多个实验。
- 对训练语义有多种合理方案且会影响学术对比时，先说明差异再选择。
- 任何声称“训练改善”或“评测提升”的结论都必须对应可复现配置、run id 和输出指标。
