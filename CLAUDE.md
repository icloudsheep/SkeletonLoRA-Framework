# SkeletonLoRA-Framework

一个预留加解密接口的多客户端联邦 LoRA 微调框架。`main.py` 显性编排整套联邦 LoRA 流程,`client` / `server` 是黑盒,只暴露必要的入参出参函数;加解密与聚合的具体逻辑由用户覆盖对应函数体实现。

## 目录结构

```
SkeletonLoRA-Framework/
├── main.py                 # 唯一入口,顶部三个 lambda 是全部业务钩子,下面是编排骨架
├── run.sh                  # 后台起 tensorboard + 跑训练;trap 清理
├── evaluate.py             # 训练完的跑分入口
├── overview.md             # 项目需求文档(与用户逐轮澄清的产物)
├── CLAUDE.md               # 本文档
│
├── client/
│   ├── __init__.py
│   └── client.py           # 瘦壳,构造时接受 encrypt_fn(由 main 传入)
├── server/
│   ├── __init__.py
│   └── server.py           # 瘦壳,构造时接受 decrypt_fn + aggregate_fn;SVD 已挪到 main
│
├── runtime/                # main 的物理拆分,不对外暴露语义
│   ├── __init__.py
│   ├── device.py           # pick_device / seed_all
│   ├── paths.py            # RUN_ID 与目录准备
│   ├── peft_ops.py         # build_peft_model / 参数遍历工具
│   ├── loss.py             # compute_loss / move_batch
│   ├── broadcast.py        # 上一轮聚合结果灌回各 adapter
│   ├── train_step.py       # 一个客户端一整轮本地训练 + 记录
│   └── checkpoint.py       # 存 round_XX checkpoint + final 软链
│
├── models/
│   ├── __init__.py         # 按 config.model.kind 分派
│   ├── dummy.py            # 冒烟用假模型(单层 Linear)
│   └── open_llama.py       # 真基座模型加载器,从本地路径读取权重
├── datasets/
│   ├── __init__.py         # 一次性 build_shards + 每轮 build_dataloader
│   ├── dummy.py            # 冒烟用随机张量数据集
│   └── dolly.py            # Dolly 15k 本地加载、causal-LM 编码与 IID 分片
├── utils/
│   ├── __init__.py         # 统一导出
│   ├── logger.py           # Python logging 初始化(INFO+ 落控制台,DEBUG 仅落文件)
│   ├── timer.py            # perf_timer() 上下文管理器
│   ├── sizeof.py           # sizeof(obj) = len(pickle.dumps(obj))
│   ├── lora_product.py     # aggregate_lora_products(state_dicts, rank)
│   ├── svd.py              # 通用二维张量 SVD 截断工具
│   ├── metrics.py          # CsvWriters + TbWriters
│   └── io.py               # yaml / safetensors / raw bytes IO
├── configs/
│   ├── smoke.yaml          # dummy 模型 + dummy 数据集,macOS CPU/MPS 秒级跑完
│   └── default.yaml        # 本地 OpenLLaMA 3B + Databricks Dolly 15k
│
├── logs/                   # <RUN_ID>/train.log,只放人读的文本日志
├── output/                 # <RUN_ID>/{checkpoints,metrics,tensorboard},结构化产物
└── hf-cache/               # HuggingFace 缓存(可选)
```

**目录约定**

- `logs/<RUN_ID>/` 与 `output/<RUN_ID>/` 共享同一个 `RUN_ID`(时间戳 `YYYY-MM-DD_HH-MM-SS`,main 启动时算一次)。
- `output/<RUN_ID>/checkpoints/round_XX/{A.safetensors, B.safetensors, adapter_model.safetensors}`,`final` 是指向最后一轮的 POSIX 软链。
- CSV 三张表全落在 `output/<RUN_ID>/metrics/`:`step.csv`(每 step 一行)、`round.csv`(每 round × client 一行,服务端字段冗余写)、`grad_norm.csv`(长表)。

## 快速开始

```bash
# 环境: skeleton_lora_fe conda 环境(python 3.10 + torch/transformers/peft/tensorboard/pandas/pyyaml)
conda activate skeleton_lora_fe

# 冒烟(dummy 模型 + dummy 数据集,几秒钟跑完)
./run.sh                                # 等价于 ./run.sh configs/smoke.yaml

# 正式(等 open_llama 权重与数据集接入后)
./run.sh configs/default.yaml
```

`run.sh` 会在后台起 tensorboard(默认 6006),脚本退出时 `trap` 杀掉它。

也可以不走脚本直接跑:

```bash
python main.py --config configs/smoke.yaml
```

`main.py` 的 CLI 只有 `--config`,configs 是唯一真相源,不支持命令行覆盖字段。

## 自定义加解密 / 聚合逻辑

框架的业务钩子**全部在 `main.py` 顶部**,以函数指针形式作为入参传入 `Client` / `Server`。默认实现是恒等 + 乘积 FedAvg,保证明文链路可跑通。

### 钩子的签名和位置

打开 `main.py`,顶部这一段就是全部业务逻辑:

```python
encrypt_fn = lambda state_dict, client_id, round_id: state_dict

decrypt_fn = lambda ciphertext, client_id, round_id: ciphertext

aggregate_fn = lambda plaintexts, rank: aggregate_lora_products(plaintexts, rank=rank)

secure_aggregate_fn = None
```

| 钩子 | 签名 | 语义 |
|---|---|---|
| `encrypt_fn` | `(Dict[str, Tensor], int, int) -> Any` | 客户端加密。后两个参数为 `client_id` / `round_id`。 |
| `decrypt_fn` | `(Any, int, int) -> Dict[str, Tensor]` | 服务端解密。后两个参数为 `client_id` / `round_id`。 |
| `aggregate_fn` | `(List[Dict[str, Tensor]], int) -> Dict[str, Tensor]` | 服务端明文聚合。默认对 `B @ A` 做等权平均,再分解回 LoRA A/B。 |
| `secure_aggregate_fn` | `Optional[(List[Tuple[int, Any]], int) -> Dict[str, Tensor]]` | 联合密文聚合。设为 `None` 时走默认解密后聚合。 |

流水线在 `main.py` 里显性写出来:

```
client 端:  state_dict + client_id + round_id  --encrypt_fn-->  ciphertext(Any)
                                                               │  上传到 server
server 端:  ciphertext + client_id + round_id  --decrypt_fn-->  plaintexts
                                      --aggregate_fn--> downstream {A, B}
                                                               │  下发回每个客户端 adapter
```

默认聚合会对客户端 LoRA 乘积做等权平均,再分解成下发所需的 A/B。

### 三种写法示例

**用 lambda(短逻辑)**

```python
encrypt_fn = lambda sd, client_id, round_id: {k: v.numpy().tobytes() for k, v in sd.items()}
decrypt_fn = lambda ct, client_id, round_id: {
    k: torch.frombuffer(v, dtype=torch.float32).view(shape[k]) for k, v in ct.items()
}
```

**用 def(长逻辑,推荐)**

```python
def encrypt_fn(state_dict, client_id, round_id):
    # 例: AES-GCM 对每个张量的 raw bytes 加密
    import io, os
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key = os.environ["FED_KEY"].encode()  # 32 bytes
    aead = AESGCM(key)
    out = {}
    for k, v in state_dict.items():
        nonce = os.urandom(12)
        buf = io.BytesIO()
        torch.save(v, buf)
        out[k] = {"nonce": nonce, "ct": aead.encrypt(nonce, buf.getvalue(), None)}
    return out

def decrypt_fn(ciphertext, client_id, round_id):
    import io
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key = os.environ["FED_KEY"].encode()
    aead = AESGCM(key)
    return {
        k: torch.load(io.BytesIO(aead.decrypt(v["nonce"], v["ct"], None)))
        for k, v in ciphertext.items()
    }
```

**用配置切换多套加密**

```python
if config.get("encryption", "none") == "aes":
    from my_crypto import aes_encrypt, aes_decrypt
    encrypt_fn, decrypt_fn = aes_encrypt, aes_decrypt
elif config.get("encryption") == "he":
    from my_crypto import he_encrypt, he_decrypt, he_aggregate
    encrypt_fn, decrypt_fn, aggregate_fn = he_encrypt, he_decrypt, he_aggregate
```

### 接口的自由度说明

- `encrypt_fn` 返回类型是 `Any`,可以是 `bytes` / `dict` / 自定义对象,`decrypt_fn` 拿到什么由自己配对。
- `client_id` / `round_id` 由框架传入,可直接用于身份相关密钥、标签或轮次隔离。
- 客户端与服务端之间传递的 `ciphertext` 是原对象,不落盘、不序列化;`main.py` 在 `sizeof(ciphertext)` 时用 `pickle.dumps` 度量大小,这是**大小度量的唯一口径**(明文、密文、下发都用它,可直接比较膨胀比)。
- **密文域聚合**:如果加密方案需要联合处理全部密文,实现 `secure_aggregate_fn(ciphertexts, round_id)` 并把它设为非 `None`。
- `aggregate_fn` 拿到的每份 `state_dict` 的 key 与顺序都一致(peft 决定),可以直接按 key 一一对应。

### 修改后如何验证

- **端到端能否跑通**:直接跑 `./run.sh` 冒烟。看 `logs/<RUN_ID>/train.log` 里的 `加密耗时` / `明文=xxB 密文=xxB` / `聚合完成` 三行是否符合预期,以及是否有 traceback。
- **加密确实生效了**:看 `output/<RUN_ID>/metrics/round.csv` 的 `ciphertext_size` 列,如果和 `plaintext_size` 完全相等且你不是恒等占位,说明加密没接上。
- **数值正确性**:看 `step.csv` 的 loss 曲线是否随 round 平稳下降。真加密下如果 loss 抖动明显,通常是加密引入了不可逆的精度损失或聚合逻辑对齐问题。

## runtime/ 里放了什么

`main.py` 保持极致精简,只保留「读 config → 建目录 / logger → 建模型 / 数据 → 循环骨架」。真正的实现细节全部拆到 `runtime/` 下:

| 文件 | 职责 |
|---|---|
| `device.py` | 挑 device(mps/cuda/cpu)、设全局种子 |
| `paths.py` | 计算 RUN_ID 与所有输出目录 |
| `peft_ops.py` | 挂 peft LoRA、注册多 adapter、遍历 adapter 参数 |
| `broadcast.py` | 把上一轮聚合结果灌回每个 adapter |
| `train_step.py` | 一个客户端一整轮本地训练 + loss / grad_norm 记录 |
| `loss.py` | 按 model kind 分派的 loss 计算与 batch 迁移 |
| `checkpoint.py` | 存 round_XX 的 A/B/adapter_model safetensors + final 软链 |

这些是 main 的物理拆分,不对外暴露语义。业务逻辑只集中在 main 顶部三个函数指针上。

## 配置字段说明(configs/*.yaml)

```yaml
seed: 42                                 # 全局种子,派生 seed 用 seed + client_id

federated:
  num_clients: 4                         # 客户端数
  num_rounds: 10                         # 总轮数
  local_steps: 50                        # 每客户端每轮本地 step 数

lora:
  rank: 4                                # LoRA rank
  alpha: 8
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

train:
  batch_size: 4
  learning_rate: 2.0e-4
  optimizer: adamw
  weight_decay: 0.0
  dtype: float32                         # 训练精度,保持与 C++ 模块兼容

model:
  kind: dummy | open_llama               # 分派到 models/__init__.py

dataset:
  kind: dummy | dolly_15k                # 分派到 datasets/__init__.py
  path: ./datasets/databricks-dolly-15k/databricks-dolly-15k.jsonl
  max_length: 512                        # Dolly causal-LM 序列长度
  split_method: iid_uniform              # 目前只实现了 iid 均分

logging:
  level: INFO                            # DEBUG 只落文件,INFO+ 同时落控制台

tensorboard:
  port: 6006
```

## 观测产物

**CSV**(下游用 pandas 直接读):

- `metrics/step.csv`: `round, client_id, step, loss` — 每 step 一行。
- `metrics/round.csv`: `round, client_id, encrypt_time, plaintext_size, ciphertext_size, aggregate_time, broadcast_size` — 每 round × client 一行,服务端级字段(`aggregate_time` / `broadcast_size`)在同一轮的 N 行里冗余写 N 次,方便 join。
- `metrics/grad_norm.csv`: `round, client_id, step, layer_name, grad_norm` — 长表。避免宽表列数爆炸。

**TensorBoard**(实时可看,`tensorboard --logdir output/<RUN>/tensorboard`):

- `server/`:全局聚合耗时、下发大小
- `client_0/` ... `client_{N-1}/`:每客户端的 loss、每层 grad_norm、加密耗时、明文/密文大小

**文本日志**:`logs/<RUN>/train.log`,人读用。

**checkpoint**:`output/<RUN>/checkpoints/round_XX/{A,B,adapter_model}.safetensors`,`final` 软链指向最后一轮。

## 未完成 / 挂起

1. **本地模型权重**:训练机器需单独准备 `openlm-research/open_llama_3b_v2`，正式实验可切换到 `open_llama_7b_v2`。
2. **真数据集文件**:训练机器需单独准备 `datasets/databricks-dolly-15k/databricks-dolly-15k.jsonl`；代码已实现本地加载、指令模板编码、labels 构造和 `iid_uniform` 分片。
3. **`run.sh` 追加 evaluate**:如需一键训练后评估,在 `run.sh` 末尾加 `python evaluate.py --config ... --run-id ...`。

## 编码铁律(项目自身遵循的)

- **明确边界**:client / server 只暴露入参出参;所有编排、计时、日志都在 main.py。
- **最简化**:不做过度防御性编程。
- **多路线先问**:遇到多种实现选择,先与用户确认再动。
- **禁止幻觉**:不熟悉的 API 先读文档 / 试跑,不猜测。
