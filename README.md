# SkeletonLoRA-Framework

预留加解密接口的多客户端联邦 LoRA 微调框架。`main.py` 显性编排整套联邦流程,加解密与聚合逻辑通过函数指针注入。

## 特性

- **黑盒边界清晰**:`client` / `server` 只是薄壳,业务钩子(`encrypt_fn` / `decrypt_fn` / `aggregate_fn`)以函数指针形式由 `main.py` 注入。
- **多客户端共享 base 权重**:通过 `peft` 的多 adapter 机制,单机可跑 N 个客户端而只加载一份基座模型。
- **完整遥测**:每 step 的 loss、每层 LoRA 的 grad_norm、加解密耗时、明文/密文/下发大小,全部落 CSV + TensorBoard。
- **端到端可跑通**:自带 dummy 模型 + dummy 数据集的冒烟配置,macOS CPU / MPS 秒级跑完,不依赖真实权重。

## 快速开始

```bash
# 创建并激活环境
conda env create -f environment.yaml
conda activate skeleton_lora_fe

# 一键冒烟(dummy 模型 + dummy 数据集,几秒钟跑完)
./run.sh

# 正式训练（需先将本地模型和数据集放到 configs/default.yaml 指定路径）
./run.sh configs/default.yaml

# 或直接跑冒烟配置
python main.py --config configs/smoke.yaml
```

产物在 `output/<时间戳>/`,人读日志在 `logs/<时间戳>/train.log`。

### SecLoRA SEL-2S

`SecLoRA_EndToEnd` 分支新增了独立的 `seclora/` 模块，不改动共享的
`client/`、`server/`、`runtime/` 和 `utils/`。在训练环境中执行：

```bash
conda activate skeleton_lora_fe
bash seclora/setup_autodl.sh
bash seclora/verify_autodl.sh
python main.py --config configs/seclora_end_to_end.yaml
```

配置默认使用 4 客户端、10% 选择性加密、`Sfp=22`、`Xmax=0.03125` 和
48 线程。原生后端复用全局 PC-MCFE 参数、聚合密钥和 BSGS 表，每轮返回
`C/M/S` 骨架并压回配置的 LoRA rank。协议边界与当前限制见
[seclora/README.md](./seclora/README.md)。

## 目录索引

| 目录 / 文件 | 用途 |
|---|---|
| `main.py` | 唯一入口,顶部三个 lambda 是全部业务逻辑,下面是循环骨架 |
| `client/`、`server/` | 黑盒瘦壳,构造时接受业务钩子 |
| `runtime/` | main 的物理拆分,包含 device / paths / peft / broadcast / train_step / loss / checkpoint |
| `utils/` | logger、timer、sizeof、svd、safetensors io、CSV / TensorBoard 写入器 |
| `models/`、`datasets/` | 按 `config.<kind>` 分派；支持本地 OpenLLaMA 与 Dolly 15k，`dummy.*` 用于冒烟 |
| `configs/` | `smoke.yaml`(冒烟)、`default.yaml`(真跑) |
| `seclora/` | SecLoRA 分支独立模块：Python 适配、PC-MCFE 原生后端和 CUR 解码 |
| `logs/`、`output/`、`hf-cache/` | 运行时产物,已加入 `.gitignore` |
| `CLAUDE.md` | 面向 AI 协作者的完整开发文档 |
| `overview.md` | 项目需求文档,与用户逐轮澄清的产物 |

## 自定义加解密

打开 `main.py`,顶部这一段就是全部业务逻辑:

```python
encrypt_fn = lambda state_dict, client_id, round_id: state_dict
decrypt_fn = lambda ciphertext, client_id, round_id: ciphertext
aggregate_fn = lambda plaintexts, rank: {...}           # 默认乘积 FedAvg 等权
secure_aggregate_fn = None                              # 可选联合密文聚合
```

- `encrypt_fn` 返回类型任意(bytes / dict / 自定义对象)
- `decrypt_fn` 拿到什么由你自己配对
- `client_id` / `round_id` 直接传给加解密钩子,便于构造客户端身份与轮次相关标签
- `aggregate_fn` 拿到的每份 `state_dict` key 与顺序一致(peft 决定)

细节看 [CLAUDE.md](./CLAUDE.md) 的「自定义加解密 / 聚合逻辑」一节,包含 lambda / def / 配置切换三种写法示例。

## 运行环境

- macOS / Linux
- Python 3.10+
- 依赖:`torch`、`transformers`、`peft`、`safetensors`、`tensorboard`、`pandas`、`pyyaml`
- 静态检查:`pyright` 0 error / `ruff` all checks passed(项目根有 `pyrightconfig.json`)

## 正式训练所需本地资源

- `models/open_llama_3b_v2/`：模型配置、权重及 tokenizer 文件。
- `datasets/databricks-dolly-15k/databricks-dolly-15k.jsonl`：Dolly 15k 原始数据。
- 上述大文件目录已加入 `.gitignore`，应在训练机器上单独下载。

`run.sh` 训练后自动追加 `evaluate.py` 仍待接入。

## 文档

- **[CLAUDE.md](./CLAUDE.md)** — 完整开发文档:目录职责、接口自由度、runtime/ 拆分、配置字段一览、观测产物说明、编码铁律。
- **[overview.md](./overview.md)** — 项目需求文档。
