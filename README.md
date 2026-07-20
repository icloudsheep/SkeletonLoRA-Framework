# SkeletonLoRA-Framework

预留加解密接口的多客户端联邦 LoRA 微调框架。`main.py` 显性编排整套 FedAvg 流程,加解密与聚合逻辑通过三个函数指针注入,替换真加密只改 `main.py` 顶部三段。

## 特性

- **黑盒边界清晰**:`client` / `server` 只是薄壳,业务钩子(`encrypt_fn` / `decrypt_fn` / `aggregate_fn`)以函数指针形式由 `main.py` 注入。
- **多客户端共享 base 权重**:通过 `peft` 的多 adapter 机制,单机可跑 N 个客户端而只加载一份基座模型。
- **完整遥测**:每 step 的 loss、每层 LoRA 的 grad_norm、加解密耗时、明文/密文/下发大小,全部落 CSV + TensorBoard。
- **端到端可跑通**:自带 dummy 模型 + dummy 数据集的冒烟配置,macOS CPU / MPS 秒级跑完,不依赖真实权重。

## 快速开始

```bash
# 使用现成的 conda 环境
conda activate skeleton_lora_fe

# 一键冒烟(dummy 模型 + dummy 数据集,几秒钟跑完)
./run.sh

# 或直接跑
python main.py --config configs/smoke.yaml
```

产物在 `output/<时间戳>/`,人读日志在 `logs/<时间戳>/train.log`。

## 目录索引

| 目录 / 文件 | 用途 |
|---|---|
| `main.py` | 唯一入口,顶部三个 lambda 是全部业务逻辑,下面是循环骨架 |
| `client/`、`server/` | 黑盒瘦壳,构造时接受业务钩子 |
| `runtime/` | main 的物理拆分,包含 device / paths / peft / broadcast / train_step / loss / checkpoint |
| `utils/` | logger、timer、sizeof、svd、safetensors io、CSV / TensorBoard 写入器 |
| `models/`、`datasets/` | 按 `config.<kind>` 分派;`dummy.*` 是冒烟实现,`open_llama.py` 待接权重 |
| `configs/` | `smoke.yaml`(冒烟)、`default.yaml`(真跑) |
| `logs/`、`output/`、`hf-cache/` | 运行时产物,已加入 `.gitignore` |
| `CLAUDE.md` | 面向 AI 协作者的完整开发文档 |
| `overview.md` | 项目需求文档,与用户逐轮澄清的产物 |

## 自定义加解密

打开 `main.py`,顶部这一段就是全部业务逻辑:

```python
encrypt_fn = lambda state_dict: state_dict              # 默认恒等,替换为真加密
decrypt_fn = lambda ciphertext: ciphertext              # 与 encrypt_fn 配对
aggregate_fn = lambda plaintexts: {...FedAvg...}        # 默认 FedAvg 等权
```

- `encrypt_fn` 返回类型任意(bytes / dict / 自定义对象)
- `decrypt_fn` 拿到什么由你自己配对
- `aggregate_fn` 拿到的每份 `state_dict` key 与顺序一致(peft 决定)

细节看 [CLAUDE.md](./CLAUDE.md) 的「自定义加解密 / 聚合逻辑」一节,包含 lambda / def / 配置切换三种写法示例。

## 运行环境

- macOS / Linux
- Python 3.10+
- 依赖:`torch`、`transformers`、`peft`、`safetensors`、`tensorboard`、`pandas`、`pyyaml`
- 静态检查:`pyright` 0 error / `ruff` all checks passed(项目根有 `pyrightconfig.json`)

## 未完成 / 挂起

1. `models/open_llama.py` 真基座模型加载,等本地 `open_llama_3b_v2 / 7b_v2` 权重就位。
2. 真数据集接入,`configs/default.yaml` 的 `dataset.kind` 目前是 `placeholder`。
3. `runtime/loss.py` 的 `open_llama` 分支,接真数据集时同步补 causal-LM loss。
4. `evaluate.py` 训练完的跑分入口,`run.sh` 完成后追加评估步骤。

## 文档

- **[CLAUDE.md](./CLAUDE.md)** — 完整开发文档:目录职责、接口自由度、runtime/ 拆分、配置字段一览、观测产物说明、编码铁律。
- **[overview.md](./overview.md)** — 项目需求文档。
