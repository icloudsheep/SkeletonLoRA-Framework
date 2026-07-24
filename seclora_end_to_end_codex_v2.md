
**关键不适配**

1. ✅ **当前 SVD 处理对象不对**

已解决：默认聚合已改为先对每个客户端计算 \(B_iA_i\)，再对 \(\frac{1}{K}\sum_iB_iA_i\) 做 rank-r 分解，生成可广播的 global_B / global_A。

```
安全恢复聚合乘积或骨架
-> 除以 scale^2 和权重分母
-> 对聚合更新做 rank-r 分解
-> 生成新的 global_B / global_A
-> 广播给所有 adapter
```

当前实现聚合未缩放的 `B @ A`，PEFT 的 `lora_alpha / rank` scaling 仍由 adapter 前向路径应用，避免重复缩放。

1. ✅ **服务端接口违反 FE 聚合语义**

已解决：服务端已增加可选 `secure_aggregate_fn(ciphertexts, round_id)`，可联合处理全部客户端密文；未配置时仍走默认明文占位链路。

1. ✅ **加密接口缺少上下文**

已解决：当前 `encrypt_fn` / `decrypt_fn` 已显式接收：

- `client_id`
- `round_id`

1. ❌ **当前路径不适合每轮多层调用**

现在每次调用都会：

- 从磁盘读取一个 layer
- 重新初始化 coordinator 和密钥
- 构造完整 `3200×3200` 明文基线
- 计算真实秩和全矩阵误差
- 只处理单层

这些适合离线实验验证，但端到端训练需要持久加密会话、内存输入、逐层流式处理，并关闭完整明文 baseline。

无法在本仓库单独闭环：该项依赖 SecLoRA/mcl 侧运行路径改造。

**框架自身尚缺内容**

- ✅ causal-LM loss 已实现。
- ✅ `evaluate.py` 已实现，可输出 loss / perplexity 到 `eval.csv`。
- ✅ 默认 LoRA rank 已统一为 4。
- ✅ checkpoint 已额外输出包含 A/B 配对张量的 `adapter_model.safetensors`。
- ✅ bf16 训练输出与当前 C++ 仅支持 F32 safetensors 不兼容。
