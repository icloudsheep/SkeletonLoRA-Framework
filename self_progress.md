# 实验与实现进展

> 截至 2026-08-03。

## CKKS 精度与算法误差

- CKKS `global_scale` 当前设置为 $2^{40}$。
- 解密结果会转换为 `float32` 后再执行聚合和 SVD，因此最终数值表示受
  FP32 的 24-bit 有效精度限制；提高 CKKS scale 不能恢复转换时丢失的精度。
- Skeleton 路径包含 CUR 重建误差，客户端乘积等权平均后还可能产生最高
  $K r$ 的秩，最终截断回 LoRA rank $r$ 又会引入低秩近似误差。这些算法误差
  与 CKKS 近似误差需要分开报告。

## LoRA 乘积聚合

明文聚合原先会为每个客户端显式构造完整的 $B_iA_i$，然后对完整平均矩阵
执行 SVD。对于 3B/7B 模型，这会产生较大的临时矩阵和分解开销。

当前实现利用

$$
\frac{1}{K}\sum_{i=1}^{K}B_iA_i
=
\left[\frac{B_1}{\sqrt K},\ldots,\frac{B_K}{\sqrt K}\right]
\begin{bmatrix}
A_1/\sqrt K\\
\vdots\\
A_K/\sqrt K
\end{bmatrix},
$$

先对拼接因子执行 reduced QR，再只对阶数不超过 $Kr$ 的 core 矩阵执行
SVD，从而避免显式构造完整权重乘积。`tests/test_lora_product.py` 用稠密
截断 SVD 作为基准验证重构结果。

## LOSS 图表可视化

- Natural Instructions 的两组 1% partial-$AB$ 实验均完成 20 轮、2 个客户端、
  每客户端每轮 300 steps。
- 全局 token-weighted loss 中，CKKS+Skeleton 从 1.376627 降至 1.061452，
  SHE-LoRA (CKKS) 从 1.376627 降至 1.060114。
- 两条轮次曲线的平均绝对差为 0.001513，最大绝对差为 0.005670。共同波动
  主要来自每轮重新打乱数据和 Natural Instructions 的目标长度/难度差异。
- 论文主图应保留原始轮次曲线；完成多随机种子实验后再增加均值和标准差带。

## 论文表格与待补实验

- 通信量统一按 $1\,\mathrm{MB}=2^{20}$ bytes 计算，100 Mbps 时延直接由
  序列化字节数计算。
- 已整理 3B、$K=4$ 下 1%/10%/25% partial-$AB$ 的训练、MMLU、计算和通信结果。
- 已补充 0% partial-$AB$ 的 MMLU/Natural Instructions 配置，以及 7B、1%
  partial-$AB$ 的 Skeleton/非 Skeleton 配置。
- 待运行 0% 对照、7B 实验、固定验证集 loss 和多随机种子实验；在这些结果
  完成前，不填写 plaintext、7B、$K=10$ 或 mean $\pm$ standard deviation 单元格。
