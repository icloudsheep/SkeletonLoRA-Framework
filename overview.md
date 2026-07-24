在这个文件夹中，需要实现一个预留加解密接口的多客户端的联邦学习低秩微调框架：

时间戳格式为：`YYYY-MM-DD_HH-MM-SS`

运行平台：Linux / macOS（不跨 Windows，可放心使用 symlink 等 POSIX 特性）。

以下需要在编写完成代码后决定：

- 数据集和模型权重合入

要求拥有至少以下文件夹：

- client：存放客户端的相关代码，多客户端的实现依靠在 main.py 中创建多个实例实现，客户端每次上传完整的 A、B 两个矩阵。客户端**暴露加密函数 `encrypt`**（明文输入为 A、B 两个矩阵的内存张量字典 `Dict[str, torch.Tensor]`，返回为任意类型，后续由用户自行处理）。`encrypt` 的函数体给恒等占位（`return state_dict`），用户可覆盖为真加密。main 只按入参出参调用。
- server：存放服务端的相关代码，服务端**暴露解密聚合二合一函数 `decrypt_aggregate`**（输入为多个客户端的任意格式加密信息，输出为处理完成后的 `Dict[str, torch.Tensor]`，可直接下发）。其内部固定流程为：
  1. 对每份密文调用用户可覆盖的 `decrypt`（默认恒等占位 `return ciphertext`）
  2. 对得到的多份明文张量字典调用用户可覆盖的 `aggregate`（默认对 `B @ A` 做等权平均后分解回 LoRA A/B）
  3. 对聚合后的结果调用 `utils` 中的 SVD 工具做后处理
  下行链路无需加密，直接作为下一轮客户端的训练起点。默认占位实现允许框架自身跑通端到端明文乘积聚合，用户接入真加密时覆盖对应函数即可。main 只按入参出参调用。
- models：存放所有需要的模型数据，要求使用本地模型权重而非现场下载。基座模型使用 `openlm-research/open_llama_3b_v2` 和 `openlm-research/open_llama_7b_v2`。
- datasets：存放所有需要的数据集，要求使用本地数据集而非现场下载，数据集需要按照客户端平均划分。数据集名称和路径（String），以及其分类和数据集平分方法（枚举）均存于 configs 中。**分片结果在 main 启动时一次性构建并缓存在内存**（避免每轮/每客户端重复读盘），`build_dataloader(config, client_id) -> DataLoader` 只从缓存中取对应 client_id 的分片。
- logs：**只存 Python logging 的文本日志**。按时间戳建子目录 `logs/<timestamp>/train.log`，与 output 物理隔离（人读的文本 vs 机器读的结构化产物）。
- configs：存放框架的所有配置信息，其余代码中禁止出现 hard code，改动客户端数量、修改 rank 等直接到此文件修改。config 文件格式为 yaml。main 只接受 `--config <path>` 参数，configs 为唯一真相源（不支持命令行覆盖字段）。默认值：
  - 随机种子：42（全局单 seed，派生用 `seed + client_id`）
  - 客户端数：4
  - LoRA rank：4
  - LoRA alpha：8
  - LoRA target_modules：`["q_proj", "k_proj", "v_proj", "o_proj"]`
  - 本地每轮 step 数：50
  - 总轮数：10
  - batch size：4
  - learning rate：2e-4
  - 优化器：AdamW，weight_decay=0.0
  - 基座模型：`open_llama_3b_v2`（快速跑通用）/ `open_llama_7b_v2`（正式实验）
  - 训练精度：float32（保持与 C++ 模块兼容）
  - tensorboard 端口：6006
  - LoRA 库：使用 huggingface `peft`
  - log_level：INFO
  - 数据集平分方法枚举初始值：`iid_uniform`（打乱后等分）
- utils：本框架所使用的所有工具，包括但不限于 SVD、存 bin、存 safetensors、`sizeof(obj)`、计时器封装等。
- hf-cache（非必要）：hf 的缓存地址。若启用，`run.sh` 中 `export HF_HOME=./hf-cache`。
- output：产物的保存总文件夹，里面的子文件夹需要自行创建，不可以新文件覆盖旧文件，即文件夹带时间戳。**只放机器读的结构化产物**（checkpoints / csv / tensorboard）。`logs/<timestamp>` 与 `output/<timestamp>` 共用同一个 `RUN_ID`（main 启动时计算一次）。约定结构：
  ```
  output/<timestamp>/
    checkpoints/round_XX/{A.safetensors, B.safetensors, adapter_model.safetensors}   (XX zero-padded 到 2 位，从 01 开始)
    checkpoints/final  (symlink → 最后一轮的 round_XX 目录，POSIX 软链)
    tensorboard/
    metrics/{step.csv, round.csv, grad_norm.csv}
  ```

还有以下若干文件：

- main.py：串行统筹所有训练，所有训练步骤在其中显性体现，无需实现断点续训。所有客户端共享同一份 base model 权重，通过 `peft` 的 adapter 切换机制在客户端之间切换。client、server 文件夹为黑盒（粒度为 step，main 负责记录日志），对外暴露只有入参出参。加解密与聚合逻辑由用户在 client/server 内部覆盖函数体，main 不出现具体加解密代码。dataloader 使用统一的 `build_dataloader(config, client_id) -> DataLoader`，按 configs 里的枚举分派。客户端切换粒度：串行执行，客户端 A 跑完本轮全部 step 再切 B。CLI 只有 `--config <path>`。
- evaluate.py：训练完成后，根据 output 中特定文件夹进行跑分，计入单独的 csv 文件。
- run.sh：快速启动整个流程。当前仅执行「后台起 tensorboard + 跑训练」，evaluate.py 完成后再追加评估步骤。脚本退出时 `trap` kill 掉 tensorboard 进程。tensorboard 端口冲突时不自动切换，让 tensorboard 自身报错退出（保持最简）。

还有如下铁需求，必须实现：

- 要有完善的 log 工具，关键步骤打日志，记录数据。文本日志落 `logs/<timestamp>/train.log`。INFO 及以上同时输出到控制台，DEBUG 仅落文件。
- 关键信息需要保存为 csv，全部落 `output/<timestamp>/metrics/`：
  - `step.csv`（粒度 step）：`round, client_id, step, loss`。`round` 从 1 起，`step` 为本轮内 0..K-1（全局累计可由 `round × K + step` 算出，不冗余存）。
  - `round.csv`（粒度 round，客户端级 + 服务端级字段混在一张表；服务端级字段在同一 round 的 N 行里冗余写 N 次，方便 join）：`round, client_id, encrypt_time, plaintext_size, ciphertext_size, aggregate_time, broadcast_size`
  - `grad_norm.csv`（长表，避免 step.csv 列数暴涨）：`round, client_id, step, layer_name, grad_norm`
  - 说明：每层 LoRA adapter 参数的梯度范数走 `grad_norm.csv`，每 step 每层一行；聚合时间包含解密时间；空间大小含原文 bin、密文 bin、下发 bin。
- 各种时间记录使用 `time.perf_counter()` 包住来记录，计时点在 main 中（client / server 为黑盒）。
- 各种空间大小使用 utils 提供的 `sizeof(obj)`，内部统一用 `pickle.dumps(obj)` 后取长度（原文 / 密文 / 下发都走同一度量，可比性优先）。
- 将上面的信息尽可能存入 tensorBoard（粒度分客户端，尽可能详细），在线实时查看。tensorboard 日志目录 = `output/<timestamp>/tensorboard/`。

编码铁律

- 明确边界
- 最简化代码逻辑，舍弃多余的健壮性改造，越简单越好
- 如果你面临多个实现路线，先问我，禁止自己私自决定
- 禁止任何幻觉
