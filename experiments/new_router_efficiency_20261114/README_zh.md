# A06：新路由效率、K 分布与码率

## Material Passport

- 范围：清单 A06。重建和生成均支持 `--kind short/long`，与最终主版本对应。
- 阶段：CPU 预算统计已完成；GPU 计时与 FLOPs 代码准备完成，等待主实验协调器分配 GPU。
- 未修改论文、正式训练实现、已有权重或 GitHub；不增加模型训练。
- 本目录没有启动 GPU 进程。现有 CFG 扫描优先，最多同时使用 4 张物理卡。

## 已有实际 50k 统计

| 场景 | 平均局部 K | 平均总 content tokens | 含位置开销固定宽度 bpp |
|---|---:|---:|---:|
| 新重建 long_c0.75_u0.9 | 96.00000 | 128.00000 | 0.03027344 |
| 新重建 short / u_semantic_rel_0.3 | 96.00000 | 128.00000 | 0.03027344 |
| 新生成 long，CFG=3 | 102.67920 | 134.67920 | 0.03170027 |
| 新生成 short，CFG=3 | 104.68342 | 136.68342 | 0.03212842 |

来源是完整 50k 推理保存的逐图 K 或完整 K 直方图。没有用第一张图片的 K 代替完整分布。

固定宽度公式：`(32*12 + 14*mean(K) + 256)/65536`。包含一个 256-bit 选区掩码；掩码人口计数给出 K，无需额外 K 字段。全局码本 4096、局部码本 16384。未包含类别标签、模型参数、文件封装，也未使用熵编码。该数值不是实际压缩文件大小。

CPU 统计、文件 SHA 与 CSV：`actual_budget_statistics_v2/`（包含两个重建版本；初版统计保留在 `actual_budget_statistics/`）。

重建 short 使用已接受的 `independent_onepass_goal105_20261112/accepted_policy.json`，固定阈值 **0.3232019245624542**；long 使用 goal100 的 accepted policy，固定阈值 **0.4321809709072113**。short 在共用 goal100 推理入口中的别名为 `short_c0.5_u0.3`，与原 `u_semantic_rel_0.3` 使用同一头、地图、U 和系数。启动时核对已接受权重 SHA、冻结阈值和系数；不重新按质量调阈值。

```bash
/home/heyefei/miniconda3/envs/MoT/bin/python \
  /home/heyefei/lichenge/paper/experiments/new_router_efficiency_20261114/budget_statistics.py \
  --output /home/heyefei/lichenge/paper/experiments/new_router_efficiency_20261114/actual_budget_statistics_v3
```

## GPU 测量协议

- 单 GPU、batch=1；10 次 warmup，50 个计时输入。
- 每个 CUDA 计时前后同步。分模块计时与独立端到端计时分两次执行，交替先后顺序。
- TF32 关闭、cuDNN benchmark 关闭；与当前生成质量评估精度一致。
- 不直接和旧 TF32 开启的 latency 表宣称同协议速度优劣；如需比较，baseline 要用相同协议重新测量。
- 输入已在 GPU；不包含磁盘、PIL/裁剪、模型加载、FID 或其他质量指标。
- 参数量记录实际调用的模块并去除共享参数重复计数；重建线性 U 系数另报。
- PyTorch `FlopCounterMode` 只统计已支持的卷积、矩阵乘法和 attention 等，按每 MAC=2 FLOPs。保存逐算子计数。
- normalization、非线性、排序、阈值、采样及其他未支持算子可能不计入 FLOPs；它们依然计入 latency，不能把零计数当成免费。
- FLOPs 默认测一个明确记录 K 的输入；50 次 latency 输入的平均 K、正式 50k 的分布分别报告。
- 默认包含动态和 frozen fixed-K=0/32/64/96/128/196/256。固定 K 是冻结同一模型后的推理诊断，不能称为各 K 独立训练模型。
- K=0 保留 dense 编码、路由、adapter 等实际路径开销，不伪装成优化后的 base-only bypass。
- 首次运行执行真实部署等价检查：重建分模块路径与 `OnePassEngine` 的 score/mask/K/RGB 完全一致；生成两 CNN 拆分结果与导出 student 的 score/mask/K 完全一致。此检查不是另加测试集或单元测试。
- 若 FLOPs 不支持某个算子而失败，先保存已完成 latency，再以 `latency_complete_FLOPs_incomplete` 标记，不能把它报为完整成功。

### 重建分模块

1D encoder+quantizer、空间展开、base RGB decoder、dense 2D encoder+quantizer+projection、B0 语义编码、原始特征/语义差图、白化余弦/静态密度、gain CNN、图像 U、分数合并、阈值计数/稳定排序、混合、final RGB decoder。

B0 将原图/base RGB 拼成 batch=2，和正式部署相同。旧 scorer 不运行。shared decoder 实际调用两次，参数只计一次。

### 生成分模块

1D generator（8 轮）、FP32 base 展开、用于新 scorer 的第二次 BF16 base 展开、local CNN、budget feature CNN、U head、阈值计数/稳定排序、adapter、2D Halton generator（默认 8 轮）、局部码本投影/混合、最终 RGB decoder。

保留正式部署中 FP32/BF16 两次展开，不把其中一条从成本里删掉。生成没有 B0、DINO、LPIPS、Inception。只计最终 RGB 解码，质量评估额外生成的 base 对照图不计入实际生成 latency。

## 待 GPU 授权后执行

以下命令仅为准备；当前没有执行。每条只使用一张卡，协调器决定实际卡号及运行顺序。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/heyefei/miniconda3/envs/MoT/bin/python \
  /home/heyefei/lichenge/paper/experiments/new_router_efficiency_20261114/profile_components.py \
  --mode reconstruction --kind long --warmup 10 --repeats 50 \
  --output /mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/new_router_efficiency_20261114/reconstruction_50repeats

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/heyefei/miniconda3/envs/MoT/bin/python \
  /home/heyefei/lichenge/paper/experiments/new_router_efficiency_20261114/profile_components.py \
  --mode generation --kind long --cfg 3.0 --stage2-steps 8 --warmup 10 --repeats 50 \
  --output /mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/new_router_efficiency_20261114/generation_long_cfg3_50repeats
```

最终主版本及 CFG 确定后再固定生成命令；脚本支持 `--kind short` 和显式 `--cfg`。输出目录要求不存在，避免覆盖失败/历史证据。先查 GPU 所属与协调器状态，再执行，不能和已占用 4 卡的 CFG 扫描重叠。

中央调度器模板与完成标记：`jobs.json`。实际 checkpoint/source SHA：`inputs_manifest.json`。该 manifest 是 CPU 文件复核，不代表 GPU profiling 已运行。固定 K=0/256 仍使用相同 256-bit mask 码率约定，不偷偷切换端点最优编码格式。

## 可复用旧测量的范围

`independent_onepass_goal100_20261113/cost_and_call_audit.json` 已实测新重建 B0/gain/U 路径，batch=1 warm E2E median=45.7035 ms，TF32 关闭，5 warmup/30 repeats。它有价值，但不满足此处 10/50、逐模块与 FLOPs 的完整新协议，故只引用为已有证据，不冒充本次已完成。
