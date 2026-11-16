# 新路由重建消融：A01/A02/A03/B01

## 状态与范围

当前为 CPU 准备，不代表实验完成。此目录不修改论文、不推送仓库、不启动 GPU。由总队列在 CFG 释放资源后统一调度，同时最多使用 4 张物理卡。

支持 `--kind short/long`，等 CFG 完成后确定正式主版本；默认不同时跑两套完整消融。编解码器保持原发布权重，所有曲线均为冻结权重的推理曲线，不是各 K 单独训练的最优曲线。

## 具体分组

- **A01 排序**：新 reconstruction scorer、新 generation scorer、旧 scorer、uniform、gradient entropy、random seeds 1/2/3。每张图使用完全相同的重建教师 K，隔离位置选择。生成 scorer 的教师 K 仅用于诊断，生成部署不需要教师。
- **A02 预算**：新动态、fixed96、严格匹配均值的近恒定预算、shuffled seeds 1/2/3、U=0 后重新标定阈值、相同局部分数配 top-p。按分数分布匹配平均 K≈96，不用质量选阈值。U 为图像级统一偏移，不改变排序，并在 softmax 中抵消。
- **A03 曲线**：fixed K=0/32/64/96/128/196/256；动态平均 K≈64/96/128，各配严格同均值近恒定和三个打乱预算对照。当前动态范围 48–144，K=32/196 为范围外固定预算诊断，K=0/256 为端点。
- **B01 组成**：分别移除 learned gain、直接 whitened cosine、直接 CAM、density 项，统一 fixed K=96。这里移除直接项不表示相关原始输入也被移除。
- **B01 完整无语义版本**：单独从随机初始化训练 RGB-only gain head；删除语义输入通道 774–777，目标只用 RGB endpoint gain；去掉 CAM、semantic U，统一 fixed K=96。部署时 `NoSemanticInputs` 不实例化、不运行 B0。该组同时移除语义输入与语义目标，不能把全部差异归因于直接 CAM。

## 训练与数据

完整无语义版本复用原 12,000 张训练图片的缓存，按原 source-index `%10` 分割为 10,843 fit / 1,157 held；从不训练在 ImageNet validation 上。缓存中保留的旧语义通道/目标在送入新模型和目标函数前删除；其余原始特征、RGB gain 标签仍来自同一冻结编解码器。

保持原 GainScorer 架构、fresh initialization、AdamW(lr=3e-4, weight_decay=.01)、batch64、clip5。short 对应 12 epochs 恒定 LR，long 对应 60 epochs cosine 到 1e-5。仅复用非语义输入归一化统计，不复用训练好的 head 权重。held 根据 RGB-only objective 选择 best，不根据验证 rFID 选择。

原始 gain head 训练记录中 short 约 37 秒、long 约 468 秒，仅作现有机器上的参考；新组实际耗时以日志为准。

## 执行方法

Python 为 `/home/heyefei/miniconda3/envs/MoT/bin/python`。

生成任务说明（**只写 JSON，不启动任务**）：

```bash
/home/heyefei/miniconda3/envs/MoT/bin/python /home/heyefei/lichenge/paper/experiments/new_router_reconstruction_ablations_20261114/job_spec.py --kind long --output /home/heyefei/lichenge/paper/experiments/new_router_reconstruction_ablations_20261114/long_jobs.json
```

JSON 内每项有完整命令、依赖和 GPU 数量。中央队列为单卡 worker 分配独立 `CUDA_VISIBLE_DEVICES`；CPU 阶段不占卡。执行依赖为：

1. 训练无语义轻量 head，同时可安排新/旧 scorer 分数提取。
2. 在相同官方 50k 验证路径提取无语义分数；每个提取任务只负责自己的 shard。
3. `build_policies.py` 合并完整分数，冻结阈值、top-p 参数、逐图 K 和最终 masks。该阶段不加载 Inception、不读取质量指标。
4. `evaluate_masks.py` 读取冻结 masks，四卡覆盖同一完整 50k；主组与无语义组分开，确保无语义组实际不运行 B0。
5. `merge_results.py` 合并全部 50k 原始 Inception features，用 FP64 统计计算一个全局 rFID，禁止平均 shard FID。

## 指标、保存与审计

- 每组完整唯一 50,000 图片，ADM center crop 256；BF16 codec，FP32 TorchMetrics-compatible Inception，TF32 关闭。
- 主重建 scorer 严格匹配原部署 FP16-rounded inputs → FP32 head；生成 scorer 使用 BF16 Fbase → FP32 head。
- 教师正式部署阈值直接核对原冻结验证分数清单：short `0.3232019245624542`（`independent_onepass_goal105_20261112/val_budget_frozen.json`），long `0.4321809709072113`（`independent_onepass_goal100_20261113/val_budget_frozen.json`）。不是早期训练集阈值。原阈值应重现验证平均 K=96；其他预算的重新标定是明确的推理消融。
- 记录 rFID、逐图 PSNR 平均、实际 K / total content tokens、保守 256-bit spatial mask 的 bpp：`(32*12+14*mean_K+256)/65536`。这是名义码率，不声称实际熵编码文件大小。
- 大文件全部写 NAS：`/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/new_router_reconstruction_ablations_20261114/{short,long}`。
- 保存 masks、K、路径顺序、source/asset SHA256、全部 Inception features、逐图分组依据和训练日志，便于重新核对预算与质量。
- 合并结果逐行区分 `codec_retrained=False` 与 `gain_head_independently_retrained`：普通控制组仅改变冻结模型的推理策略；无语义组确实独立训练轻量 head，编解码器仍不重训。新训练一遍 head 不等于进行了多训练 seed 稳定性实验。
- 预估主组 35 个重建配置；按已有 12.5k 单组约 199 秒、四卡同时覆盖全验证的记录，主质量阶段约 35–60 分钟，再加分数提取、CPU pooled statistics 与 no-semantic 组。该估计尚未用本组实际运行校准。

## 当前边界

不开展可选 B04 教师—学生交换误差分解；不启动用户在其他机器训练的 LlamaGen。完整无语义组不再含 scalar semantic U，比较限定为 fixed K=96 的位置质量，不把它当作新动态预算控制的单独消融。
