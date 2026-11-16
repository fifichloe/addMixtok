# MixTok Long：无 density 的重建与生成代码

当前版本保留 **gain、white、CAM、图像级 U**，完全删除部署评分中的 density。生成使用本次重训后的 base-only Long router 和 20k-update MaskGIT。这是当前主实验代码；不是刚讨论但尚未执行的“fixed20 codec 再做动态微调”。当前结果见 [RESULTS_CURRENT_20261116.md](RESULTS_CURRENT_20261116.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `recon/` | 冻结特征、gain CNN、Long评分与阈值选区、完整50k重建评估 |
| `recon/training/` | 实际gain CNN和U回归训练公式；缓存路径由环境变量提供 |
| `gene/router.py`、`gene/router_bundle/` | 当前验收导出的生成router源码和必要科学配置 |
| `gene/scorer_training/` | 学生蒸馏、局部评分重建微调、预算分支/U微调代码 |
| `gene/maskgit/` | 选区缓存与当前四卡MaskGIT训练 |
| `gene/eval/` | 原两阶段采样、CFG参数、正式50k ADM gFID与独立FP64核对 |
| `configs/` | 方法参数、外部资产SHA256、原训练源码哈希和可填写的阶段配置 |
| `qualitative/reconstruction_200/` | 200张当前重建模型的原图、1D base、白色2D选区和精修对比图；另含作者选出的55张及联系表 |
| `qualitative/generation_200/` | 200张当前Long生成模型的1D base、白色2D选区和最终生成对比图，含原图与选区mask |

## 评分方式

```text
重建：score_i = gain_i + 0.5*white_i + 0.75*CAM_i + 0.9*U
生成：score_i = predicted_local_i + 0.9*predicted_U
K = clamp(count(score_i >= threshold), 48, 144)
mask = stable_top_K(score, K)
```

重建读取原图、base重建和两支特征；EfficientNet-B0冻结。gain CNN和U回归模型已经训练。生成router单独训练，只输入BF16的1D展开base feature，在FP32中评分；不输入原图、真实2D特征或B0。一次评分、一次最终选区，无迭代选区。当前是阈值计数加top-K，不是top-p累计概率选区。

## 依赖与权重

本目录发布当前代码与必要配置。公开基础依赖请单独准备：

- MixTok：<https://github.com/LCGLiChenge/MixTok>，使用其 `reconstruction/` 和 `generation/`。
- 原生TiTok与LlamaGen代码及其官方初始化权重。
- Halton-MaskGIT：使用基础MixTok生成代码中 `halton_upstream.py` 声明的固定commit和源码哈希。
- 环境版本见基础MixTok的 reconstruction环境文件与generation依赖文件。

**本次没有上传权重、数据、内部实验归档或机器路径元数据。** 新gain、U、router和MaskGIT权重需要单独准备；文件SHA256见 `configs/assets.json`。本目录不声称这些新权重已在原公开Hugging Face仓库发布。

环境变量：

```bash
export MIXTOK_REPO=/path/to/MixTok
export MIXTOK_TITOK_ROOT=/path/to/1d-tokenizer
export MIXTOK_LLAMAGEN_ROOT=/path/to/LlamaGen
export HALTON_UPSTREAM_ROOT=/path/to/Halton-MaskGIT
export MIXTOK_WEIGHTS=/path/to/current_long_weights
export MIXTOK_CODEC="$MIXTOK_WEIGHTS/top_p_ema_inference_v1_step247482.pt"
export MIXTOK_ADAPTER="$MIXTOK_WEIGHTS/adapter_step20000.pt"
export MIXTOK_HALTON_PRETRAINED=/path/to/ImageNet_256_base.pth
export MOTAR_ASSETS=/path/to/generation_assets
export MIXTOK_PAIRED_CODES=/path/to/paired_top_p_codes
export MIXTOK_MASKGIT_ROOT=/path/to/long_maskgit_runs
export MIXTOK_ROUTER_BUNDLE=/path/to/accepted_long_inference
export MIXTOK_TRAIN_GPUS=0,1,2,3
```

router bundle包含这里的 `model.py`、`config.json`与匹配的 `student_state.pt`。`MOTAR_ASSETS`保持原生成资源布局：`weights/tokenizer_titok_l32.bin`、`weights/generator_titok_l32.bin`、`fid/VIRTUAL_imagenet256_labeled.npz`、`fid/classify_image_graph_def.pb`。各代码/权重哈希必须与训练来源匹配；不应通过删除检查来迁移不匹配的资产。

## 重建评估

在addMixtok仓库根目录：

```bash
python -m mixtok_long_no_density.recon.evaluate \
  --images /path/to/validation_files.txt \
  --output /path/to/new_recon_output \
  --codec /path/to/mixtok_recon.pt \
  --titok /path/to/tokenizer_titok_l32.bin \
  --llamagen /path/to/vq_ds16_c2i.pt \
  --semantic /path/to/efficientnet_b0_rwightman-7f5810bc.pth \
  --gain /path/to/long_head/best.pt \
  --scales /path/to/input_scales.npz \
  --utility /path/to/semantic_rel.npz \
  --inception /path/to/weights-inception-2015-12-05-6726825d.pth \
  --batch 32
```

使用唯一50,000张验证清单和新的输出目录。默认阈值 `0.3803865313529968` 对应当前主codec；`--calibrate-mean-K`仅根据分数校准平均K96，在测质量之前冻结mask；`--fixed-K 96`是同排序固定预算。输出原始评分、K、mask、Inception特征、rFID和PSNR。codec默认为已发布主codec的SHA身份，不接受把fixed20诊断权重静默替换进来。

## 生成：缓存、训练、正式评估

paired codes是原中心裁剪与水平翻转两个视图，共2,562,334条；GT2D codes必须匹配当前codec。新选区缓存需重新生成：

```bash
python mixtok_long_no_density/gene/maskgit/cache_routes.py prepare \
  --output /path/to/new_routes --batch 64
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  mixtok_long_no_density/gene/maskgit/cache_routes.py extract \
  --output /path/to/new_routes --batch 64
python mixtok_long_no_density/gene/maskgit/cache_routes.py finalize \
  --output /path/to/new_routes --batch 64

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  mixtok_long_no_density/gene/maskgit/train.py \
  --kind long --routes /path/to/new_routes/long \
  --output "$MIXTOK_MASKGIT_ROOT/long_seed0" --seed 0
```

四卡各microbatch32、累积2次、global batch256、20k更新。官方Halton初始化与fresh AdamW；codec、embedding adapter、router冻结。实际训练公式和缓存算法保留。

现有生成权重和收据分别放在 `$MIXTOK_MASKGIT_ROOT/long_seed0/latest.pt` 与 `latest.json`：

```bash
python mixtok_long_no_density/gene/eval/evaluate.py \
  --kind long --cfg 3.0 --stage2-steps 8 --seed 4 \
  --output /path/to/new_cfg3_seed4_8p8
```

第一阶段8轮/CFG4.5；第二阶段由参数指定，默认8轮，即8+8。`--stage2-steps 16`是8+16。固定四个逻辑分片、batch8、原随机流，最多四张物理卡。五个完整50k、seed4的CFG扫描已完成，其中CFG3.0的gFID最低；其他采样seed的结果见当前结果文件。

现有checkpoint的科学身份逐项比较；安装目录字段在迁移比较中忽略。原配置SHA在精简配置中保留，模型/权重SHA仍按实际文件验证。训练代码中仅 `common.py` 外置路径绑定；旧checkpoint记录的该文件SHA与 `configs/original_training_sources.json` 核对，新训练记录当前文件SHA。其余原始训练和生成模块逐文件核对实际SHA。

## 评分器训练

- `recon/training/fit_gain.py`：从已缓存的base-vs-full2D像素/语义增益监督训练Long gain CNN；`--cache`指定四个标签分片，`--output`为新目录。设置 `MIXTOK_BASE_FEATURE_CACHE`，其中包含输入归一化初始化 `heads/rgb0.5_best.pt`。这些缓存/归一化资产需单独迁移。
- `recon/training/fit_utility.py`：图像级U的岭回归。设置 `MIXTOK_GAIN_CACHE`（含四个dense_gain分片）和 `MIXTOK_BASE_FEATURE_CACHE`。保持fit/held划分及目标定义。
- `gene/scorer_training/distill/train.py`：base-only学生20k蒸馏；读取现有全量教师标签cache。该cache的完整policy元数据由外部 `teacher_policies` 文件提供。
- `gene/scorer_training/spatial/train.py`：原Long局部评分重建微调，上限2000更新。
- `gene/scorer_training/budget/train.py`：复制预算特征分支和U训练，上限2000更新，保持局部排序冻结。最终导出使用训练held选择的快照，不自动取最后一步。

三个学生阶段均通过 `MIXTOK_SCORER_STAGE_CONFIG=/path/to/stage.json` 读取外部数据/父权重与阶段凭据；模板见 `configs/scorer_stage.example.json`。spatial/budget阶段还需设置 `MIXTOK_RECON_CODEC`、`MIXTOK_TITOK_WEIGHTS`、`MIXTOK_LLAMAGEN_WEIGHTS`、`MIXTOK_B0_WEIGHTS`；codec与第三方代码根目录沿用前述变量。验证扫描可设置 `MIXTOK_VALIDATION_MANIFEST`、`MIXTOK_INCEPTION_WEIGHTS`。模板中的父权重SHA、step和审计文件必须替换为真实上一步产物，不能把模板当完成凭据。

该代码交付保留训练核心和阶段顺序，未提供数据或阶段完成凭据，也不是空机器一键从零重训的调度器。迁移入口做了静态语法、算法及资产身份检查，**尚未在干净机器执行端到端模型验证**；此次上传没有额外启动训练或测试。
