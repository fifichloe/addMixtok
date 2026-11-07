# Archived K=256 generation-only control

**This is not the from-original-weights K=256 reconstruction experiment.** See the repository root README before using this code.

This repository contains the routing, training, and evaluation scripts used for the fixed **K=256** generation control. Every location in the 16×16 grid is selected. The 2D generator is trained from the official pretrained Halton initialization for 20,000 optimizer updates, with global batch size 256 and the same frozen MixTok tokenizer, adapter, decoder, and training objective used in the sparse generation experiments. The pretrained Halton weights are an initialization; the 2D generator is trained afresh for this control.

**Interpretation:** This is a MixTok dense control, **not an original LlamaGen checkpoint or a complete LlamaGen reproduction**. With all 256 positions replaced by 2D tokens, the 1D spatial context does not affect the 2D output or final RGB reconstruction, although the evaluation pipeline still runs the 1D stage for its diagnostic base result. The MixTok shared decoder was trained with sparse mixed grids. Therefore this endpoint does not isolate the effect of K alone and must not be presented as LlamaGen's best attainable quality or as a compute-matched baseline.

## Dependencies and data

Clone [MixTok](https://github.com/LCGLiChenge/MixTok) at commit `6623ccdc7a48688497f7b1f312548d1f14107e4d` and use its `generation` Python dependencies and asset instructions. Its `generation/experiments/continuous_halton_embedding_20260923/` modules provide the tokenizer, adapter, Halton model, training objective, cache format, and evaluator. The upstream Halton-MaskGIT checkout is pinned by MixTok's `halton_upstream.py` to commit `f61b0a1314717004dc7487531fd16a8bb71e1888`. The scripts verify upstream source hashes and checkpoint compatibility.

Provide the paired ImageNet training code cache in MixTok's `ContinuousRouteDataset` format (`titok_codes.npy`, `llamagen_codes.npy`, `labels.npy`, `meta.json`). The public `generation/scripts/extract_codes_top_p.py` creates it using the released tokenizer checkpoints. The route builder expects the two-view cache used in the experiment (2,562,334 rows). The MixTok generation assets must include the ADM FID reference and evaluator for 50,000-image evaluation.

Set paths to your local copies before running **any** script in `generation_only_control/`:

```bash
export MIXTOK_REPO=/path/to/MixTok
export MIXTOK_SOURCE_CACHE=/path/to/paired_top_p_codes
export MIXTOK_ASSETS=/path/to/MixTok/generation/assets
export MIXTOK_TOP_CHECKPOINT=/path/to/top_p_ema_inference_v1_step247482.pt
export MIXTOK_ADAPTER_CHECKPOINT=/path/to/adapter_step20000.pt
export MIXTOK_SCORER_CHECKPOINT=/path/to/scorer_top_p_1d_only_step20000.pt
export MIXTOK_HALTON_PRETRAINED=/path/to/ImageNet_256_base.pth
export MIXTOK_HALTON_UPSTREAM=/path/to/Halton-MaskGIT
```

The scorer checkpoint is checked for cache provenance but **is not used to select K=256 locations**. All routes are the full spatial grid. The route script retains the MixTok cache schema and its integrity checks.

## Train

From this repository root, build the static routes and launch six-GPU training:

```bash
python generation_only_control/cache_routes.py --policy dense --output /path/to/dense_routes
torchrun --standalone --nproc_per_node=6 generation_only_control/train.py \
  --policy dense --routes /path/to/dense_routes --output /path/to/dense_run --seed 0
```

Set `CUDA_VISIBLE_DEVICES` to the desired six GPUs. The 20,000-update run uses eight logical microbatches of 32 examples per update across six GPUs, with accumulation `[2,2,1,1,1,1]`. `--resume` resumes a checkpoint made by the **same** six-GPU script, paths, and source files. The configuration and source hashes are recorded in the checkpoint and checked on resume.

The reported run was interrupted after step 18,000 and resumed for the final 2,000 updates on four GPUs with `generation_only_control/train_four_gpu.py`. That handoff is included to document the actual execution, not required for a new uninterrupted six-GPU run. It expects an intact step-18,000 checkpoint and matching original paths:

```bash
torchrun --standalone --nproc_per_node=4 generation_only_control/train_four_gpu.py \
  --policy dense --routes /path/to/dense_routes --output /path/to/four_gpu_run \
  --source-checkpoint /path/to/six_gpu_run/latest.pt --seed 0
```

The four-GPU handoff retains global batch 256 and maps the first four saved rank RNG streams to the four new ranks. The original six-GPU step-18,000 checkpoint is no longer distributed here. Because this published `common.py` replaces machine-specific paths with environment variables, the original checkpoint cannot be resumed from this checkout without matching its saved configuration and source hashes.

## Evaluate

The evaluation scripts perform the same 50,000-image ADM gFID protocol with four GPU workers, seed 4, 8+8 sampling rounds and CFG 4.5/3.0:

```bash
python generation_only_control/eval_fixed_parallel.py \
  --policy dense --checkpoint /path/to/dense_run/latest.pt \
  --output /path/to/dense_eval_seed4 --n 50000 --seed 4 --workers 4 \
  --allowed-gpus 0,1,2,3 \
  --top-p-checkpoint "$MIXTOK_TOP_CHECKPOINT" \
  --adapter-checkpoint "$MIXTOK_ADAPTER_CHECKPOINT" \
  --scorer-checkpoint "$MIXTOK_SCORER_CHECKPOINT"
```

Use the four-GPU run's `latest.pt` when reproducing the reported interrupted run. The evaluation code checks the training policy, K, selected positions, checkpoint assets, and sample coverage.

## Reported result and scope

The interrupted run's final checkpoint SHA256 is `1ebe919828c59dc1685ced8927f6b119a089cfef82236ebd6c537d22a289b5a6`. The 50,000-image, single-seed gFID was **15.54999895** at effective 256 content tokens. The dense result is poor under this specific MixTok decoder and training protocol; it should not be interpreted as a general dense-token or LlamaGen result. Weights, training caches, ImageNet data, and FID assets are not included in this code repository.
