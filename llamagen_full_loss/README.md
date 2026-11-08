# Pure LlamaGen VQ reconstruction control: H200 runbook

This experiment fine-tunes the **original LlamaGen VQ tokenizer** from its `vq_ds16_c2i.pt` EMA weights, using its full 16×16 grid of 256 visual tokens. It trains the VQ encoder, quantization convolution, post-quantization convolution, and decoder with the MixTok fixed-96 reconstruction loss recipe. The codebook is frozen. There is **no K selection** in this trainer: the TiTok model, MixTok adapter, grid scorer, sparse routing, and generation pipeline are not used. The TiTok **source tree**, but not its weights, is needed only to import the projected discriminator implementation.

The loss weights are L1 ×2, LPIPS ×0.6, DINOv2 ×0.5, CLIP ×0.5, projected ConvNeXt GAN ×0.12, discriminator feature matching ×0.5, and LeCam ×0.001. GAN training starts at epoch 1.5. `loss_config.yaml` is a copy of the fixed-96 configuration; router settings in that file are unused by this trainer.

## H200 setup: follow these steps in order

The Codex agent on H200 should find the actual workspace and ImageNet paths and replace the three placeholders below. **Do not run with placeholder paths.** Reuse verified existing repos, weights, and environments if present. The ImageNet training directory must contain class subdirectories (`train/<class>/*`); validation images may be nested recursively. Use `tmux new -s llamagen-full-loss` or the machine's job scheduler before the long download and training commands.

```bash
export MIXTOK_H200_WORK=/absolute/path/to/writable/workspace
export MIXTOK_IMAGENET_TRAIN=/absolute/path/to/ImageNet/train
export MIXTOK_IMAGENET_VAL=/absolute/path/to/ImageNet/validation
export MIXTOK_H200_RUN="$MIXTOK_H200_WORK/results/llamagen_full_loss"
export TORCH_HOME="$MIXTOK_H200_WORK/.cache/torch"
mkdir -p "$MIXTOK_H200_WORK" "$TORCH_HOME"
cd "$MIXTOK_H200_WORK"
```

Clone the sources into this workspace. For an existing tree, check its revision before reuse; use a fresh clone if changing revisions would disturb local work. `MixTok/reconstruction` must be at the pinned public commit below. Do **not** use the older `reconstruction_k256/` or `generation_only_control/` commands for this run.

```bash
git clone https://github.com/fifichloe/addMixtok.git addMixtok
git clone https://github.com/LCGLiChenge/MixTok.git MixTok
git -C MixTok checkout 6623ccdc7a48688497f7b1f312548d1f14107e4d
git clone https://github.com/bytedance/1d-tokenizer.git 1d-tokenizer
git -C 1d-tokenizer checkout 942a96fbdd873780179d1b78d5462911528bf8c8
git clone https://github.com/FoundationVision/LlamaGen.git LlamaGen
git -C LlamaGen checkout ce98ec41803a74a90ce68c40ababa9eaeffeb4ec
```

Create the MixTok reconstruction environment once, or activate it if already installed. The public-weight helper downloads and verifies the original LlamaGen VQ checkpoint, local DINOv2 source and weights, and LlamaGen's LPIPS/VGG weights. TiTok weights are intentionally omitted because this trainer only imports its discriminator source. If H200 cannot download, supply these verified files at the same paths before proceeding.

```bash
conda env create -n mixtok-reconstruction -f "$MIXTOK_H200_WORK/MixTok/reconstruction/environment.yml"
conda activate mixtok-reconstruction
python "$MIXTOK_H200_WORK/MixTok/reconstruction/download_public_weights.py" \
  --project-root "$MIXTOK_H200_WORK" \
  --torch-cache-root "$TORCH_HOME" \
  --only llamagen_vq_ds16_c2i dinov2_repo dinov2_vits14 lpips_vgg
```

Preload the other frozen backbones **once** before eight workers start, avoiding concurrent first-time downloads:

```bash
mkdir -p "$MIXTOK_H200_WORK/.cache/open_clip"
python - <<'PY'
import os
from pathlib import Path
import open_clip
from torchvision import models

cache = Path(os.environ['MIXTOK_H200_WORK']) / '.cache/open_clip'
cfg = open_clip.get_pretrained_cfg('ViT-B-32', 'openai')
open_clip.pretrained.download_pretrained(cfg, prefer_hf_hub=False, cache_dir=str(cache))
models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
print('CLIP and ConvNeXt weights are ready')
PY
```

Check that the required paths and eight matching, idle H200 GPUs are available. The launcher accepts only GPUs with at most 1 GiB already in use. The validation evaluation expects 50,000 images. For the default **online** W&B run, complete `wandb login --relogin --verify` in the H200 account before allocating GPUs; the project is `mixtok-rebuttal`. Keep the API key out of commands, README files, and logs.

```bash
set -e
test -d "$MIXTOK_IMAGENET_TRAIN" && test -d "$MIXTOK_IMAGENET_VAL"
test -f "$MIXTOK_H200_WORK/MixTok/reconstruction/train_titok_llamagen_recon.py"
test -f "$MIXTOK_H200_WORK/1d-tokenizer/modeling/modules/discriminator.py"
test -f "$MIXTOK_H200_WORK/LlamaGen/pretrained_models/vq_ds16_c2i.pt"
test -f "$TORCH_HOME/hub/facebookresearch_dinov2_main/hubconf.py"
test -f "$TORCH_HOME/hub/checkpoints/dinov2_vits14_pretrain.pth"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
```

## Train on eight H200 GPUs

Run from any working directory. The launcher selects eight idle GPUs by default, probes per-GPU batch sizes with **all losses including GAN enabled**, and chooses the largest measured batch whose peak memory falls within **92%–95%** of total GPU memory. If a batch is too large, the failed probe is recorded and the search continues. If no integer batch falls in this interval, training does not start and the probe logs explain the gap. Report the neighboring measured batches to the user; do not silently relax the memory interval or reduce the GPU count. The selected batch, memory measurements, and GPU IDs are written to `batch_selection.json` before training starts.

```bash
python "$MIXTOK_H200_WORK/addMixtok/llamagen_full_loss/run.py" \
  --data-root "$MIXTOK_IMAGENET_TRAIN" \
  --output "$MIXTOK_H200_RUN" \
  --source-root "$MIXTOK_H200_WORK/MixTok/reconstruction" \
  --titok-root "$MIXTOK_H200_WORK/1d-tokenizer" \
  --llamagen-root "$MIXTOK_H200_WORK/LlamaGen" \
  --llamagen-ckpt "$MIXTOK_H200_WORK/LlamaGen/pretrained_models/vq_ds16_c2i.pt" \
  --dino-repo "$TORCH_HOME/hub/facebookresearch_dinov2_main" \
  --clip-cache-dir "$MIXTOK_H200_WORK/.cache/open_clip" \
  --gpus 8 --epochs 20 --min-memory 0.92 --max-memory 0.95 \
  --wandb-mode online
```

The defaults are eight GPUs, 20 epochs, gradient accumulation 1, a 92%–95% memory band, and online W&B project `mixtok-rebuttal`. Global batch size is determined by the H200 probe and recorded in the run configuration and W&B. Add `--gpu-ids 0,1,2,3,4,5,6,7` only if specific physical GPUs must be used. Use `--wandb-mode offline` only if online W&B is unavailable and the user agrees. Run the full ImageNet experiment: do not pass the short smoke-test options `--limit-samples` or `--train-max-steps`. `run.py` stays attached until training finishes; detach from `tmux` with `Ctrl-B`, then `D`.

`latest.pt` is atomically replaced after **every epoch**. The only separately retained intermediate checkpoint is `epoch_10.pt`. No `epoch_20.pt` is written; `latest.pt` holds epoch 20 after completion. Re-running the same command with `latest.pt` present resumes from the next epoch using the saved batch selection. Keep the same GPU count and accumulation on resume. `wandb_run_id.txt` keeps a resumed training job in the same W&B run.

The training log is `train.log`, and full-loss probe logs are under `probes/`. The probe forces GAN on from the first update to expose the higher-memory training phase. The actual 20-epoch training retains the scheduled GAN start at epoch 1.5. Each batch candidate initializes the distributed models and runs 12 full-loss steps, so probing can take time.

W&B logs the mean across GPUs and accumulation microbatches every 100 optimizer steps (and for steps 1–2): total generator and discriminator losses; raw and weighted image, LPIPS, DINOv2, CLIP, adversarial, feature-matching, and LeCam terms; total and separate real/fake discriminator hinge losses; discriminator real/fake logits and their LeCam moving averages; generator gradient norm before clipping; all learning rates; GAN/warmup state; epoch/progress, throughput, elapsed time, and rank-0 GPU memory. The total hinge loss is half the sum of its real and fake parts. Its run configuration records the selected batch, global batch, source revision, loss weights, and probe peak-memory fraction. Probe candidates themselves are recorded locally in `batch_selection.json`; probe W&B runs are disabled. Epoch completion and duration are logged separately. This training run does not calculate rFID/PSNR during training; the final evaluation below logs them as a second run in the same W&B group.

Monitor from another shell:

```bash
cat "$MIXTOK_H200_RUN/batch_selection.json"
tail -f "$MIXTOK_H200_RUN/train.log"
ls -lh "$MIXTOK_H200_RUN"/*.pt
```

## Evaluate

After training, evaluate the EMA model on all 50,000 ImageNet validation images using one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python "$MIXTOK_H200_WORK/addMixtok/llamagen_full_loss/eval.py" \
  --source-root "$MIXTOK_H200_WORK/MixTok/reconstruction" \
  --llamagen-root "$MIXTOK_H200_WORK/LlamaGen" \
  --llamagen-ckpt "$MIXTOK_H200_WORK/LlamaGen/pretrained_models/vq_ds16_c2i.pt" \
  --data-root "$MIXTOK_IMAGENET_VAL" \
  --checkpoint "$MIXTOK_H200_RUN/latest.pt" \
  --output "$MIXTOK_H200_RUN/val50k.json" \
  --wandb-mode online
```

The evaluator reports reconstruction FID and PSNR with the EMA weights, saves them in `val50k.json`, and logs both to a separate W&B evaluation run grouped with the training run. It computes FID with TorchMetrics Inception-2048; record the evaluator identity when comparing with results from another FID implementation. Its Inception weight may need a first-time download. Report `val50k.json`, the selected per-GPU and global batch sizes and peak-memory fraction, training duration, W&B run links, and any deviations from this runbook. The short 5090 execution checks are in [`SMOKE_TEST_5090.md`](SMOKE_TEST_5090.md); they are not full ImageNet results.
