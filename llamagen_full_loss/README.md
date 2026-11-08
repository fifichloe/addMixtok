# Pure LlamaGen VQ reconstruction control

This experiment starts from the original LlamaGen `vq_ds16_c2i.pt` EMA weights. It trains the VQ encoder, quantization convolution, post-quantization convolution, and decoder with the MixTok fixed-96 reconstruction loss recipe. The codebook is frozen. The TiTok model, MixTok adapter, grid scorer, and sparse routing are not used in reconstruction; the TiTok source tree is needed to import the projected discriminator implementation.

The loss weights are L1 ×2, LPIPS ×0.6, DINOv2 ×0.5, CLIP ×0.5, projected ConvNeXt GAN ×0.12, discriminator feature matching ×0.5, and LeCam ×0.001. GAN training starts at epoch 1.5. `loss_config.yaml` is a copy of the fixed-96 configuration; router settings in that file are unused by this trainer.

## Requirements

Use the MixTok reconstruction environment and its public source at commit `6623ccdc7a48688497f7b1f312548d1f14107e4d`. Provide the ImageNet train and validation folders, the original LlamaGen VQ checkpoint, TiTok source, cached DINOv2 source/weights, and CLIP weights. The run script validates the required paths before allocating GPUs. Log in to W&B for the default online mode, or use `--wandb-mode offline`.

## Train on eight H200 GPUs

Run from any working directory. The launcher selects eight idle GPUs by default, probes per-GPU batch sizes with **all losses including GAN enabled**, and chooses the largest measured batch whose peak memory falls within **92%–95%** of usable GPU memory. If a batch is too large, the failed probe is recorded and the search continues. If no integer batch falls in this interval, training does not start and the probe logs explain the gap. The selected batch, memory measurements, and GPU IDs are written to `batch_selection.json` before training starts.

```bash
python /path/to/addMixtok/llamagen_full_loss/run.py \
  --data-root /path/to/ImageNet/train \
  --output /path/to/results/llamagen_full_loss \
  --source-root /path/to/MixTok/reconstruction \
  --titok-root /path/to/1d-tokenizer \
  --llamagen-root /path/to/LlamaGen \
  --llamagen-ckpt /path/to/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --dino-repo /path/to/facebookresearch_dinov2_main \
  --clip-cache-dir /path/to/open_clip_cache
```

The defaults are eight GPUs, 20 epochs, gradient accumulation 1, a 92%–95% memory band, and online W&B project `mixtok-rebuttal`. `--gpus`, `--gpu-ids`, `--accum`, `--min-memory`, `--max-memory`, and `--wandb-mode` may be changed. Use `tmux` or a job scheduler for the full run; `run.py` stays attached until training finishes.

`latest.pt` is atomically replaced after **every epoch**. The only separately retained intermediate checkpoint is `epoch_10.pt`. No `epoch_20.pt` is written; `latest.pt` holds epoch 20 after completion. Re-running the same command with `latest.pt` present resumes from the next epoch using the saved batch selection. Keep the same GPU count and accumulation on resume.

The training log is `train.log`, and full-loss probe logs are under `probes/`. The probe forces GAN on from the first update to expose the higher-memory training phase. The actual 20-epoch training retains the scheduled GAN start at epoch 1.5. The launcher requires a measured batch within the requested memory band. The batch choice changes global batch size and is recorded in `run_config.json` and W&B.

## Evaluate

After training, evaluate the EMA model on all 50,000 ImageNet validation images:

```bash
CUDA_VISIBLE_DEVICES=0 python /path/to/addMixtok/llamagen_full_loss/eval.py \
  --source-root /path/to/MixTok/reconstruction \
  --llamagen-root /path/to/LlamaGen \
  --llamagen-ckpt /path/to/LlamaGen/pretrained_models/vq_ds16_c2i.pt \
  --data-root /path/to/ImageNet/validation \
  --checkpoint /path/to/results/llamagen_full_loss/latest.pt \
  --output /path/to/results/llamagen_full_loss/val50k.json
```

The evaluator reports reconstruction FID and PSNR with the EMA weights.
