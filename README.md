# K=256 full-grid control for MixTok

This repository separates two experiments that were previously conflated.

## Fair reconstruction control: train from the original weights

[`reconstruction_k256/train_full256_20epoch.yaml`](reconstruction_k256/train_full256_20epoch.yaml) and [`reconstruction_k256/continue_full256_6epoch.yaml`](reconstruction_k256/continue_full256_6epoch.yaml) are the K=256 full-grid counterparts to MixTok's fixed-96 20-epoch stage and 6-epoch continuation. Run them with the **same reconstruction training script and ImageNet data**, starting from the original TiTok-L32 and LlamaGen VQ-16 checkpoints. They keep the K=96 run's image, LPIPS, DINOv2, CLIP, GAN, and discriminator feature-matching losses, optimizer settings, batch sizes, learning-rate schedules, and preprocessing. The full-grid mask is one at every location; the grid scorer and 1D adapter are not trained because neither can influence the reconstruction when K=256. The LlamaGen VQ encoder, quantization convolution, post-quantization convolution, and decoder follow the same training flags as K=96. The codebook receives only a zero-gradient term in the shared trainer, so this recipe does not claim to update the codebook.

The config uses the existing training script's `per_image_random` mask mode with `mask_ratio_min=mask_ratio_max=1.0`. In this special case the implementation selects **all 256 positions deterministically**; there is no random spatial selection. `lambda_mix=2.0` and the other reconstruction losses remain the same as the K=96 training config. The config loads the same 2.5-epoch adapter initialization as the K=96 run to satisfy the shared model/evaluator checkpoint contract, but keeps it frozen. Its output has zero weight in the all-2D reconstruction. The LlamaGen VQ weights still start at the original pretrained checkpoint, just as in K=96.

The K=96 run spends the first 0.5 epoch of its 20-epoch stage training only the scorer. Full-grid K=256 has no scorer to train, so its 2D network receives those additional 0.5 epoch of updates. This favors the dense control slightly; the runs are matched in overall epochs, data, losses, and optimizer settings, but not in exact 2D-network update count or FLOPs.

Clone [MixTok](https://github.com/LCGLiChenge/MixTok) at commit `6623ccdc7a48688497f7b1f312548d1f14107e4d`, arrange its ImageNet and original pretrained-weight paths as described in its reconstruction README, then run from its `reconstruction/` directory. If the original 2.5-epoch adapter checkpoint has not been produced yet, first run the same Stage 1 command as K=96:

```bash
torchrun --standalone --nproc_per_node=8 train_titok_llamagen_recon.py \
  --config configs/adapter_only_2p5epoch.yaml
```

Then train and continue the full-grid model with the matching 20+6 epoch schedule:

```bash
torchrun --standalone --nproc_per_node=8 \
  train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config /path/to/addMixtok/reconstruction_k256/train_full256_20epoch.yaml

torchrun --standalone --nproc_per_node=8 \
  train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config /path/to/addMixtok/reconstruction_k256/continue_full256_6epoch.yaml
```

For reconstruction evaluation, run the existing evaluator with the matching full-grid config (edit its `ckpt` and data paths to match your installation):

```bash
python eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py \
  --config /path/to/addMixtok/reconstruction_k256/eval_full256_50k.yaml
```

The evaluation mask also covers every position, so `mix` is the pure 2D reconstruction. This configuration is intended to answer whether MixTok's sparse hybrid reconstruction improves over a 256-token LlamaGen VQ tokenizer trained with the same full reconstruction loss recipe. No K=256 reconstruction result is claimed here; the training and evaluation have not been run from this config.

## Earlier generation-only control

[`generation_only_control/`](generation_only_control/) contains the 20,000-update K=256 Halton generator experiment previously uploaded here. It starts its **generator** from pretrained Halton weights, but its token cache and RGB decoder come from the already-trained **sparse MixTok** reconstruction checkpoint. Its reported gFID therefore does **not** constitute the full-grid control above. The original README is retained inside that directory for provenance. To compare full-grid generation fairly, retrain the generator on codes extracted from the new K=256 reconstruction tokenizer and evaluate with its decoder; the earlier generation-only scripts do not yet implement that chain.
