# Current density-free Long results (ImageNet 256×256)

This file records completed measurements for the current Long model. It is not the original top-p model or the later fixed20 diagnostic. Values come from the full 50,000-image validation/generation evaluations unless another sample count is stated. Exact values and SHA256 receipts are in [`metrics_current_20261116.json`](metrics_current_20261116.json).

## Main results

| Task | Result | Representation / setup |
|---|---:|---|
| Reconstruction rFID ↓ | **1.000763** | mean local K = 96 (range 48–144); 128 content tokens |
| Reconstruction PSNR ↑ | **19.4012 dB** | same 50k validation images |
| Reconstruction fixed-width bitrate | **0.03027344 bpp** | 32×12 + 96×14 token bits + 256-bit location mask |
| Generation gFID ↓, four-seed mean ± sample SD | **1.8652 ± 0.0020** | CFG 4.5/3.0; 8+8 rounds; mean local K = 106.38153; 138.38153 content tokens |
| Generation end-to-end latency | **250.97 ms/image** | one RTX 5090, batch 1, class 207, 10 warmups + 50 timed runs |
| Generation supported-operator FLOPs | **1.636 T/image** | class 207, seed 4; unsupported operators excluded |

The generation timing cohort has mean local K = 92.20, whereas the 50k seed-4 quality cohort has mean local K = 106.38386. Do not equate the two cohorts or infer proportional compute savings from token count.

## Generation CFG scan (same 20k-update checkpoint, seed 4)

| 2D CFG | 1D base gFID ↓ | Full gFID ↓ | Mean local K | Mean content tokens |
|---:|---:|---:|---:|---:|
| 2.50 | 4.2890 | 1.9147 | 106.38386 | 138.38386 |
| 2.75 | 4.2890 | 1.8814 | 106.38386 | 138.38386 |
| 3.00 | 4.2890 | 1.8654 | 106.38386 | 138.38386 |
| 3.25 | 4.2890 | 1.8794 | 106.38386 | 138.38386 |
| 3.50 | 4.2890 | 1.9116 | 106.38386 | 138.38386 |

The lowest full-50k seed-4 gFID in this scan is at 2D CFG 3.0. All five points use 8+8 rounds and the same generated token budget distribution.

## Current Long sampling-round comparison (seed 4, full 50k)

| Sampling rounds | gFID ↓ | Mean local K |
|---:|---:|---:|
| 8+4 | 2.006164 | 106.38386 |
| 8+8 | 1.865404 | 106.38386 |
| 8+16 | 1.848165 | 106.38386 |

All completed rows use the same 20k-update checkpoint, CFG 4.5/3.0, seed 4, and the same mean 2D token budget. The 8+24 evaluation is still waiting for a generation GPU; it is not included yet.

## Main CFG 3.0 sampling seeds

| Seed | 1D base gFID ↓ | Full gFID ↓ | Mean local K | Mean content tokens |
|---:|---:|---:|---:|---:|
| 1 | 4.3209 | 1.8629 | 106.30722 | 138.30722 |
| 2 | 4.3851 | 1.8647 | 106.52304 | 138.52304 |
| 3 | 4.3118 | 1.8677 | 106.31200 | 138.31200 |
| 4 | 4.2890 | 1.8654 | 106.38386 | 138.38386 |
| Mean ± sample SD | — | **1.8652 ± 0.0020** | — | — |

These seeds vary sampling for one fixed checkpoint; they are not independent training repeats.

## Reconstruction controls (same frozen codec, 50k validation images)

| Selection | Mean local K | rFID ↓ | PSNR (dB) ↑ |
|---|---:|---:|---:|
| `full_dynamic96` | 96.00 | 1.000763 | 19.4012 |
| `full_fixed96` | 96.00 | 1.035990 | 19.3266 |
| `U_zero_dynamic96` | 96.00 | 1.030274 | 19.3393 |
| `without_white_dynamic96` | 96.00 | 1.002403 | 19.3401 |
| `without_gain_dynamic96` | 96.00 | 1.001141 | 18.2065 |
| `without_cam_dynamic96` | 96.00 | 1.131262 | 19.6117 |
| `without_white_fixed96` | 96.00 | 1.042281 | 19.2685 |
| `without_gain_fixed96` | 96.00 | 1.039787 | 18.0473 |
| `without_cam_fixed96` | 96.00 | 1.200785 | 19.5625 |

These are inference ablations with the same trained codec and scorer weights. They do not estimate variability across independently trained models. The primary planned paper control groups gain+white+CAM as one local score module; individual rows are diagnostic.

## Reconstruction selection and budget controls (same frozen checkpoint, full 50k)

| Selection | Mean local K | Fixed-width bpp incl. mask | rFID ↓ | PSNR (dB) ↑ |
|---|---:|---:|---:|---:|
| `teacher_dynamic96` | 96.00 | 0.030273 | 1.000752 | 19.4012 |
| `rank_uniform` | 96.00 | 0.030273 | 1.575418 | 17.2231 |
| `rank_random` | 96.00 | 0.030273 | 1.491160 | 17.2867 |
| `rank_gradient_entropy` | 96.00 | 0.030273 | 1.340215 | 18.0730 |
| `fixed0` | 0.00 | 0.009766 | 2.729450 | 15.4488 |
| `fixed32` | 32.00 | 0.016602 | 1.485138 | 17.3096 |
| `fixed64` | 64.00 | 0.023438 | 1.125010 | 18.4526 |
| `fixed96` | 96.00 | 0.030273 | 1.035932 | 19.3266 |
| `fixed128` | 128.00 | 0.037109 | 1.109700 | 20.0076 |
| `fixed196` | 196.00 | 0.051636 | 1.710688 | 20.9364 |
| `fixed256` | 256.00 | 0.064453 | 2.513099 | 21.2596 |
| `dynamic64` | 64.00 | 0.023438 | 1.080178 | 18.5146 |
| `matched_mean64` | 64.00 | 0.023438 | 1.125010 | 18.4526 |
| `shuffled64` | 64.00 | 0.023438 | 1.149850 | 18.4052 |
| `matched_mean96` | 96.00 | 0.030273 | 1.035932 | 19.3266 |
| `shuffled96` | 96.00 | 0.030273 | 1.049778 | 19.2434 |
| `dynamic128` | 128.00 | 0.037109 | 1.115123 | 20.0562 |
| `matched_mean128` | 128.00 | 0.037109 | 1.109700 | 20.0076 |
| `shuffled128` | 128.00 | 0.037109 | 1.128081 | 19.9643 |

All 19 policies were evaluated on the same 50,000 validation images, with FID pooled across four feature shards. The learned ranking strongly improves on random, uniform, and image-gradient-entropy rankings at mean K=96. Dynamic allocation improves rFID over fixed K at mean K=64 and 96, but not at 128. The K sweep changes masks at inference on one frozen checkpoint; K=0 and K=256 are not separately trained optima. The fixed-width bpp includes a raw 256-bit mask even when K=0 or 256, where that mask is redundant.

Source result SHA256: `3c97eecec0d52caad9e0a2c717be04833b5df664412d384fed06e24878dc1b4e`; protocol SHA256: `7c9fbeca9c09a57dd914e3cf3e98836c690ff77e1f6fdca1b36af10112772ae5`.

## Generation fixed-budget inference control

| Selection on seed 4 | Mean local K | gFID ↓ |
|---|---:|---:|
| Dynamic | 106.38386 | 1.865404 |
| Fixed K = 106 | 106.00000 | 2.022093 |

This control changes selection at inference with the same generator checkpoint. Its mean K differs by 0.38386, and its base features were numerically equivalent within 1e-5 rather than bitwise equal. It is not an independently trained fixed-budget generator.

## Separate matched-loss LlamaGen tokenizer control

| Tokenizer | Content tokens | rFID ↓ | PSNR (dB) ↑ | Fixed-width bpp |
|---|---:|---:|---:|---:|
| Pure LlamaGen VQ-16 EMA, H200 full-loss training | 256 | 1.238138 | 21.2151 | 0.05468750 |

This is a separate 50k reconstruction control with its own tokenizer checkpoint. No pure LlamaGen gFID is claimed from this checkpoint yet.

## Runtime detail

| Measurement | Value | Protocol |
|---|---:|---|
| Reconstruction end-to-end mean | 44.476 ms/image | one RTX 5090; batch 1; 200 validation images; 20 warmups |
| Reconstruction grid scoring + K selection | 1.218 ms | same-pass CUDA event attribution |
| Reconstruction complete routing, including preparation | 16.151 ms | 35.85% of instrumented CUDA timeline |
| Generation end-to-end median | 250.965 ms/image | class 207; batch 1; 50 timed runs |
| Generation stage 1 component median | 145.155 ms | separately instrumented |
| Generation stage 2 component median | 82.989 ms | separately instrumented |
| Generation RGB decoder component median | 7.792 ms | separately instrumented |

Component medians come from separate instrumented passes and should not be summed as the exact end-to-end time. FLOPs count supported PyTorch operations; sorting, sampling, and other unsupported operations still contribute to latency.

## Pending measurements

- Independent random/uniform and nearconstant generator training and corresponding gFID are pending.
- Current Long sampling-round sweep is pending.
- Pure LlamaGen 12/16/24-round generator evaluation is pending; its tokenizer rFID is a separate control and is not a MixTok Long result.

## Model identity

- MaskGIT checkpoint SHA256: `04a2f5e061fa0ce463755f73e008a620e8e0cce56d429aa21aeb79bfc2c869af`; 20,000 optimizer updates, global batch 256.
- Generation router design SHA256: `7fccc46707b7a641e2f85dc5750849e258c81784233227a12ce986808b1bc0eb`.
- Codec, adapter, router and official initialization SHA256 values: [`configs/assets.json`](configs/assets.json).
- Exact values, K histograms, protocols and source-result SHA256 values: [`metrics_current_20261116.json`](metrics_current_20261116.json).
