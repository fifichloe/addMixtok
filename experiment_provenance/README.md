# MixTok experiment source and actual run records (November 2026)

This folder indexes **source snapshots and completed run records** for the current density-free Long router. Python entry points are under [`../experiments`](../experiments); `actual_runs/` contains copied configuration and results from executed jobs. The existing `mixtok_long_no_density/configs/scorer_stage.example.json` is a template and is **not** the source of the batch sizes or checkpoint steps below.

These scripts retain the absolute asset/data paths of the machine used for the experiments. They require the released MixTok code and pretrained assets, ImageNet manifests, and the intermediate caches named in the configs. This archive contains no ImageNet images, route/teacher arrays, Inception feature arrays, or model checkpoints. Manifest hashes and source hashes in the records identify the omitted data and weights.

## Where each requested item is

| Item | Source files | Actual record or artifact |
| --- | --- | --- |
| Dense benefit labels | `experiments/independent_onepass_goal110_20261112/cache_dense_gain.py`, `routing.py`, `gain_model.py`, `fit_dense_gain.py`; `experiments/fresh_reconstruction_scorer_20261112/runtime.py` | `experiments/independent_onepass_goal110_20261112/dense_gain_head/report.json` |
| Input statistics and 12,000-image split | `experiments/light_grid_105_20261109/prepare_input_scales.py`, `prepare_gradient_fit.py`; `experiments/fresh_reconstruction_scorer_20261112/cache_targets.py`, `fit.py` | `actual_runs/input_scales.npz`, `actual_runs/input_stats_780.npz`, `actual_runs/input_stats_provenance.json`, `experiments/light_grid_105_20261109/gradient_fit_report.json` |
| Generation teacher targets and manifest | `experiments/generation_dynamic_router_20261112/prepare_manifest.py`; `experiments/no_density_revision_20261114/{cache_final_teacher.py,teacher_final.py,final_teacher_config.py}` | `experiments/generation_dynamic_router_20261112/data/manifest.json`, `actual_runs/generation_router/` |
| Reconstruction ablations | `experiments/no_density_revision_20261114/{prepare_remaining_reconstruction.py,audit_remaining_reconstruction.py,freeze_teachers.py}`; `experiments/new_router_reconstruction_ablations_20261114/` (earlier broad-scan harness); `experiments/full_white_cam_ablation_20261116/` | `actual_runs/reconstruction/` |
| Generation Random/Uniform/Near-constant controls | `experiments/no_density_revision_20261114/independent_controls/` (especially `routing.py`, `cache_routes.py`, `train.py`, `eval_worker.py`, `evaluate.py`) | `actual_runs/generation_controls/` |
| Efficiency | `experiments/no_density_revision_20261114/efficiency/`; `experiments/no_density_revision_20261114/recon_routing_share_20261115/`; `experiments/long_generation_speed_paper_protocol_20261115/profile_paper.py` | `actual_runs/efficiency/` |
| ALIT transfer | `experiments/alit_transfer_20261116/{cache_train.py,merge_cache.py,fit_scorer.py,runtime.py,evaluate.py}` | `actual_runs/alit/summary.json` |

`experiments/no_density_revision_20261114/maskgit/cache_routes.py` caches **the generator's selected routes**. It is not the teacher cache: `cache_final_teacher.py` writes `f_base`, `local`, and `utility` labels for router distillation. The fit/held list generator is `prepare_manifest.py`; its manifest records 1,271,167 fit images and 10,000 held images (ten per class), with hashes of the omitted path lists.

## Label and input definitions

`cache_dense_gain.py` decodes the base and full 2D features with the frozen decoder. Its two `gain.npy` channels are signed reductions in (1) local RGB squared error and (2) EfficientNet-B0 middle/deep cosine error. RGB errors use adaptive average pooling to 16×16; semantic feature errors use bilinear interpolation to 16×16. The exact combination and weighting are in that script. The 780-channel input is three 256-channel feature blocks plus 12 spatial maps, defined in `fresh_reconstruction_scorer_20261112/runtime.py`.

The 12,000 images are sampled from a train-only fit population: twelve per ImageNet class, ordered by a SHA-256 path hash. The scorer split uses source index modulo ten: 10,843 fit and 1,157 held. `input_scales.npz` has **256** channel means/stds for the 1D/2D feature whitening, computed from 84,996 train-fit images. The separate `input_stats_780.npz` contains the **780** input-channel normalization values embedded in the executed fresh-scorer checkpoint; `fit.py` computed them from 256 deterministic fit samples. Neither file is a 12,000-image full-pixel cache.

The final semantic utility ridge is `experiments/independent_onepass_goal110_20261112/image_utility/semantic_rel.npz` (included despite the repository's general `*.npz` ignore rule). It contains all 1,560 coefficients (`beta`), input mean/std, bias, target scaling, and clipping. Its held-selected relative ridge penalty is **0.01**, as recorded in `image_utility/report.json`. The density-free teacher loads this exact file in `teacher_final.py`.

## Executed Long router configuration

| Stage | Actual source config | Per-GPU microbatch | GPUs | Global batch | Selected checkpoint step |
| --- | --- | ---: | ---: | ---: | ---: |
| Base-feature teacher distillation | `actual_runs/generation_router/stage1_config.json` | 128 | 2 | 256 | 20,000 (`best.pt`) |
| Spatial reconstruction refinement | `actual_runs/generation_router/stage2_config.json` | 16 | 4 | 64 | 21,500 (`candidate_1500.pt`) |
| Budget feature and utility refinement | `actual_runs/generation_router/stage3_config.json` | 16 | 4 | 64 | 22,500 (`candidate_1000.pt`) |

Each stage has its executed `report.json` and the latter two have selection records in `actual_runs/generation_router/`. The configured stage ceilings were 20,000, 22,000 and 2,000 additional updates, respectively; selected checkpoints can precede the ceiling. The selected 22,500-step ancestry is independently recorded in `experiments/no_density_revision_20261114/final_execution/Long_selected_final_budget_checkpoint_audit.json`.

The final reconstruction-score threshold is **-0.23861974477767944** for mean local K≈96. It was calibrated on the complete 50,000-image ImageNet validation **score** set, 50 images per class, before quality evaluation; `actual_runs/generation_router/validation_threshold.json` and `Long_official50k_score_only_threshold_audit.json` preserve the population and hashes. The teacher's earlier score threshold is 0.3803865313529968; do not substitute it for the final student threshold. The generation policy applies the frozen student threshold to each generated image and has an observed mean K of 106.38386 for seed 4; it was not calibrated by looking up other images at generation time.

## Short result record and scope

The current Long seed-4 generation result is gFID **1.8654** at mean 106.38386 local tokens, with 8+8 rounds and CFG 4.5/3.0 (`experiments/no_density_revision_20261114/recovered_generation_20261116/main_selection.json`). Independently trained Random and Uniform controls reuse the same per-image budget but replace the spatial locations; their seed-4 gFIDs are **2.5844** and **2.3921**. The near-constant control uses an ID-based affine assignment of the floor/ceiling budgets and preserves the total budget; its audited seed-4 gFID is **2.0367**. The policy JSON and checkpoint hashes in `actual_runs/generation_controls/` match each corresponding summary. This near-constant control is an offline matched-budget diagnostic, not a single fixed integer K.

The **current density-free** Long reconstruction ablations, including Random, Uniform, Gradient entropy, shuffled K, and the fixed/dynamic budget scan, are in `actual_runs/reconstruction/current_long_main_results.json`; `current_long_component_results.json` records the component design gate. `long_main_results.json` and `long_policy_protocol.json` come from the earlier broad-scan harness and are retained only as source provenance, not substituted for the current results. `full_white_cam_dynamic.json` and `full_white_cam_fixed.json` are the later explicit-input ablations; they remove White/CAM from all relevant inputs as specified in `score_worker.py` and `prepare_policies.py`, unlike older direct-score-only ablations. Current generation score/U ablation scripts are included in `experiments/generation_score_ablation_20261117/`; their new 50,000-image quality results are still running and are not represented as completed here.

The ALIT transfer uses the same frozen 20-epoch ALIT Mix checkpoint and an ALIT-trained scorer. At 64 local tokens, original router fixed K, new scorer fixed K, and new scorer dynamic mean K have rFID **2.2665**, **1.9377**, and **1.9029**, respectively; the source and exact PSNR values are in `actual_runs/alit/summary.json`.

The generation latency record uses one RTX 5090, batch 1, 10 warmups, 50 synchronized measurements, 8+8 rounds and the paper's TF32/cuDNN settings. Its measured median is **250.97 ms** over the 50 fixed class/seed inputs; the measured mean K of those inputs is 92.2 and is not the 50,000-image mean. `actual_runs/efficiency/generation_latency_result.json` includes per-component times, counted FLOPs, and parameter groups. `recon_routing_protocol.json` records the separate batch-1 reconstruction timing run; those timings include dense 2D encoding.
