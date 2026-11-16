#!/usr/bin/env bash
set -euo pipefail

stage=${1:?spatial or budget}
kind=${2:?ablation kind}
gpus=${3:?comma-separated GPU indices}
case "$stage" in
  spatial|budget) ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
case "$kind" in
  without_gain|without_white_all_explicit|without_cam_all_explicit) ;;
  *) echo "unknown kind: $kind" >&2; exit 2 ;;
esac

here=$(cd "$(dirname "$0")" && pwd)
paper=$(cd "$here/../.." && pwd)
original="$paper/addMixtok_metrics_work/mixtok_long_no_density/gene/scorer_training/$stage"
config="$here/${stage}_${kind}/stage_config.json"
[[ -f "$config" ]] || { echo "missing $config" >&2; exit 2; }
count=$(awk -F, '{print NF}' <<< "$gpus")
[[ "$count" == 2 || "$count" == 4 ]] || { echo 'use 2 or 4 GPUs' >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="$gpus"
export MIXTOK_SCORER_STAGE_CONFIG="$config"
export MIXTOK_REPO=/home/heyefei/lichenge/MixTok-Official
export MIXTOK_TITOK_ROOT=/home/heyefei/lichenge/1d-tokenizer
export MIXTOK_RECON_CODEC=/home/heyefei/lichenge/MixTok-Official/assets/released/recon/mixtok_recon.pt
export MIXTOK_TITOK_WEIGHTS=/home/heyefei/lichenge/1d-tokenizer/tokenizer_titok_l32.bin
export MIXTOK_LLAMAGEN_ROOT=/home/heyefei/lichenge/LlamaGen
export MIXTOK_LLAMAGEN_WEIGHTS=/home/heyefei/lichenge/LlamaGen/pretrained_models/vq_ds16_c2i.pt
export MIXTOK_B0_WEIGHTS=/tmp/mixtok_light_backbones_20261111/efficientnet_b0_rwightman-7f5810bc.pth
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

extra=()
if [[ "$stage" == spatial ]]; then extra+=(--baseline); fi
exec python -m torch.distributed.run --standalone --nproc_per_node="$count" \
  "$original/train.py" --output "$here/${stage}_${kind}" --updates 2000 --batch 16 --eval-every 500 "${extra[@]}"
