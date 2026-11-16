#!/usr/bin/env bash
set -euo pipefail

source_dir=/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/no_density_revision_20261114/final/full/cache
destination_dir=/home/heyefei/lichenge/paper/experiments/generation_score_ablation_20261117/base_local
mkdir -p "$destination_dir"
rsync -a --bwlimit=35000 --info=stats2 "$source_dir"/ "$destination_dir"/
date -Is > "$destination_dir/copy.complete"
