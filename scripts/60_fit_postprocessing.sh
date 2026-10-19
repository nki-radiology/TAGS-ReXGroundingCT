#!/bin/bash
# Threshold model and component head, fitted on VALIDATION maps of the 9 train-only twins (scripts/30) fused
# 0.8/0.2 with train-only TAP-CT validation maps (tapct_maps/val200_prob; scripts/50 without --train-on-val).
set -euo pipefail
cd tags; mkdir -p results
TP2_MEMBERS=$(head -1 subsets_afternoon.txt) TP2_TARGET=__MEAN__ TP2_PREFIX=CACHE_sf25_ TP2_WORKERS=12 \
  TP2_TAP="${TAGS_TAPCT_VAL_MAPS:-../tapct_maps/val200_prob}" TP2_TAPW=0.2 TP2_OUT=results/thr_model_lobe16.pkl \
  LOBE_K=16 python thr_predictor2.py
HC_TAPCH=1 HC_OUT=results/comp_crops_val_tap.pkl python harvest_components_crops.py
CH_PKL=results/comp_crops_val_tap.pkl CH_SKIP_OOF=1 CH_DEVICE=cpu CH_OUT=checkpoints/comp_head4.pt python comp_head.py
