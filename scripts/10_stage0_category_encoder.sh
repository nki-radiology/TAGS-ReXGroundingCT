#!/bin/bash
# Stage 0: text-free 14-category nnU-Net (ResEnc-L) on the challenge TRAIN set, encoder initialised from
# VoxTell v1.1. Its encoder initialises stage 1 (SUBMISSION_all), the starting point of 4 of the 12 members.
# LEGACY: an ablation on 6 Aug 2026 found no measurable benefit (val 0.335 with vs 0.331 without), so all
# later members start from VoxTell v1.1 directly. Needed only to reproduce cautious/aux2e/auxdil2/fusion24. Needs nnU-Net v2 + the VoxTell package with our trainer
# files (stage0/voxtell_training/*) copied into voxtell/training/ and stage0/nnunetv2_variant/voxtell_link.py
# into nnunetv2/training/nnUNetTrainer/variants/.
set -euo pipefail
: "${nnUNet_raw:?}" "${nnUNet_preprocessed:?}" "${nnUNet_results:?}"
export TAGS_STAGE0_DATA="$PWD/stage0/data"        # class / case weights (computed from the train labels)
cd stage0
python 24_build_nnunet_dataset.py                  # Dataset500_ReXGround14 (HU images + 14-category labels)
for s in $(seq 0 15); do python 26_build_region_dataset.py --shard "$s" --nshards 16 & done; wait   # Dataset501
nnUNetv2_plan_and_preprocess -d 501 -pl nnUNetPlannerResEncL -c 3d_fullres --verify_dataset_integrity -np 16
voxtell-finetune 501 3d_fullres 0 -p nnUNetResEncUNetLPlans -tr VoxTellTrainer_region_presence_tversky_b90 \
    -pretrained_weights "${TAGS_WEIGHTS:-../weights}/voxtell_v1.1/fold_0/checkpoint_final.pth"
# The submission used checkpoint_best.pth of fold 0 (epoch 241) as weights/enc501_best.pth.
