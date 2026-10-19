#!/bin/bash
# Train-only "twins" whose VALIDATION maps are used to fit the threshold model and the component head
# (scripts/60). They never see validation data, so their validation maps are honest.
# The first four are older, simpler versions of members 1-4 (one 60-epoch fine-tune of the train-only
# SUBMISSION_1ch), not exact train-only copies. fleet_plain, fleet_tvA, fleet_tvB, mb_2dboost and mb_catbal have
# exact twins (same recipe, train index only).
# lobeaux, fleet_seed2 and mb_boost20 have no twin.
set -euo pipefail
cd tags; C=checkpoints
BASE="--iters 64 --batch 1 --n-crops 8 --no-anat --optim sgd --lr 1e-4"
torchrun --standalone --nproc_per_node=3 train_t1_full.py --epochs 250 $BASE --micro 2 --workers 4 \
    --index t1_index_train.jsonl --pair-prob 0.5 --enc-from ../weights/enc501_best.pth --out $C/SUBMISSION_1ch.pt
TW="--epochs 310 $BASE --micro 2 --workers 4 --index t1_index_train.jsonl --pair-prob 0.5 --resume $C/SUBMISSION_1ch.pt"
python train_t1_full.py $TW --tversky 0.3,0.7 --out $C/cautious.pt
python train_t1_full.py $TW --aux-index aux_bank/index_2e_only.jsonl --aux-bank aux_bank --aux-prob 0.4 --aux-dilate 0 --out $C/aux_2e.pt
python train_t1_full.py $TW --aux-index aux_bank/index_pilot.jsonl --aux-bank aux_bank --aux-prob 0.4 --aux-dilate 2 --out $C/aux_dil2.pt
python train_t1_full.py $TW --fusion-stage 3 --fusion-lr-mult 10 --out $C/fusion24.pt
FL="--epochs 250 $BASE --micro 4 --workers 14 --index t1_index_train.jsonl --fg-prob 0.33 --pair-prob 0.0 --text-aug 0.5 --save-every 5"
python train_t1_full.py $FL --out $C/twin_plain.pt
python train_t1_full.py $FL --tversky 0.3,0.7 --out $C/twin_tvA.pt
python train_t1_full.py $FL --tversky 0.6,0.4 --out $C/twin_tvB.pt
python train_t1_full.py $FL --cat-boost 2d:1.5 --out $C/twin_aug_2dboost.pt
python train_t1_full.py $FL --cat-balance 0.5 --out $C/twin_aug_catbal.pt
# validation probability maps (stride 1/4)
for m in cautious aux_2e aux_dil2 fusion24 twin_plain twin_tvA twin_tvB twin_aug_2dboost twin_aug_catbal; do
  python cache_val_probs.py --ckpt $C/$m.pt --out CACHE_sf25_$m --index t1_index_val.jsonl --step-frac 0.25
done
