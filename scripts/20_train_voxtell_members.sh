#!/bin/bash
# The 12 VoxTell members of the submission (all trained on challenge train + val). They come from three
# recipes, because older members were extended rather than retrained when the recipe improved (README):
#   members 1-4: July recipe + stage-0 encoder, rebuilt on train+val, then val x5, then the new recipe as a last stage
#   member 5:    lobe-aux experiment from VoxTell v1.1, then the new recipe as a last stage
#   members 6-12: the new recipe from VoxTell v1.1 (1/3 finding-centred crops, text augmentation) One GPU each unless noted;
# the submission used A100/H100 80 GB. Training is unseeded, so retrained models will differ slightly.
set -euo pipefail
cd tags; C=checkpoints; mkdir -p $C
BASE="--iters 64 --batch 1 --n-crops 8 --no-anat --optim sgd --lr 1e-4"

# Stage 1: SUBMISSION_all.pt = VoxTell v1.1 + stage-0 encoder, 250 epochs, 3-GPU DDP (effective batch 24)
torchrun --standalone --nproc_per_node=3 train_t1_full.py --epochs 250 $BASE --micro 2 --workers 6 \
    --index t1_index_all.jsonl --pair-prob 0.5 --enc-from ../weights/enc501_best.pth --out $C/SUBMISSION_all.pt

# Members 1-4: three fine-tuning stages each from SUBMISSION_all.pt (epochs are global: 250->310->370->430)
declare -A EXTRA=(
  [cautious]="--tversky 0.3,0.7"
  [aux_2e]="--aux-index aux_bank/index_2e_only.jsonl --aux-bank aux_bank --aux-prob 0.4 --aux-dilate 0"
  [aux_dil2]="--aux-index aux_bank/index_pilot.jsonl --aux-bank aux_bank --aux-prob 0.4 --aux-dilate 2"
  [fusion24]="--fusion-stage 3 --fusion-lr-mult 10" )
for m in cautious aux_2e aux_dil2 fusion24; do
  python train_t1_full.py --epochs 310 $BASE --micro 2 --workers 4 --index t1_index_all.jsonl --pair-prob 0.5 \
      --resume $C/SUBMISSION_all.pt ${EXTRA[$m]} --out $C/${m}_all.pt
  python train_t1_full.py --epochs 370 $BASE --micro 2 --workers 8 --index t1_all_valx5.jsonl --pair-prob 0.5 \
      --resume $C/${m}_all.pt --out $C/${m}_v5.pt
  python train_t1_full.py --epochs 430 --no-anat --index t1_all_valx5.jsonl --text-aug 0.5 --fg-prob 0.33 \
      --pair-prob 0.0 --micro 4 --workers 14 --save-every 5 --resume $C/${m}_v5.pt --out $C/${m}_v5_aug33.pt
done

# Member 5: lobe-aux (from VoxTell v1.1; lobe-localisation crops in the first two stages)
python train_t1_full.py --epochs 120 $BASE --micro 2 --workers 4 --index t1_index_train.jsonl --pair-prob 0.5 \
    --lobe-prob 0.6 --fg-jitter 0.3 --cat-dropout 0.3 --out $C/lobeaux_v11C_lp6.pt
python train_t1_full.py --epochs 250 $BASE --micro 2 --workers 8 --index t1_index_all.jsonl --pair-prob 0.5 \
    --lobe-prob 0.6 --fg-jitter 0.3 --resume $C/lobeaux_v11C_lp6.pt --out $C/lobeaux_all.pt
python train_t1_full.py --epochs 310 --no-anat --index t1_all_valx5.jsonl --text-aug 0.5 --fg-prob 0.33 \
    --pair-prob 0.0 --micro 4 --workers 14 --save-every 5 --resume $C/lobeaux_all.pt --out $C/lobeaux_all_aug33.pt

# Members 6-12: from VoxTell v1.1, 250 epochs (~7 h on one H100)
FLEET="--epochs 250 $BASE --micro 4 --workers 14 --index t1_index_all.jsonl --fg-prob 0.33 --pair-prob 0.0 --text-aug 0.5 --save-every 5"
python train_t1_full.py $FLEET --out $C/fleet_plain.pt
python train_t1_full.py $FLEET --tversky 0.3,0.7 --out $C/fleet_tvA.pt
python train_t1_full.py $FLEET --tversky 0.6,0.4 --out $C/fleet_tvB.pt
python train_t1_full.py $FLEET --out $C/fleet_seed2.pt                      # second unseeded run of fleet_plain
python train_t1_full.py $FLEET --cat-boost 2d:1.5 --out $C/mb_2dboost.pt
python train_t1_full.py $FLEET --cat-balance 0.5 --out $C/mb_catbal.pt
python train_t1_full.py $FLEET --cat-boost 2d:2.0 --out $C/mb_boost20.pt
