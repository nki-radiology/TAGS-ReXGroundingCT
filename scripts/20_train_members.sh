#!/bin/bash
# Train the 12 VoxTell members, or their train-only twins (needed only to fit the post-processing on validation).
#   bash scripts/20_train_members.sh all     -> tags/checkpoints/<member>.pt             (challenge train + validation)
#   bash scripts/20_train_members.sh train   -> tags/checkpoints/<member>_trainonly.pt  + validation probability maps
# Every member starts from VoxTell v1.1 and trains 250 epochs on one GPU (about 7 h on an H100) with the same recipe:
# one third of the crops centred on a finding, text augmentation (p=0.5), SGD 1e-4 with polynomial decay, batch 8.
# Members differ only in the option listed next to their name. Training is not seeded.
set -euo pipefail
MODE=${1:?usage: $0 all|train}
cd tags; mkdir -p checkpoints
if [ "$MODE" = all ]; then IDX=t1_index_all.jsonl; SUF=""; else IDX=t1_index_train.jsonl; SUF=_trainonly; fi
BASE="--epochs 250 --iters 64 --batch 1 --micro 4 --n-crops 8 --workers 14 --no-anat --index $IDX \
      --fg-prob 0.33 --pair-prob 0.0 --text-aug 0.5 --optim sgd --lr 1e-4 --save-every 5"
AUX="--aux-bank aux_bank_clean --aux-prob 0.4"
MEMBERS=(
  "plain|"                                                                   # the recipe as is
  "plain_2|"                                                                 # second run of the same recipe
  "tversky_fp|--tversky 0.3,0.7"                                             # Tversky loss penalising false positives
  "tversky_fp_2|--tversky 0.3,0.7"                                           # second run
  "tversky_fn|--tversky 0.6,0.4"                                             # Tversky loss penalising false negatives
  "aux_effusion|$AUX --aux-index aux_bank_clean/index_2e_only_clean.jsonl --aux-dilate 0"   # + CT-RATE effusion pseudo-labels
  "aux_nodule_effusion|$AUX --aux-index aux_bank_clean/index_pilot_clean.jsonl --aux-dilate 2"  # + CT-RATE nodule and effusion pseudo-labels
  "fusion24|--fusion-stage 3 --fusion-lr-mult 10"                            # finer 24^3 text-image fusion grid
  "lobe_aux|--lobe-prob 0.6 --fg-jitter 0.3"                                 # lobe masks as auxiliary target
  "nodule_boost15|--cat-boost 2d:1.5"                                        # scans with a nodule finding sampled 1.5x
  "nodule_boost20|--cat-boost 2d:2.0"                                        # ... sampled 2x
  "category_balance|--cat-balance 0.5"                                       # rare finding categories sampled more often
)
for e in "${MEMBERS[@]}"; do
  m=${e%%|*}; extra=${e#*|}
  python train_t1_full.py $BASE $extra --out checkpoints/$m$SUF.pt
done
if [ "$MODE" = train ]; then   # validation probability maps of the twins (stride 1/4), inputs of scripts/60
  for e in "${MEMBERS[@]}"; do m=${e%%|*}
    python cache_val_probs.py --ckpt checkpoints/$m$SUF.pt --out CACHE_sf25_$m$SUF --index t1_index_val.jsonl --step-frac 0.25
  done
fi
