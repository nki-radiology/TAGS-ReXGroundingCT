#!/bin/bash
# Test-set probability maps of the 12 members (stride 1/8; ~3.7 min per scan per member on one H100).
set -euo pipefail
cd tags; C=checkpoints
declare -A CK=( [cautious_aug33]=cautious_v5_aug33 [aux2e_aug33]=aux_2e_v5_aug33 [auxdil2_aug33]=aux_dil2_v5_aug33
  [fusion24_aug33]=fusion24_v5_aug33 [lobeaux_aug33]=lobeaux_all_aug33 [fleet_plain]=fleet_plain [fleet_tvA]=fleet_tvA
  [fleet_tvB]=fleet_tvB [fleet_seed2]=fleet_seed2 [mb_2dboost]=mb_2dboost [mb_catbal]=mb_catbal [mb_boost20]=mb_boost20 )
for m in "${!CK[@]}"; do
  python predict_test.py --ckpt $C/${CK[$m]}.pt --thr 0.4 --restrict 1 --step-frac 0.125 \
      --save-prob TESTPROB_${m}_sf125 --out /tmp/unused_masks_$m
done
