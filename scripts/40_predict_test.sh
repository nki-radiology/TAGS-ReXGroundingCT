#!/bin/bash
# Test-set probability maps of the 12 members (sliding window, stride 1/8; about 3.7 min per scan per member on an H100).
set -euo pipefail
cd tags
for m in plain plain_2 tversky_fp tversky_fp_2 tversky_fn aux_effusion aux_nodule_effusion fusion24 lobe_aux nodule_boost15 nodule_boost20 category_balance; do
  python predict_test.py --ckpt checkpoints/$m.pt --thr 0.4 --restrict 1 --step-frac 0.125 --save-prob TESTPROB_$m --out /tmp/unused_masks_$m
done
