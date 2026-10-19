#!/bin/bash
# TAP-CT member (weight 0.2): TAP-CT ViT-B encoder (huggingface.co/fomofo/tap-ct-b-3d) fine-tuned with FiLM text
# conditioning (Qwen3-Embedding-0.6B). Run from the repository root.
set -euo pipefail
cd tapct; export PRIMUS_WIN_Z=24 PRIMUS_Z_MM=1.5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CACHE=cache/primus_cache; mkdir -p cache results/primus
python primus/build_cache.py --dataset rexground --out $CACHE/rexground          # 1.5 mm, LPS, int16
python primus/build_cache.py --dataset rexground_test --out $CACHE/rexground_test
python primus/embed_texts_tokens.py $CACHE/rexground                                # cache/rexground_text_tokens_qwen06b.npz
python primus/embed_texts_tokens_test.py                                            # cache/rexground_text_tokens_qwen06b_test.npz
python primus/paraphrase_texts.py --out cache/rexground_text_aug.json --n-para 3    # Qwen2.5-7B-Instruct, train sentences
python primus/embed_texts_tokens_aug.py --aug-json cache/rexground_text_aug.json --out cache/rexground_text_tokens_aug.npz

# head initialisation, encoder frozen (50k steps, train only)
python primus/referring2.py --root $CACHE/rexground --out rex_combo_z24_e16_50k --variant tap-ct-b-3d \
  --heads v3both_cat_4_6_9_12,v3bothk_cat_4_6_9_12 --text-bank tokens --iters 50000 --batch 8 --accum 2 \
  --lr 3e-4 --wd 0.05 --warmup 250 --seed 0 --stem lite --loss dc_ce --sampler legacy
# five seeds, encoder fine-tuned, train + val (these produce the TEST maps; drop --train-on-val for val-scorable twins)
for k in 0 1 2 3 4; do
  torchrun --standalone --nproc_per_node=1 primus/referring2.py --root $CACHE/rexground \
    --out rex_fttxt_all_z24_e16_25k_s$k --variant tap-ct-b-3d \
    --text-bank tokens+cache/rexground_text_tokens_aug.npz --heads v3both_cat_4_6_9_12 --finetune \
    --init-heads results/primus/rex_combo_z24_e16_50k/last.pt --iters 25000 --batch 4 --accum 4 --lr 3e-4 \
    --enc-lr 2e-5 --wd 0.05 --warmup 250 --workers 6 --val-every 5000 --val-subset 30 --seed $k --stem lite \
    --loss dc_ce --sampler legacy --train-on-val --p-para 0.5 --p-hardneg 0.5 --ddp
  python primus/export_challenge.py --runs rex_fttxt_all_z24_e16_25k_s$k --head v3both_cat_4_6_9_12 \
    --root $CACHE/rexground_test --split test --win-z 24 --bank cache/rexground_text_tokens_qwen06b_test.npz \
    --save-prob --out test300_fttxt_all_s$k
done
python primus/mean_probs.py --dirs $(printf 'test300_fttxt_all_s%s_prob,' 0 1 2 3 4 | sed 's/,$//') --split test \
    --out ../tapct_maps/test300_prob
