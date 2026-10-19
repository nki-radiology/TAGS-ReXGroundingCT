# LLM lobe mapping of the ReXGroundingCT finding sentences

The files used for the submission, included so the restriction step can run without the LLM.
`eval_t1_val200.named_region()` reads both files.

* `rexg_findings.jsonl` holds every unique finding sentence of the challenge dataset (train, val and test;
  6,926 sentences), with keys `<split>|<case>|<finding>`. It can be rebuilt with `tags/build_rexg_findings.py`.
* `rexg_ts_labels.jsonl` holds the output of `llm_mapping/map_categories_llm.py` on those sentences:
  * TotalSegmentator labels, category and polarity per sentence;
  * model Qwen3-32B-FP8, vLLM 0.17.1, temperature 0, run 2026-07-23;
  * only the lung-lobe labels are used.

These files contain text from ReXGroundingCT (https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT) and
are distributed under its licence, CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike). They are not
covered by the Apache-2.0 licence of the code.
