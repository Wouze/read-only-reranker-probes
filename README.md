# Qwen3-Reranker-4B, twice as fast

Read-only probes score a frozen Qwen3-Reranker-4B part-way through, drop most passages, and let the original model finish the shortlist. The weights are a few small checkpoints. The 4B model stays `Qwen/Qwen3-Reranker-4B`.

**2× speed, measured.** On one RTX 3090 the half-latency schedule takes 1,644 ms against a 3,276 ms prefix-cached forward, at an nDCG@10 drop of +0.00048. The reference top 10 is still there on 0.80 of a 400-query check split.

**About 4× less work than a normal transformer forward.** A stock Hugging Face forward repeats the query on every passage and projects the full vocabulary. On the pilot lists that costs 1.73× the prefix-cached forward in multiply-accumulates. The compacted schedule is 2.29× that same forward in wall-clock (1,430 ms, drop +0.0012). 1.73 × 2.29 = 3.96, about 4×. The 2× is the time to quote. The 4× is that product, so the two measurements stay labeled.

The paper is in [`paper/`](paper/). Every row is in [`results/schedules.json`](results/schedules.json). The checkpoints are at [huggingface.co/Wouze/read-only-reranker-probes](https://huggingface.co/Wouze/read-only-reranker-probes).

## Schedules

The cut rule was fixed before any 4B probe score existed: keep 60 at layer 9, 40 at layer 15, and 15 at a layer-21 ridge. Its file is `data/fast06/paper_r1.json`, SHA-256 `750cc3641a728eac51426765e8db15a014ab71e314e0ccf55540ccd45996b43e`.

| Schedule | What changes | Code |
|---|---|---|
| Exact | The 40 layer-15 survivors restart from layer 0, so the final margin is the joint bf16 margin. 1.09×, drop +0.0002. | `Cascade` in `src/paper_eval.py` |
| Tier A | The suffix continues from the document cache. Nine-token probe: 1.47×, drop +0.0012. One-token probe: 1.52×, drop +0.0020. | `Catchup` in `src/next_exps.py` |
| Tier B | Layers 0–15 see the first 192 tokens, the tail is restored, the document freezes after layer 25, and a layer-25 ridge blend (α = 0.15) corrects the margin. 1.99×, drop +0.00048. | `MergedTail` in `src/week3_fast_tail.py` |
| Tier C | The same schedule with QKV, the output projection, and the MLP gathered onto real tokens, and the blend removed. 2.29×, drop +0.00123. | `paper_eval.COMPACT = "all"` |

Head-192, TailCatch, Tier B, and Tier C use the one-token probe (`q4_p1.pt`). The nine-token probe is `q4_full.pt`. The 1.99× and 2.29× ratios compare eight check queries with a 3,276 ms baseline from a separate ten-query pilot profile.

## Weights

```bash
hf download Wouze/read-only-reranker-probes --local-dir data/fast06
```

That places `probes/q4_p1.pt`, `probes/q4_full.pt`, `probes/ridge_4b_l21.pt`, and `week3_f25_ridge_beta.pt` where the code looks. The base model stays `Qwen/Qwen3-Reranker-4B` and is downloaded by transformers. `data/manifest.json` holds the prompt pins (yes token 9693, no token 2152).

```bash
pip install -r requirements.txt
```

A CUDA GPU is required. The 4B model in bf16 needs about 8 GB, plus activation memory for the passage batch.

## What is not in this repository

The check and pilot passage lists, the per-query rankings, the Arabic dev lists, the TREC DL19 lists, and the 261-query holdout. Dev and DL19 were scored once, under the exact schedule only, after the rule was hashed. Both 95% intervals contain zero. The holdout has not been read.

## License

The code and these checkpoints are Apache 2.0. `Qwen/Qwen3-Reranker-4B` is a separate model under its own Apache 2.0 license. The paper text is CC BY 4.0.
