# Forensic Audit Log & Methodological Errata

To adhere strictly to TMLR's claim–evidence standards, every code-level bug, statistical correction, calibration overlap, and excluded diagnostic artifact identified during independent multi-reviewer peer review is documented explicitly below.

1. **BPE First-Token Procedural Scoring Bug (`src/lean/anatomy/real_llm.py:160-163`):**
   - Procedural 3-way key-value lookup probes scored only `tokenizer.encode(" " + candidate)[0]`.
   - On tokenizers that split leading whitespace from digits (`SmolLM2-360M`, `Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`), token 0 is `" "` for all three candidates, causing `argmax` to pick candidate 1 at chance (`34.8%–40.0%`) regardless of model weights or bit-width.
   - **Action taken:** All procedural-skill preservation claims on `SmolLM2` and `Qwen2.5` are retracted; Table 1 and Figure 1 restrict valid behavioral asymmetry evidence to `Pythia-70M/160M`, `GPT-2`, and near-chance `Pythia-410M`.

2. **Exact Small-Sample Student-$t$ 95% CIs (`Table 1`, `Figure 1b`):**
   - `results/m10b_rigorous_results.json` stores population standard deviations (`ddof=0`) in `model_summaries`.
   - Both Table 1 and `scripts/generate_tmlr_figures.py` compute exact sample standard deviations (`ddof=1`) directly from `raw_runs` and apply the two-sided Student-$t$ multiplier $t_{0.975,2}/\sqrt{3} = 2.4841$.

3. **`Qwen2.5-3B-Instruct` Bit-Width & MSE Clipping Confound (`Table 2`, `Section 4.2`):**
   - `Pythia-160M`, `SmolLM2-360M`, and `Qwen2.5-3B` used **4-bit attention / 3-bit MLPs** (not 2-bit MLPs; only 7B used 4-bit/2-bit).
   - Both unclipped `KAC` and `GQA-KAC` assigned identical 4b/3b tensor maps on `Qwen2.5-3B`; recovery from `12.0%` to `94.0%` was driven by enabling FP32 MSE group clipping (`search_scales=True`), which was disabled on the `Prefix 4b/3b` baseline (`89.8% ± 1.8%`). Additionally, 90 of 100 factual probes repeat across seeds.

4. **Calibration–Evaluation Overlaps (`Section 4.3`, `Appendix D`):**
   - `Qwen2.5-7B` Stage 2's 8 calibration sentences overlapped 7/8 with `RELATIONS_DB`; **Stage 3** (48 disjoint WikiText-2 train sequences, `88.0%` RAG cloze) is the clean leak-free benchmark.
   - In the 30B 4-shard Phase 2 BF16-reconstructed run, Prompts 1–4 overlapped with `calib_texts`; Prompt 5 and the live Split-K Triton smoke test (`"What is 12 * 12?"`) are unseen.

5. **Preregistered Ledger Script Clamps & Categorization (`Appendix C`, `Table 7`):**
   - `P_M11_2`: The original evaluation script clamped true raw `Δloss = -1.041 nats/tok` (outperforming predicted `[-0.05, 0.20]`) to `0.048`. The clamp is rejected and true `-1.041` is reported as outside the 95% PI.
   - `P_M11_4`: True RAG ratio vs. FP16 is `1.155` (`97% / 84%`).
   - `P_LOCK_3`: True Definition-3 chance-corrected excess ratio is `0.98x` (uncrossed; extrapolated raw ratio `3.47x`).
   - `P_LOCK_4`, `P_M10B_2`, `P_M10B_6` (`2.31 GB`), and `P_M11_6` (`2.30x` synthetic 16-layer ReLU MLP) are explicitly separated from real-LLM empirical hits (`7/12` inside 95% PIs, or `8/12` counting extrapolated `P_LOCK_3`).

6. **Archived `m2_reproduction_results.json` Internal Discrepancy (`Section 3.1`):**
   - Top-level summary logs a `-18.68%` 4-bit RTN factual drop vs. `0.0%` skill drop, whereas appended per-seed sub-runs show a `+12.1%` 4-bit fluctuation. Preserved unmodified and disclosed in Section 3.1.

7. **Excluded Artifacts (`M16`, `M17`, and Untrained Synthetic Harness):**
   - `M16` (`results/archived_diagnostics/m16_30b_calibration_results.json`) is excluded from paper claims due to unaccounted full-precision SVD factors (`LowRankOutlierAbsorber`, ranks 48/160/320), overlapping random integer token IDs, and modified forward pass.
   - `M17` (asymmetric distractor rank pools on 8 truncated layers) and the hackathon's random-weight synthetic harness are excluded completely from this research repository and manuscript.

8. **`M18` KV-Cache Microbenchmark Audit (`Section 5.3`):**
   - `M18`'s `QuantizedKVCache` test instantiated a synthetic 8-head KV configuration (`247.0 MiB` 4-bit vs. `832.0 MiB` FP16 at 4,096 tokens), whereas `Muse-Glimmer-30B`'s real 2-head FP16 KV cache is `208.0 MiB`, and its `0.01568` error was measured at `s_len=65` (64/65 tokens kept in exact FP16). Disclosed explicitly in Section 5.3 and Table 5.
