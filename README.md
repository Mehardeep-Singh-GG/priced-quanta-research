# Priced-Quanta: Decoupling Algorithmic Routing from Associative Memory for Sub-4-Bit and 8 GiB Edge Language Model Serving

[![Paper Format](https://img.shields.io/badge/Format-TMLR%20Style-0072B2?style=flat-square)](paper/priced_quanta_tmlr_preprint.pdf)
[![Reproducibility](https://img.shields.io/badge/Raw%20JSON%20Audit-100%25%20Verified-009E73?style=flat-square)](scripts/verify_paper_numbers.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-gray?style=flat-square)](LICENSE)

Research paper, LaTeX source, vector figures, preregistrations, verified raw experimental JSON logs, and automated forensic verification scripts for **Priced-Quanta / Project LEAN** — separated cleanly from the [Meta Hackathon engineering/demo repository](https://github.com/Mehardeep-Singh-GG/lean-meta-hackathon).

> **Paper Artifacts (Official TMLR Style, 25 pages):**
> - **Preprint PDF (`[preprint]` mode):** [`paper/priced_quanta_tmlr_preprint.pdf`](paper/priced_quanta_tmlr_preprint.pdf)
> - **Anonymous Double-Blind Submission PDF (`\usepackage{tmlr}`):** [`paper/priced_quanta_tmlr_submission.pdf`](paper/priced_quanta_tmlr_submission.pdf)
> - **Plain-Text Export:** [`paper/priced_quanta_tmlr_paper.txt`](paper/priced_quanta_tmlr_paper.txt)
> - **LaTeX Source & BibTeX:** [`paper/main.tex`](paper/main.tex), [`paper/main_preprint.tex`](paper/main_preprint.tex), [`paper/references.bib`](paper/references.bib)
> - **Quantized 30B Weights Release (4 `.safetensors` shards, `8.57 GiB`):** hosted on [`lean-meta-hackathon` v1.0.0](https://github.com/Mehardeep-Singh-GG/lean-meta-hackathon/releases/tag/v1.0.0)

---

## Abstract Summary

Uniform post-training quantization assigns identical bit-widths to attention routing projections and feed-forward MLP blocks, despite mechanistic evidence that MLPs primarily store static associative facts while attention executes context routing. We investigate this asymmetry theoretically and empirically across pretrained transformers spanning **70M to 29.78B parameters**:

- **Theory (Section 2 & Appendix A):** Corrected Wedin $\sin\Theta$ singular-subspace stability bound, Cauchy–Schwarz argmax routing margin condition, first-order linearized MLP associative noise characterization (Theorem 1, Lemma 1), and LP-relaxation optimality gap for efficiency-sorted greedy Multiple-Choice Knapsack allocation (Proposition 1).
- **Controlled Scaling & Preregistered Blind Ledger (Section 3 & Appendix C):** Synthetic multi-world scaling verification (`m3`–`m9`) and a forensically audited ledger of all 18 preregistered blind predictions (`M10`, `M10B`, `M11`: `7/12` valid empirical hits inside 95% PIs, `8/12` counting extrapolated `P_LOCK_3`, plus 2 BPE-invalidated and 4 analytical/synthetic entries).
- **Cross-Family Survival & Byte-Parity 4b/3b Evaluation (Sections 4.1–4.2):** Transformer-bit half-life asymmetry ratios of `1.20x–5.30x` across valid checkpoints (`Pythia-70M/160M`, `GPT-2`, near-chance `Pythia-410M`; exact `ddof=1` Student-$t$ 95% CIs); unconfounded 4b/3b backbone byte-parity gains on `Pythia-160M` (`+5.5 pp`, `-1.50 nats`) and `SmolLM2-360M` (`+8.0 pp` raw cloze, `+1.0 pp` RAG cloze); and MSE-optimal group clipping recovery on `Qwen2.5-3B-Instruct` (`12.0% -> 94.0%` vs `89.8%` unclipped prefix baseline, exact two-sided permutation `p = 0.0625`, explicitly disclosing the clipping confound and closed-book trade-off).
- **Physical 8.00 GiB RTX 4060 7B Serving (Section 4.3):** 4-bit/2-bit AWQ-calibrated `GQA-KAC` packs `Qwen2.5-7B-Instruct` (`14.19 GiB` FP16) into `3.84 GiB` weights (`4.34 GiB` allocated VRAM) and lifts retrieval cloze from `12.0%` to `88.0%` using 48 disjoint WikiText-2 calibration sequences (with gentler AWQ exponent `alpha = 0.4` winning in `74/84 = 88.1%` of SwiGLU projections).
- **30B Anatomy, Block-Cholesky Hessian Compensation, and Split-K Triton Serving (Section 5):** Full 52-layer spectral profiling of `meta-models/Muse-Glimmer-30B` (`29.78B` params, `55.46 GiB` BF16), a `6.0805 GiB` (`1.7541 bpp`) greedy Lagrangian blueprint, `-47.0%` macro calibration output activation $L_2$ error reduction across 8 real sliced 30B projections (`m15`), and a live 52-layer deferred-reduction Split-K fused Triton daemon running beneath an `8.00 GiB` GPU cap (`7.29 GiB` PyTorch / `7.92 GiB` `nvidia-smi` + `~3.26 GiB` host CPU RAM offload, `9.0 tok/s`).

---

## Forensic Transparency & Scientific Scope

This repository intentionally **excludes** non-rigorous or confounded hackathon artifacts (specifically synthetic random-weight probes and `M17`'s asymmetric-distractor proxy) and documents every methodological limitation, calibration overlap, tokenizer artifact, and script clamp uncovered during our multi-round independent peer-review audit in [`AUDIT_ERRATA.md`](AUDIT_ERRATA.md).

---

## Repository Structure

```text
priced-quanta-research/
├── paper/
│   ├── main.tex                              # Double-blind anonymous TMLR source
│   ├── main_preprint.tex                     # De-anonymized [preprint] TMLR source
│   ├── priced_quanta_tmlr_submission.pdf     # 25-page anonymous submission PDF
│   ├── priced_quanta_tmlr_preprint.pdf       # 25-page preprint PDF
│   ├── priced_quanta_tmlr_paper.txt          # Plain-text export of the full paper
│   ├── references.bib                        # 58 verified BibTeX references
│   ├── tmlr.sty, tmlr.bst, math_commands.tex # Official TMLR style files (unmodified)
│   └── figures/                              # 5 vector PDF figures generated from raw JSONs
├── results/                                  # Verified raw JSON logs & Colab worker logs
│   ├── colab_jobs/                           # Sanitized A100 52-layer export & Triton logs
│   └── archived_diagnostics/                 # Excluded M16 diagnostic + forensic notice
├── prereg/                                   # Sealed preregistrations (LOCK_v1, M4-M11, M16)
├── scripts/
│   ├── generate_tmlr_figures.py              # Regenerates all 5 vector PDFs from results/
│   └── verify_paper_numbers.py               # Automated audit asserting paper numbers vs JSONs
├── src/lean/                                 # Core compression, anatomy, theory & runtime code
├── experiments/                              # Verified benchmark & calibration scripts
├── AUDIT_ERRATA.md                           # Full forensic audit log & errata
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

---

## Quick Reproduction

### 1. Verify Every Paper Number Against Raw JSONs
```bash
pip install -r requirements.txt
python3 scripts/verify_paper_numbers.py
```

### 2. Regenerate All 5 Vector PDF Figures Directly From Raw Data
```bash
python3 scripts/generate_tmlr_figures.py
```

### 3. Recompile Both TMLR PDFs (Anonymous & Preprint)
Using [Tectonic](https://tectonic-typesetting.github.io/) (or standard `pdflatex` + `bibtex`):
```bash
cd paper
tectonic main.tex           # Anonymous double-blind submission PDF
tectonic main_preprint.tex  # De-anonymized preprint PDF
```

---

## Citation

```bibtex
@misc{singh2026pricedquanta,
  title  = {Priced-Quanta: Decoupling Algorithmic Routing from Associative Memory for Sub-4-Bit and 8 GiB Edge Language Model Serving},
  author = {Mehardeep Singh and Gurarpit Singh},
  year   = {2026},
  url    = {https://github.com/Mehardeep-Singh-GG/priced-quanta-research}
}
```
