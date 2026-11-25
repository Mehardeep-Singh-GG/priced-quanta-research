# Archived Diagnostics (Excluded from Paper Claims)

`m16_30b_calibration_results.json` is preserved solely as a transparent diagnostic archive and is explicitly excluded from all paper claims (Section 5.2, Diagnostic Note).

Forensic audit findings:
- Rungs 2-4 added unaccounted full-precision truncated SVD residual factors (`LowRankOutlierAbsorber`, ranks 48, 160, and 320 -- exceeding the 256-row height of `k_proj`/`v_proj` and rendering them lossless FP32).
- Calibration and evaluation drew overlapping random integer token IDs rather than held-out text.
- The forward pass applied a synthetic `+0.18 * |sw|` residual modification.
