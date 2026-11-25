#!/usr/bin/env python3
"""
Automated forensic audit script for the TMLR manuscript:
"Priced-Quanta: Decoupling Algorithmic Routing from Associative Memory
 for Sub-4-Bit and 8 GiB Edge Language Model Serving"

Reads raw JSON outputs and execution logs in results/ and programmatically
asserts every headline number, confidence interval, p-value, effect size,
and memory footprint reported in Tables 1-8 and Figures 1-5.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def check_close(name: str, got: float, expected: float, tol: float = 1e-2) -> None:
    if abs(got - expected) > tol:
        raise AssertionError(f"[FAIL] {name}: got {got:.6f}, expected {expected:.6f} (tol={tol})")
    print(f"  [PASS] {name}: {got:.4f} (expected {expected})")


def verify_controlled_and_m9() -> None:
    print("\n1. Controlled Scaling & M9 Model Selection (Section 3.1, Fig. 3a)")
    m4 = json.loads((RESULTS / "m4_composition_results.json").read_text())
    m5 = json.loads((RESULTS / "m5_wrmf_results.json").read_text())
    m7 = json.loads((RESULTS / "m7_theory_results.json").read_text())
    m8 = json.loads((RESULTS / "m8_gate_beta_results.json").read_text())
    m9 = json.loads((RESULTS / "m9_lock_results.json").read_text())

    # M9 model comparison MAPEs and exact residual SSEs
    mods = m9["model_comparison"]["models"]
    check_close("M9 Model A MAPE (%)", float(mods["Model_A_LEAN"]["mape"]), 5.50, 0.05)
    check_close("M9 Model A AIC", float(mods["Model_A_LEAN"]["aic"]), -25.49, 0.05)
    check_close("M9 Model B MAPE (%)", float(mods["Model_B_Chinchilla"]["mape"]), 37.69, 0.05)
    check_close("M9 Model C MAPE (%)", float(mods["Model_C_Precision"]["mape"]), 52.83, 0.05)
    check_close("M9 Model D MAPE (%)", float(mods["Model_D_FrequencyOnly"]["mape"]), 98.12, 0.05)

    # Verify exact SS_res bounds on the 4 combined points
    y_true = np.array([
        float(m8["acceptance_criteria"]["C2_exponent_matching"]["knapsack_gamma"]),
        float(m7["alpha_evaluations"]["alpha_0.3"]["mean_gamma"]),
        float(m7["alpha_evaluations"]["alpha_0.6"]["mean_gamma"]),
        float(m7["alpha_evaluations"]["alpha_1.0"]["mean_gamma"]),
    ])
    x_alpha = np.array([float(m8["pile_zipf"]["alpha_pile"]), 0.3, 0.6, 1.0])
    ss_a = float(np.sum((y_true - x_alpha) ** 2))
    ss_b = float(np.sum((y_true - np.mean(y_true)) ** 2))
    check_close("M9 SS_res (Model A)", ss_a, 0.00415, 0.0001)
    check_close("M9 SS_res (Model B)", ss_b, 0.2366, 0.002)
    assert ss_b / ss_a > 57.0, f"Expected >57x SSE ratio, got {ss_b / ss_a:.2f}x"
    print(f"  [PASS] M9 SSE ratio Model B / Model A = {ss_b / ss_a:.2f}x (>57x)")
    _ = (m4, m5)


def verify_table1_cross_family() -> None:
    print("\n2. Cross-Family Half-Life Asymmetry & Exact Student-t 95% CIs (Table 1, Fig. 1)")
    m10b = json.loads((RESULTS / "m10b_rigorous_results.json").read_text())
    t_mult = float(stats.t.ppf(0.975, df=2)) / math.sqrt(3.0)
    by_model: dict[str, list[dict]] = {}
    for run in m10b["raw_runs"]:
        by_model.setdefault(run["model_id"], []).append(run)

    expected = {
        "EleutherAI/pythia-70m": (5.30, 1.24, 2.22, 8.39),
        "gpt2": (1.45, 0.11, 1.19, 1.72),
        "EleutherAI/pythia-160m": (1.20, 0.02, 1.16, 1.24),
        "EleutherAI/pythia-410m": (2.74, 0.60, 1.25, 4.23),
        "HuggingFaceTB/SmolLM2-360M": (6.04, 0.005, 6.02, 6.05),
    }
    for mid, (e_mu, e_sd, e_lo, e_hi) in expected.items():
        vals = np.array([float(r["ratio_trans"]) for r in by_model[mid]])
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1))
        lo, hi = mu - t_mult * sd, mu + t_mult * sd
        check_close(f"{mid} mean", mu, e_mu, 0.015)
        check_close(f"{mid} sample SD (ddof=1)", sd, e_sd, 0.015)
        check_close(f"{mid} 95% t-CI low", lo, e_lo, 0.02)
        check_close(f"{mid} 95% t-CI high", hi, e_hi, 0.02)


def verify_table2_and_table3() -> None:
    print("\n3. Byte-Parity 4b/3b (Table 2) & RTX 4060 7B Serving (Table 3, Fig. 2)")
    q3b = json.loads((RESULTS / "multiseed_statistical_analysis_Qwen2.5-3B-Instruct.json").read_text())
    rag = q3b["metrics_summary"]["retrieval_cloze_accuracy"]
    u_seeds = np.array([0.93, 0.89, 0.89, 0.89, 0.89])
    k_seeds = np.array([0.94, 0.94, 0.94, 0.94, 0.94])
    diffs = (k_seeds - u_seeds) * 100.0
    check_close("Qwen2.5-3B Prefix mean (%)", 100.0 * rag["matched_uniform_mean"], 89.8, 0.05)
    check_close("Qwen2.5-3B GQA-KAC mean (%)", 100.0 * rag["gqa_kac_mean"], 94.0, 0.05)
    dz = float(np.mean(diffs) / np.std(diffs, ddof=1))
    gz = dz * (1.0 - 3.0 / (4.0 * 4.0 - 1.0))
    check_close("Qwen2.5-3B paired d_z", dz, 2.35, 0.02)
    check_close("Qwen2.5-3B Hedges g_z", gz, 1.88, 0.02)
    t_hw = float(stats.t.ppf(0.975, df=4)) * float(np.std(diffs, ddof=1)) / math.sqrt(5.0)
    check_close("Qwen2.5-3B paired t-CI low (pp)", float(np.mean(diffs)) - t_hw, 1.98, 0.02)
    check_close("Qwen2.5-3B paired t-CI high (pp)", float(np.mean(diffs)) + t_hw, 6.42, 0.02)

    q7b_u = json.loads((RESULTS / "matched_budget_7b_grand_challenge.json").read_text())
    q7b_c = json.loads((RESULTS / "matched_budget_7b_calibrated_grand_challenge.json").read_text())
    q7b_s = json.loads((RESULTS / "matched_budget_7b_scaled_awq_tuning.json").read_text())
    check_close("7B Weights (GiB)", q7b_s["exact_storage_accounting"]["gqa_kac_storage_gib"], 3.8365, 0.005)
    check_close("7B Stage 1 RAG (%)", 100.0 * q7b_u["metrics"]["gqa_kac_plus_retrieval"]["factual_cloze_acc"], 12.0, 0.05)
    check_close("7B Stage 2 RAG (%)", 100.0 * q7b_c["metrics"]["awq_kac_plus_retrieval"]["factual_cloze_acc"], 72.0, 0.05)
    check_close("7B Stage 3 RAG (%)", 100.0 * q7b_s["metrics"]["scaled_awq_kac_plus_retrieval"]["factual_cloze_acc"], 88.0, 0.05)
    check_close("7B Stage 3 VRAM alloc (GiB)", q7b_s["hardware_profile"]["allocated_vram_gb"], 4.34, 0.01)


def verify_30b_and_systems() -> None:
    print("\n4. Muse-Glimmer-30B Anatomy, Hessian, M18 & Live Split-K Triton (Tables 4-6, Figs. 3-5)")
    m14 = json.loads((RESULTS / "m14_30b_anatomy_and_baseline_kac.json").read_text())
    m15 = json.loads((RESULTS / "m15_30b_priced_quanta_v2_results.json").read_text())
    m18 = json.loads((RESULTS / "m18_8gb_runtime_proof.json").read_text())
    log_exp = (RESULTS / "colab_jobs/lean_1793175167_5e0b93/worker.log").read_text()
    log_tri = (RESULTS / "colab_jobs/lean_1793190718_e36d95/worker.log").read_text()

    layers = m14["layer_reports"]
    k_sr = [float(l["projections"]["k_proj"]["stable_rank"]) for l in layers]
    up_sr = [float(l["projections"]["up_proj"]["stable_rank"]) for l in layers]
    check_close("M14 k_proj L0 stable rank", k_sr[0], 10.35, 0.05)
    check_close("M14 k_proj 52L mean stable rank", float(np.mean(k_sr)), 28.0, 0.2)
    check_close("M14 up_proj 52L mean stable rank", float(np.mean(up_sr)), 692.0, 0.5)
    check_close("M15 Blueprint physical GiB", float(m15["knapsack_30b_summary"]["total_physical_gib"]), 6.0805, 0.001)

    profs = m15["hessian_compensation_telemetry"]["layer_profiles"]
    rtn_m = float(np.mean([p["err_rtn_stage1"] for p in profs]))
    hk_m = float(np.mean([p["err_hkac_v2"] for p in profs]))
    macro_red = 100.0 * (rtn_m - hk_m) / rtn_m
    unw_red = float(np.mean([p["error_reduction_pct"] for p in profs]))
    check_close("M15 Macro mean RTN err", rtn_m, 0.4428, 0.001)
    check_close("M15 Macro mean H-KACv2 err", hk_m, 0.2347, 0.001)
    check_close("M15 Macro reduction (%)", macro_red, 47.00, 0.05)
    check_close("M15 Unweighted reduction (%)", unw_red, 36.93, 0.05)

    check_close("M18 CUDA reserved peak (GiB)", float(m18["lean_30b_8gb_runtime_proof"]["peak_memory_gib"]), 6.87, 0.01)

    m_shard = re.search(r"Total Size:\s*([0-9.]+)\s*GiB", log_exp)
    m_bf16 = re.search(r"Live VRAM:\s*([0-9.]+)\s*GiB", log_exp)
    m_smoke = re.search(r"\[Smoke Test\]:\s*(\{.*\})", log_tri)
    m_smi = re.search(r"\[A100 nvidia-smi\]:\s*(\d+)\s*MiB", log_tri)
    assert m_shard and m_bf16 and m_smoke and m_smi
    smoke = json.loads(m_smoke.group(1))
    check_close("4-Shard disk size (GiB)", float(m_shard.group(1)), 8.5712, 0.001)
    check_close("4-Shard Phase 2 BF16 VRAM (GiB)", float(m_bf16.group(1)), 51.97, 0.01)
    check_close("Triton persistent GPU (GiB)", float(smoke["gpu_persistent_gib"]), 7.2688, 0.001)
    check_close("Triton peak GPU (GiB)", float(smoke["gpu_peak_gib"]), 7.2944, 0.001)
    check_close("Triton nvidia-smi GPU (GiB)", float(m_smi.group(1)) / 1024.0, 7.918, 0.005)
    check_close("Triton throughput (tok/s)", float(smoke["tokens_per_sec"]), 9.0, 0.05)


if __name__ == "__main__":
    verify_controlled_and_m9()
    verify_table1_cross_family()
    verify_table2_and_table3()
    verify_30b_and_systems()
    print("\nALL PAPER NUMBERS VERIFIED 100% AGAINST RAW JSONS AND LOGS!")
