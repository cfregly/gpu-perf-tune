"""Tests for the generalized analytical roofline math (roofline_math.py)."""

from tools.perf_tune_report import roofline_math as rm

RIDGE_GB300_NVFP4 = (15.0 * 1e15) / (8.0 * 1e12)  # 1875 FLOP/byte


def test_quant_and_kv_bytes():
    assert abs(rm.quant_bytes_per_param("NVFP4") - 0.5625) < 1e-6
    assert rm.quant_bytes_per_param("FP8") == 1.0
    assert rm.quant_bytes_per_param("BF16") == 2.0
    assert rm.quant_bytes_per_param("unknown") == 2.0  # safe default
    assert rm.kv_bytes_per_elem("fp8_e4m3") == 1.0
    assert rm.kv_bytes_per_elem("bf16") == 2.0


def test_glm51_registry_matches_exemplar():
    """The GB300 GLM-5.1 chart shown to a stakeholder: decode AI ~3.4 at c=1 climbing
    to ~40 at c=192, ridge 1875, decode always memory-bound (left of ridge)."""
    s = rm.shape_for_model("zai-org/GLM-5.1")
    assert s is not None
    assert s.is_moe and s.is_mla

    ai_c1 = s.decode_arithmetic_intensity(1, 512, "NVFP4", "fp8")
    ai_c192 = s.decode_arithmetic_intensity(192, 512, "NVFP4", "fp8")
    assert 3.0 < ai_c1 < 4.0          # exemplar 3.4
    assert 30.0 < ai_c192 < 45.0      # exemplar ~40
    # decode is memory-bound across the whole feasible serving range
    for c in (1, 2, 4, 8, 16, 32, 64, 128, 192):
        assert s.decode_arithmetic_intensity(c, 512, "NVFP4", "fp8") < RIDGE_GB300_NVFP4

    # prefill climbs toward the ridge as the chunk grows (compute regime)
    p512 = s.prefill_arithmetic_intensity(512, "NVFP4")
    p8192 = s.prefill_arithmetic_intensity(8192, "NVFP4")
    assert p8192 > p512 > ai_c192     # prefill AI >> decode AI
    assert p8192 > 1000               # near/above the ridge

    # FLOP/token is positive and the active-weight bytes amortize with the union
    assert s.flop_per_token > 0
    assert s.active_weight_bytes(256, "NVFP4") > s.active_weight_bytes(8, "NVFP4")


def test_decode_ai_monotonic_in_concurrency():
    s = rm.shape_for_model("zai-org/GLM-5.1")
    ais = [s.decode_arithmetic_intensity(c, 512, "NVFP4", "fp8") for c in (1, 2, 4, 8, 16, 32, 64, 128, 192)]
    assert ais == sorted(ais)  # AI grows with concurrency (weight-byte amortization)


def test_from_hf_config_dense_gqa():
    """A Llama-3.1-8B-style dense GQA config -> dense shape, KV = 2*kv_heads*head_dim."""
    cfg = {
        "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 8, "vocab_size": 128256, "intermediate_size": 14336,
    }
    s = rm.from_hf_config(cfg, name="meta-llama/Llama-3.1-8B")
    assert not s.is_moe and not s.is_mla
    assert s.kv_elems_per_token_per_layer() == 2 * 8 * 128  # GQA: 8 kv heads * 128 dim * (K+V)
    # ~8B params for an 8B model (weight-GEMM budget, no embeddings double count)
    total_b = s.active_params(0) / 1e9
    assert 6.0 < total_b < 12.0
    # decode is memory-bound; prefill climbs
    assert s.decode_arithmetic_intensity(1, 1024, "BF16", "fp8") < s.prefill_arithmetic_intensity(4096, "BF16")


def test_from_hf_config_detects_moe_and_mla():
    cfg = {
        "hidden_size": 7168, "num_hidden_layers": 61, "num_attention_heads": 128,
        "vocab_size": 129280, "intermediate_size": 18432,
        "n_routed_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
        "n_shared_experts": 1, "first_k_dense_replace": 3,
        "kv_lora_rank": 512, "q_lora_rank": 1536, "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64, "v_head_dim": 128,
    }
    s = rm.from_hf_config(cfg, name="deepseek-ai/DeepSeek-V3")
    assert s.is_moe and s.is_mla
    assert s.n_routed_experts == 256 and s.n_experts_per_tok == 8
    assert s.kv_elems_per_token_per_layer() == 512 + 64  # MLA latent
    assert s.decode_arithmetic_intensity(1, 512, "NVFP4", "fp8") < RIDGE_GB300_NVFP4


def test_summary_roundtrip():
    s = rm.shape_for_model("zai-org/GLM-5.1")
    summ = s.to_summary("NVFP4", "fp8")
    assert summ["is_moe"] and summ["flop_per_token"] > 0
    d = rm.shape_to_dict(s)
    s2 = rm.shape_from_dict(d)
    assert s2.decode_arithmetic_intensity(8, 512, "NVFP4", "fp8") == s.decode_arithmetic_intensity(8, 512, "NVFP4", "fp8")
