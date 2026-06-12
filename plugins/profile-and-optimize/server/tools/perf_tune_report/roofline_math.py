"""Generalized analytical roofline math for the prefill/decode roofline (page 7).

This is the model-agnostic generalization of the GLM-5.1-only
``perf-tune-glm51/.../driver/roofline_math.py`` probe. It turns a HuggingFace
``config.json`` into the two analytical quantities a Williams roofline needs:

- **arithmetic intensity (FLOP/byte)** -- the x-axis. Derived from the model's
  shapes + the serving point (decode concurrency / prefill chunk length). This is
  ANALYTICAL by construction (a property of the algorithm + shapes), exactly like
  every published roofline; it is NOT a measured number and is cross-checked
  against the measured DCGM tensor/dram active ratio in the renderer.
- **FLOP per token** -- multiplied by the MEASURED tok/s (and divided by the GPU
  count) to get the achieved-compute y-axis. The perf number (tok/s) is measured;
  the FLOP/token coefficient is analytical. This split is the whole point: the
  roofline plots a *measured* operating point against an *analytical* intensity
  and a *datasheet* ceiling (see ``ROOFLINE-METHODOLOGY.md``).

Supported architecture families (auto-detected from config keys):

- dense transformers (Llama/Qwen/Gemma/Phi/Nemotron-style), MHA or GQA;
- MoE transformers (DeepSeek/GLM/Qwen-MoE/MiniMax-style) with shared + routed
  experts and an optional leading block of dense layers;
- MLA latent attention (DeepSeek/GLM ``kv_lora_rank`` style) vs standard
  KV-head attention (GQA/MHA).

Everything is per-GPU normalizable: divide achieved FLOP/s and delivered bytes/s
by the tensor-parallel GPU count so TP2/TP4/TP8 share one set of per-GPU ceilings.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

# ---------------------------------------------------------------------------
# Bytes-per-parameter by weight quant. NVFP4 = 4-bit weight + an fp8 group-16
# scale (~0.5 + 1/16 B/param); FP8 = 1 B; BF16/FP16 = 2 B. The lm_head is
# (almost) always left in bf16 even when the body is quantized, so it is
# accounted separately at BF16.
# ---------------------------------------------------------------------------
QUANT_BYTES_PER_PARAM: dict[str, float] = {
    "NVFP4": 4.0 / 8.0 + 1.0 / 16.0,  # ~0.5625
    "FP4": 4.0 / 8.0 + 1.0 / 16.0,
    "FP8": 1.0,
    "INT8": 1.0,
    "BF16": 2.0,
    "FP16": 2.0,
    "FP32": 4.0,
}
LM_HEAD_BYTES_PER_PARAM = 2.0  # bf16 lm_head even under NVFP4/FP8 body quant

#: KV-cache dtype -> bytes/element. Our standing GB300 deploys use fp8 KV.
KV_BYTES_PER_ELEM: dict[str, float] = {
    "fp8": 1.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
    "int8": 1.0,
    "nvfp4": 0.5,
    "bf16": 2.0,
    "fp16": 2.0,
    "auto": 1.0,
}


def quant_bytes_per_param(quant: str) -> float:
    return QUANT_BYTES_PER_PARAM.get((quant or "BF16").upper(), 2.0)


def kv_bytes_per_elem(kv_dtype: str) -> float:
    return KV_BYTES_PER_ELEM.get((kv_dtype or "fp8").lower(), 1.0)


@dataclass
class ModelShape:
    """Architecture shapes needed for the analytical roofline.

    Only the fields the roofline needs; populated from a HF ``config.json`` by
    :func:`from_hf_config` or hand-built for the test/registry path.
    """

    name: str
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    vocab_size: int
    intermediate_size: int          # dense-MLP intermediate (gated SwiGLU: 3 matrices)
    num_kv_heads: int = 0           # GQA; 0 => MHA (== num_attention_heads)
    head_dim: int = 0               # 0 => hidden_size / num_attention_heads
    gated_mlp: bool = True          # SwiGLU (gate+up+down=3) vs (up+down=2)
    # Architecture-specific extra attention params per layer (e.g. the GLM-DSA
    # "lightning indexer" q/k projections). Generic families leave this 0.
    extra_attn_params_per_layer: float = 0.0

    # MoE
    is_moe: bool = False
    n_routed_experts: int = 0
    n_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    n_shared_experts: int = 0
    first_k_dense_replace: int = 0  # leading dense layers before MoE kicks in

    # MLA (DeepSeek/GLM latent attention). is_mla=False => standard GQA/MHA.
    is_mla: bool = False
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0

    # -- derived geometry -------------------------------------------------
    def _head_dim(self) -> int:
        return self.head_dim or (self.hidden_size // max(self.num_attention_heads, 1))

    def _num_kv_heads(self) -> int:
        return self.num_kv_heads or self.num_attention_heads

    @property
    def n_dense_layers(self) -> int:
        if self.is_moe:
            return min(self.first_k_dense_replace, self.num_layers)
        return self.num_layers

    @property
    def n_moe_layers(self) -> int:
        return self.num_layers - self.n_dense_layers if self.is_moe else 0

    # -- parameter budgets (per layer) -----------------------------------
    def _attn_params_per_layer(self) -> float:
        H = self.hidden_size
        if self.is_mla:
            n = self.num_attention_heads
            q_a = H * self.q_lora_rank if self.q_lora_rank else 0
            q_b = (self.q_lora_rank or H) * (n * (self.qk_nope_head_dim + self.qk_rope_head_dim))
            kv_a = H * (self.kv_lora_rank + self.qk_rope_head_dim)
            kv_b = self.kv_lora_rank * (n * (self.qk_nope_head_dim + self.v_head_dim))
            o = (n * self.v_head_dim) * H
            return float(q_a + q_b + kv_a + kv_b + o) + self.extra_attn_params_per_layer
        # standard MHA / GQA: q proj (H x n*d) + k,v proj (H x kv*d) + o proj
        d = self._head_dim()
        n = self.num_attention_heads
        kv = self._num_kv_heads()
        q = H * (n * d)
        k = H * (kv * d)
        v = H * (kv * d)
        o = (n * d) * H
        return float(q + k + v + o) + self.extra_attn_params_per_layer

    def _mlp_params(self, inter: int) -> float:
        mats = 3 if self.gated_mlp else 2
        return float(mats * self.hidden_size * inter)

    def _router_params(self) -> float:
        return float(self.hidden_size * self.n_routed_experts) if self.is_moe else 0.0

    @property
    def lm_head_params(self) -> float:
        return float(self.hidden_size * self.vocab_size)

    def active_params(self, experts_per_layer: float) -> float:
        """Active (touched) parameters for ``experts_per_layer`` routed experts
        engaged in each MoE layer. ``experts_per_layer == n_experts_per_tok``
        gives the per-token active count; ``== n_routed_experts`` gives the
        all-experts-engaged (prefill / large-batch) count."""
        p = self.num_layers * self._attn_params_per_layer()
        if self.is_moe:
            p += self.n_dense_layers * self._mlp_params(self.intermediate_size)
            shared = self.n_shared_experts * self._mlp_params(self.moe_intermediate_size)
            routed = experts_per_layer * self._mlp_params(self.moe_intermediate_size)
            p += self.n_moe_layers * (shared + self._router_params() + routed)
        else:
            p += self.num_layers * self._mlp_params(self.intermediate_size)
        p += self.lm_head_params
        return p

    def active_weight_bytes(self, experts_per_layer: float, quant: str) -> float:
        """Bytes of weights touched, body at ``quant`` + lm_head at bf16."""
        bpp = quant_bytes_per_param(quant)
        body = self.active_params(experts_per_layer) - self.lm_head_params
        return body * bpp + self.lm_head_params * LM_HEAD_BYTES_PER_PARAM

    def kv_elems_per_token_per_layer(self) -> float:
        """Cached KV elements per token per layer."""
        if self.is_mla:
            return float(self.kv_lora_rank + self.qk_rope_head_dim)  # latent
        return float(2 * self._num_kv_heads() * self._head_dim())     # K + V

    def kv_bytes_per_token(self, ctx_len: int, kv_dtype: str = "fp8") -> float:
        """Bytes of KV read per generated token (full context, all layers)."""
        return (
            ctx_len
            * self.num_layers
            * self.kv_elems_per_token_per_layer()
            * kv_bytes_per_elem(kv_dtype)
        )

    @property
    def flop_per_token(self) -> float:
        """2 x active params per token (weight-GEMM FLOPs; attention FLOPs are
        excluded -> a conservative lower bound on prefill achieved-compute, by
        construction, matching the GLM exemplar)."""
        experts = self.n_experts_per_tok if self.is_moe else 0
        return 2.0 * self.active_params(experts)

    # -- roofline operating-point intensities ----------------------------
    def decode_arithmetic_intensity(
        self, concurrency: int, ctx_len: int, quant: str, kv_dtype: str = "fp8"
    ) -> float:
        """FLOP/byte for a decode step at batch=concurrency, context=ctx_len.

        Weight bytes amortize across the batch (the expert union ``min(n_routed,
        n_experts_per_tok * c)`` is loaded once and reused by every token that
        routes to it); KV bytes scale per token with context. AI therefore grows
        ~linearly with concurrency but stays far left of the ridge for any
        feasible serving batch -- the memory-bound regime."""
        c = max(int(concurrency), 1)
        if self.is_moe:
            union = min(self.n_routed_experts, self.n_experts_per_tok * c)
        else:
            union = 0
        wb_per_tok = self.active_weight_bytes(union, quant) / c
        kv_per_tok = self.kv_bytes_per_token(ctx_len, kv_dtype)
        denom = wb_per_tok + kv_per_tok
        return self.flop_per_token / denom if denom > 0 else 0.0

    def prefill_arithmetic_intensity(self, chunk_tokens: int, quant: str) -> float:
        """FLOP/byte for a prefill chunk of ``chunk_tokens`` tokens.

        A large chunk engages ~all routed experts but loads each weight once and
        reuses it across the chunk's tokens, so AI ~ chunk_tokens -- the
        compute-bound regime (points climb toward the ceiling)."""
        t = max(int(chunk_tokens), 1)
        experts = self.n_routed_experts if self.is_moe else 0
        wb = self.active_weight_bytes(experts, quant)
        return self.flop_per_token / (wb / t) if wb > 0 else 0.0

    def to_summary(self, quant: str, kv_dtype: str = "fp8") -> dict[str, Any]:
        """Compact, self-describing analytical block to embed in
        ``roofline_sweep.json`` so the renderer is config-free at render time."""
        experts = self.n_experts_per_tok if self.is_moe else 0
        return {
            "model": self.name,
            "is_moe": self.is_moe,
            "is_mla": self.is_mla,
            "num_layers": self.num_layers,
            "flop_per_token": self.flop_per_token,
            "active_weight_bytes_per_tok_experts": self.active_weight_bytes(experts, quant),
            "kv_elems_per_token_per_layer": self.kv_elems_per_token_per_layer(),
            "kv_bytes_per_elem": kv_bytes_per_elem(kv_dtype),
            "quant": quant,
            "kv_dtype": kv_dtype,
            "total_params_b": round(self.active_params(
                self.n_routed_experts if self.is_moe else 0) / 1e9, 2),
        }


def _first(config: dict[str, Any], *keys: str, default: Any = 0) -> Any:
    for k in keys:
        if k in config and config[k] is not None:
            return config[k]
    return default


def from_hf_config(config: dict[str, Any], name: str = "") -> ModelShape:
    """Build a :class:`ModelShape` from a HuggingFace ``config.json`` dict.

    Tolerant of the common key-name variants across model families. Unknown
    architectures degrade to a dense transformer using the standard keys.
    """
    # Some configs nest the LM config under "text_config" (multimodal) or
    # "language_config"; unwrap if the top level lacks hidden_size.
    if "hidden_size" not in config:
        for sub in ("text_config", "language_config", "llm_config"):
            if isinstance(config.get(sub), dict) and "hidden_size" in config[sub]:
                config = {**config[sub], **{k: v for k, v in config.items() if k not in (sub,)}}
                break

    H = int(_first(config, "hidden_size", "n_embd", default=4096))
    L = int(_first(config, "num_hidden_layers", "n_layer", "num_layers", default=32))
    n_heads = int(_first(config, "num_attention_heads", "n_head", default=max(H // 128, 1)))
    n_kv = int(_first(config, "num_key_value_heads", "num_kv_heads", default=n_heads))
    head_dim = int(_first(config, "head_dim", default=0))
    vocab = int(_first(config, "vocab_size", default=128000))
    inter = int(_first(config, "intermediate_size", "ffn_dim", "n_inner", default=4 * H))

    # MoE detection
    n_routed = int(_first(config, "n_routed_experts", "num_experts",
                          "num_local_experts", "n_experts", default=0))
    n_per_tok = int(_first(config, "num_experts_per_tok", "num_experts_per_token",
                           "moe_topk", "n_experts_per_tok", default=0))
    moe_inter = int(_first(config, "moe_intermediate_size", "expert_intermediate_size",
                           default=0))
    n_shared = int(_first(config, "n_shared_experts", "num_shared_experts",
                          "moe_num_shared_experts", default=0))
    first_dense = int(_first(config, "first_k_dense_replace", "moe_layer_start_index",
                             default=0))
    is_moe = n_routed > 0 and n_per_tok > 0
    if is_moe and moe_inter == 0:
        moe_inter = inter  # fall back to dense intermediate if expert size absent

    # MLA detection (DeepSeek / GLM latent attention)
    kv_lora = int(_first(config, "kv_lora_rank", default=0))
    q_lora = int(_first(config, "q_lora_rank", default=0))
    qk_nope = int(_first(config, "qk_nope_head_dim", default=0))
    qk_rope = int(_first(config, "qk_rope_head_dim", default=0))
    v_head = int(_first(config, "v_head_dim", default=0))
    is_mla = kv_lora > 0 and qk_rope > 0

    # gated SwiGLU is the default for all modern families here; honor an
    # explicit hidden_act that is not a *glu variant only when clearly stated.
    act = str(_first(config, "hidden_act", "hidden_activation", default="silu")).lower()
    gated = True  # silu/gelu both used inside SwiGLU on these models
    _ = act

    return ModelShape(
        name=name or str(_first(config, "_name_or_path", "model_type", default="model")),
        hidden_size=H, num_layers=L, num_attention_heads=n_heads, num_kv_heads=n_kv,
        head_dim=head_dim, vocab_size=vocab, intermediate_size=inter, gated_mlp=gated,
        is_moe=is_moe, n_routed_experts=n_routed, n_experts_per_tok=n_per_tok,
        moe_intermediate_size=moe_inter, n_shared_experts=n_shared,
        first_k_dense_replace=first_dense,
        is_mla=is_mla, kv_lora_rank=kv_lora, q_lora_rank=q_lora,
        qk_nope_head_dim=qk_nope, qk_rope_head_dim=qk_rope, v_head_dim=v_head,
    )


# ---------------------------------------------------------------------------
# Built-in shape registry, keyed by a substring of the served model name. Lets
# the renderer re-create the analytical roofline for an EXISTING campaign that
# predates the embedded-analytical-block importer, with no config.json at hand.
# Shapes are from each family's published config.json (sources in
# ROOFLINE-METHODOLOGY.md). Extend as families are benched.
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, ModelShape] = {
    # GLM-5.1 (GlmMoeDsaForCausalLM) -- the exemplar, exact from /work/model/config.json
    "glm-5.1": ModelShape(
        name="zai-org/GLM-5.1", hidden_size=6144, num_layers=78, num_attention_heads=64,
        vocab_size=154880, intermediate_size=12288, gated_mlp=True,
        is_moe=True, n_routed_experts=256, n_experts_per_tok=8, moe_intermediate_size=2048,
        n_shared_experts=1, first_k_dense_replace=3,
        is_mla=True, kv_lora_rank=512, q_lora_rank=2048,
        qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256,
        # GLM-DSA lightning indexer: q/k proj over IDX_HEADS=32 x IDX_DIM=128
        extra_attn_params_per_layer=2 * 6144 * (32 * 128),
    ),
}


def shape_for_model(name: str) -> ModelShape | None:
    """Best-effort registry lookup by served model name (case-insensitive
    substring). Returns None when the family is not registered (renderer then
    degrades to the DCGM-proxy placement, clearly labeled)."""
    low = (name or "").lower()
    for key, shape in _REGISTRY.items():
        if key in low:
            return shape
    return None


def shape_to_dict(shape: ModelShape) -> dict[str, Any]:
    return asdict(shape)


def shape_from_dict(d: dict[str, Any]) -> ModelShape:
    fields = {k: v for k, v in d.items() if k in ModelShape.__dataclass_fields__}
    return ModelShape(**fields)
