"""Tests for the experiment_provenance_v1 parser/validator/flattener."""

from __future__ import annotations

import pytest

from tools.perf_tune_report import provenance as p

_BLOCK = """# SOURCE -- human prose

Some narrative the operator wrote.

```provenance
schema: experiment_provenance_v1
identity:
  run_id: glm51-quant-ab-20260604T103817Z
  title: NVFP4 vs FP8 decode A/B
  hypothesis: NVFP4 beats FP8 on TPOT at parity accuracy
  status: verified
  tags: [glm51, nvfp4, fp8]
source:
  - repo: example/vllm
    branch: feature/nvfp4-kv
    commit: b5743e12e
    dirty: false
    delivery: overlay
    image: registry.example.com/infr/vllm:v2.12.3
    image_pip_version: 0.21.1.dev0+gad7125a43
  - repo: example/perf-tune-glm51
    commit: eafb4b4
verdict:
  tier: verdict
  baseline: block-FP8
```

trailing prose ignored.
"""


def test_parse_extracts_block():
    prov = p.parse_text(_BLOCK)
    assert prov is not None
    assert prov["schema"] == p.SCHEMA
    assert prov["identity"]["run_id"] == "glm51-quant-ab-20260604T103817Z"


def test_no_block_returns_none():
    assert p.parse_text("# SOURCE\n\njust prose, no fenced block\n") is None


def test_validate_clean_block_has_no_problems():
    assert p.validate(p.parse_text(_BLOCK)) == []


def test_validate_flags_missing_required_fields():
    prov = {
        "schema": "wrong",
        "identity": {"status": "bogus"},
        "source": [{"branch": "x"}],
        "verdict": {"tier": "maybe"},
    }
    probs = p.validate(prov)
    joined = " | ".join(probs)
    assert "schema must be" in joined
    assert "identity.run_id" in joined
    assert "source[0].repo" in joined
    assert "source[0].commit" in joined
    assert "identity.status" in joined
    assert "verdict.tier" in joined


def test_validate_absent_schema_defaults_to_current_version():
    # A provenance block with NO ``schema:`` line is unambiguously the single
    # current version (experiment_provenance_v1) -- validate() must NOT flag its
    # absence (only a present-but-wrong value flags). Guards the systemic
    # false-positive on hand-authored fleet bundles that omit the marker.
    prov = {
        "identity": {"run_id": "x-20260607T000000Z"},
        "source": [{"repo": "r", "commit": "c"}],
    }
    probs = p.validate(prov)
    assert not any("schema" in x for x in probs), probs


def test_validate_flags_present_but_wrong_schema():
    prov = {
        "schema": "experiment_provenance_v2",
        "identity": {"run_id": "x-20260607T000000Z"},
        "source": [{"repo": "r", "commit": "c"}],
    }
    assert any("schema must be" in x for x in p.validate(prov))


def test_flatten_picks_vllm_and_harness_entries():
    flat = p.flatten_for_lake(p.parse_text(_BLOCK))
    assert flat["vllm_repo"] == "example/vllm"
    assert flat["vllm_branch"] == "feature/nvfp4-kv"
    assert flat["vllm_commit"] == "b5743e12e"
    assert flat["delivery"] == "overlay"
    assert flat["tags"] == "glm51,nvfp4,fp8"
    assert flat["experiment_status"] == "verified"
    assert flat["code_repo"] == "example/perf-tune-glm51"
    assert flat["code_sha"] == "eafb4b4"


def test_flat_bullets_are_parseable_back():
    bullets = p.flat_bullets(p.parse_text(_BLOCK))
    assert "- vllm_commit: b5743e12e" in bullets
    assert "- experiment_status: verified" in bullets


def test_source_gate_passes_clean_verdict():
    assert p.source_provenance_problems(p.parse_text(_BLOCK), "verdict") == []


def test_source_gate_drafts_never_blocked():
    assert p.source_provenance_problems(None, "draft") == []
    assert p.source_provenance_problems({}, "draft") == []


def test_source_gate_blocks_verdict_without_block():
    probs = p.source_provenance_problems(None, "verdict")
    assert probs and "provenance block" in probs[0]


def test_source_gate_blocks_dirty_or_uncommitted_verdict():
    prov = p.parse_text(_BLOCK)
    prov["source"][0]["dirty"] = True
    prov["source"][0]["commit"] = ""
    probs = p.source_provenance_problems(prov, "verdict")
    joined = " | ".join(probs)
    assert "commit pinned" in joined
    assert "clean source tree" in joined


def test_render_block_roundtrips():
    prov = p.parse_text(_BLOCK)
    rendered = p.render_block(prov)
    assert rendered.startswith("```provenance\n")
    again = p.parse_text(rendered)
    assert again["identity"]["run_id"] == prov["identity"]["run_id"]


def test_malformed_block_raises():
    bad = "```provenance\n: : : not yaml : :\n- [unterminated\n```\n"
    with pytest.raises(Exception):
        p.parse_text(bad)


# --- provenance_match_problems (code-under-test provenance match, rigor principle p) ---

def test_match_ok_same_delivery_and_commit():
    refs = [{"delivery": "infr-patch", "commit": "abc123def456"}]
    camp = {"delivery": "infr-patch", "vllm_commit": "abc123def456789"}  # flattened, full SHA
    assert p.provenance_match_problems(refs, camp) == []


def test_match_flags_delivery_mismatch_overlay_vs_infr_patch():
    # The 2026-06-08 Gemma failure: an overlay campaign cited as the infr-patch benefit.
    refs = [{"delivery": "infr-patch", "commit": "abc123"}]
    camp = {"delivery": "overlay", "vllm_commit": "abc123"}
    probs = p.provenance_match_problems(refs, camp, label="gemma4-26b-a4b-gb300")
    assert len(probs) == 1
    assert "overlay" in probs[0] and "infr-patch" in probs[0]
    assert probs[0].startswith("gemma4-26b-a4b-gb300: ")


def test_match_flags_commit_mismatch():
    refs = [{"delivery": "infr-patch", "commit": "deadbeef0000"}]
    camp = {"delivery": "infr-patch", "vllm_commit": "feedface1111"}
    probs = p.provenance_match_problems(refs, camp)
    assert len(probs) == 1 and "commit" in probs[0]


def test_match_accepts_nested_campaign_block():
    refs = [{"delivery": "infr-patch", "commit": "b5743e12e"}]
    camp = p.parse_text(_BLOCK)  # nested block: delivery=overlay
    probs = p.provenance_match_problems(refs, camp)
    assert any("overlay" in s and "infr-patch" in s for s in probs)


def test_match_commit_prefix_is_tolerant():
    # short ref SHA vs full campaign SHA (one a prefix of the other) is a match.
    refs = [{"delivery": "overlay", "commit": "b5743e1"}]
    camp = p.parse_text(_BLOCK)  # delivery=overlay, commit=b5743e12e
    assert p.provenance_match_problems(refs, camp) == []


def test_match_no_false_positive_on_missing_info():
    assert p.provenance_match_problems(None, {"delivery": "overlay"}) == []
    assert p.provenance_match_problems([{"delivery": "infr-patch"}], None) == []
    assert p.provenance_match_problems([{"delivery": "infr-patch"}], {}) == []
    # campaign provenance carries neither delivery nor commit -> nothing to assert.
    assert p.provenance_match_problems(
        [{"delivery": "infr-patch", "commit": "x"}], {"foo": "bar"}
    ) == []
