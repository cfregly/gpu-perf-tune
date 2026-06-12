"""Unit tests for the loader auto-select resolver (the single source of truth for loader choice)."""
from loader_advisor import mtp_from_serve_args, render_md, resolve


def test_glm_tput_nomtp_clean_win_runai():
    # non-MTP throughput tier with a runai image + S3 -> RunAI streamer (clean win)
    r = resolve(mtp=False, hf_egress_ok=True, image_has_runai=True, s3_available=True)
    assert r.recommended == "runai"
    assert r.tier == "clean-win"
    assert r.needs_mtp_patch is False
    assert r.fragment_key == "baseline-runai"


def test_glm_lat_mtp_hf_prefers_hfpull():
    # MTP latency tier with HF egress -> hf-pull (MTP-native, avoids the runai double-stream)
    r = resolve(mtp=True, hf_egress_ok=True, image_has_runai=True, s3_available=True)
    assert r.recommended == "hf-pull"
    assert r.tier == "clean-win"
    assert r.fragment_key == "baseline"


def test_glm_lat_mtp_no_hf_runai_with_patch_tradeoff():
    # MTP tier, no HF egress -> runai + the MTP drafter patch (tradeoff: ~16min double-stream)
    r = resolve(mtp=True, hf_egress_ok=False, image_has_runai=True, s3_available=True)
    assert r.recommended == "runai"
    assert r.needs_mtp_patch is True
    assert r.tier == "tradeoff"
    assert r.fragment_key == "baseline-runai"


def test_nomtp_no_runai_image_falls_back_to_hfpull():
    r = resolve(mtp=False, hf_egress_ok=True, image_has_runai=False, s3_available=True)
    assert r.recommended == "hf-pull"
    assert r.tier == "ok"


def test_nomtp_no_hf_no_runai_s3_only_returns_none():
    # s3fs FUSE fallback retired 2026-06-09: S3-only + no runai image + no HF egress -> none
    r = resolve(mtp=False, hf_egress_ok=False, image_has_runai=False, s3_available=True)
    assert r.recommended == "none"
    assert r.fragment_key == ""
    assert any("No viable loader" in x for x in r.reasons)


def test_mtp_no_hf_no_runai_s3_only_returns_none():
    # MTP variant: s3fs fallback retired -> none (provide a runai image for the S3 path)
    r = resolve(mtp=True, hf_egress_ok=False, image_has_runai=False, s3_available=True)
    assert r.recommended == "none"
    assert r.fragment_key == ""


def test_none_viable_when_no_hf_and_no_s3():
    r = resolve(mtp=False, hf_egress_ok=False, image_has_runai=True, s3_available=False)
    assert r.recommended == "none"
    assert r.fragment_key == ""
    assert any("No viable loader" in x for x in r.reasons)


def test_mtp_autodetect_from_serve_args():
    assert mtp_from_serve_args('--speculative-config \'{"method":"mtp","num_speculative_tokens":2}\'') is True
    assert mtp_from_serve_args("--max-num-seqs 384 --max-num-batched-tokens 12288") is False


def test_render_md_has_recommendation_and_gates():
    r = resolve(mtp=False, hf_egress_ok=True, image_has_runai=True, s3_available=True)
    md = render_md(r, mtp=False)
    assert "RECOMMENDED LOADER: `runai`" in md
    assert "[pass] **runai_image**" in md
    assert "## Rationale" in md
