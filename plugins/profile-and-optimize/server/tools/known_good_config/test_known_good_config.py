"""Unit tests for known_good_config.

Pure-Python; no Slurm / MCP / network. Run under
``pytest -q tools/known_good_config/test_known_good_config.py`` from ``server/``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tools.known_good_config.known_good_config_cli import (
    CONTRACT,
    _find_model,
    _load_registry,
    _parse_required_flag,
    build_parser,
    main,
)

REGISTRY_HEADER = """\
# HAND-CURATED -- a comment that record() MUST preserve.
schema: known_good_config_v1

models:
  - model: nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4
    slug: qwen3-next-80b-thinking
    engine: vllm
    required_flags:
      - flag: '--additional-config gdn_prefill_backend=triton'
        match: 'gdn_prefill_backend["\\:= ]+triton'
        severity: boot-blocker
        why: FlashInfer GDN prefill wedges on the interleaved-GQA layout.
  - model: zai-org/GLM-5.1
    slug: glm51
    engine: vllm
    required_flags:
      - flag: 'VLLM_ATTENTION_BACKEND=FLASHMLA'
        match: 'FLASHMLA'
        severity: perf
        why: pin the attention backend.
"""


@pytest.fixture()
def registry(tmp_path: Path) -> Path:
    p = tmp_path / "known-good-configs.yaml"
    p.write_text(REGISTRY_HEADER)
    return p


# ---------------------------------------------------------------------------
# Contract + helpers
# ---------------------------------------------------------------------------


def test_contract_shape() -> None:
    assert set(CONTRACT) == {"record", "check"}
    assert CONTRACT["record"]["safety"] == "writes_artifacts"
    assert CONTRACT["check"]["safety"] == "read_only"
    assert all(CONTRACT[v]["json"] for v in CONTRACT)


def test_build_parser_accepts_json_for_both_verbs() -> None:
    parser = build_parser()
    for verb in ("record", "check"):
        ns = parser.parse_args([verb, "--model", "m", "--json"])
        assert ns.json is True


def test_parse_required_flag_defaults() -> None:
    rf = _parse_required_flag("--foo")
    assert rf["flag"] == "--foo"
    assert rf["severity"] == "boot-blocker"
    assert rf["match"]  # defaults to re.escape(flag)


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def test_check_pass_when_required_flag_present(registry: Path, capsys) -> None:
    rc = main([
        "check", "--registry", str(registry),
        "--model", "nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4",
        "--serve-args", "vllm serve --additional-config '{\"gdn_prefill_backend\":\"triton\"}'",
        "--json",
    ])
    assert rc == 0
    assert '"verdict": "pass"' in capsys.readouterr().out


def test_check_fail_when_boot_blocker_missing(registry: Path, capsys) -> None:
    rc = main([
        "check", "--registry", str(registry),
        "--model", "nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4",
        "--serve-args", "vllm serve --tensor-parallel-size 4",  # no gdn_prefill_backend
        "--json",
    ])
    assert rc == 1  # fail-closed
    out = capsys.readouterr().out
    assert '"verdict": "fail"' in out
    assert "gdn_prefill_backend" in out


def test_check_perf_severity_missing_is_warning_not_failure(registry: Path, capsys) -> None:
    rc = main([
        "check", "--registry", str(registry),
        "--model", "zai-org/GLM-5.1",
        "--serve-args", "vllm serve --tensor-parallel-size 8",  # no FLASHMLA (perf only)
        "--json",
    ])
    assert rc == 0  # perf-severity missing -> warn, not fail
    out = capsys.readouterr().out
    assert '"verdict": "pass"' in out
    assert "FLASHMLA" in out  # still reported in missing_required


def test_check_require_registered_fails_for_unknown_model(registry: Path, capsys) -> None:
    rc = main([
        "check", "--registry", str(registry),
        "--model", "made/up-model", "--require-registered", "--json",
    ])
    assert rc == 1
    assert "model_not_registered" in capsys.readouterr().out


def test_check_unknown_model_without_require_registered_passes(registry: Path) -> None:
    rc = main(["check", "--registry", str(registry), "--model", "made/up-model", "--json"])
    assert rc == 0


def test_check_registered_no_serveargs_passes(registry: Path, capsys) -> None:
    rc = main([
        "check", "--registry", str(registry),
        "--model", "zai-org/GLM-5.1", "--json",
    ])
    assert rc == 0
    assert '"checked_args": false' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------


def test_record_appends_new_model_and_preserves_comments(registry: Path) -> None:
    rc = main([
        "record", "--registry", str(registry),
        "--model", "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "--slug", "qwen3-235b-thinking", "--engine", "vllm",
        "--champion-verdict", "VERDICT vLLM 1.6x SGLang",
        "--json",
    ])
    assert rc == 0
    text = registry.read_text()
    assert "# HAND-CURATED -- a comment that record() MUST preserve." in text
    reg = _load_registry(registry)
    entry = _find_model(reg, "Qwen/Qwen3-235B-A22B-Thinking-2507")
    assert entry is not None
    assert entry["slug"] == "qwen3-235b-thinking"
    assert entry["champion"]["verdict"] == "VERDICT vLLM 1.6x SGLang"
    # The pre-existing models are still present.
    assert _find_model(reg, "zai-org/GLM-5.1") is not None
    assert len(reg["models"]) == 3


def test_record_rejects_existing_model(registry: Path) -> None:
    rc = main([
        "record", "--registry", str(registry),
        "--model", "zai-org/GLM-5.1", "--json",
    ])
    assert rc == 2  # must edit by hand


def test_record_parses_required_flag_tuple(registry: Path) -> None:
    rc = main([
        "record", "--registry", str(registry),
        "--model", "new/model",
        "--required-flag", "--moe-backend=cutlass|moe-backend[ =]+cutlass|crash-high-c|default MoE crashes c>=64|vllm 0.22|some/evidence",
        "--json",
    ])
    assert rc == 0
    reg = _load_registry(registry)
    entry = _find_model(reg, "new/model")
    assert entry["required_flags"][0]["severity"] == "crash-high-c"
    assert entry["required_flags"][0]["match"] == "moe-backend[ =]+cutlass"
