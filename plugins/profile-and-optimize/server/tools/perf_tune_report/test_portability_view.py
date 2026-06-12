"""Unit tests for the perf_tune_report portability_view verb (lever x model matrix)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.perf_tune_report_cli import main
from tools.perf_tune_report.portability_view import (
    build_portability,
    cell_state,
    collect_models,
    render_markdown,
)


def _registry() -> dict:
    return {
        "baseline_context": "x",
        "findings": [
            {"id": "a", "title": "A", "lifecycle": "done", "applies_to": "all",
             "validated_on": ["GLM-5.1"], "candidate_on": ["Kimi-K2.6", "DSv3.2"],
             "refuted_on": []},
            {"id": "b", "title": "B", "lifecycle": "closed_negative", "applies_to": "moe",
             "validated_on": [], "candidate_on": [], "refuted_on": ["GLM-5.1"]},
            {"id": "c", "title": "C", "lifecycle": "not_done", "applies_to": "various",
             "validated_on": [], "candidate_on": [], "refuted_on": []},  # names no model
        ],
    }


def test_collect_models_is_sorted_union():
    assert collect_models(_registry()) == ["DSv3.2", "GLM-5.1", "Kimi-K2.6"]


def test_cell_state_precedence():
    fa = _registry()["findings"][0]
    assert cell_state(fa, "GLM-5.1") == "validated"
    assert cell_state(fa, "Kimi-K2.6") == "candidate"
    assert cell_state(fa, "V4-Flash") == "untested"
    fb = _registry()["findings"][1]
    assert cell_state(fb, "GLM-5.1") == "refuted"


def test_refuted_beats_candidate():
    """A model that is BOTH candidate and refuted reads as refuted (measured verdict wins)."""
    f = {"validated_on": [], "candidate_on": ["GLM-5.1"], "refuted_on": ["GLM-5.1"]}
    assert cell_state(f, "GLM-5.1") == "refuted"


def test_build_excludes_modelless_finding_and_lists_candidates():
    v = build_portability(_registry())
    assert [r["id"] for r in v["rows"]] == ["a", "b"]  # c excluded (names no model)
    assert v["candidates"]["Kimi-K2.6"] == ["a"]
    assert v["candidates"]["DSv3.2"] == ["a"]
    assert v["candidates"]["GLM-5.1"] == []  # validated/refuted, not a candidate


def test_render_markdown_matrix_and_trynext():
    md = render_markdown(build_portability(_registry()))
    assert "portability matrix" in md.lower()
    assert "| lever | applies_to |" in md
    assert "GLM-5.1" in md and "DSv3.2" in md
    assert "Try-next by model" in md
    assert "**Kimi-K2.6**: a" in md


def test_cli_portability_view(tmp_path, capsys):
    reg = tmp_path / "value-findings.yaml"
    reg.write_text(yaml.safe_dump(_registry()))
    campaigns = tmp_path / "campaigns"
    campaigns.mkdir()
    out = tmp_path / "PORTABILITY-MATRIX.md"
    rc = main(["portability_view", "--registry", str(reg),
               "--campaigns-dir", str(campaigns), "--out", str(out), "--json"])
    assert rc == 0
    env = json.loads(capsys.readouterr().out)
    assert env["lever_count"] == 2
    assert env["model_count"] == 3
    assert out.is_file() and "Try-next by model" in out.read_text()
