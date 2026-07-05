"""Per-number exact shape (no smoothing) in the TPM-by-hardware caption.

docs/METHODOLOGY.md "Per-number exact shape (no smoothing)" / CLAUDE.md principle j:
a per-hardware shape caption must NOT collapse heterogeneous-shape variant groups to one
ISL/OSL label (which would hide per-point variation). _shape_caption emits a single
ISL/OSL only when the groups share it; otherwise "ISL/OSL: per-row (varies)".
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.perf_tune_report.renderer.tpm_table import (
    _shape_caption,
    shape_label_problems,
)


def _g(isl, osl, cache="warm"):
    # _shape_caption only reads .mean_isl / .mean_osl / .cache_mode (duck-typed).
    return SimpleNamespace(mean_isl=isl, mean_osl=osl, cache_mode=cache)


def test_homogeneous_groups_get_one_shape_label():
    bits = _shape_caption([_g(1024, 512), _g(1024, 512)])
    assert "ISL~1024" in bits
    assert "OSL~512" in bits
    assert "cache: warm" in bits
    assert not any("varies" in b for b in bits)


def test_heterogeneous_groups_not_smoothed():
    # c=1 @ ISL1024/OSL256 + c=64 @ ISL4096/OSL512 -- the PR #742 case.
    bits = _shape_caption([_g(1024, 256), _g(4096, 512)])
    assert not any(b.startswith("ISL~") for b in bits), bits
    assert not any(b.startswith("OSL~") for b in bits), bits
    assert "ISL/OSL: per-row (varies)" in bits
    assert "cache: warm" in bits


def test_partial_shape_does_not_force_varies():
    # One group has no shape (None,None) -- it must not create a phantom second shape.
    bits = _shape_caption([_g(1024, 512), _g(None, None)])
    assert "ISL~1024" in bits
    assert "OSL~512" in bits
    assert not any("varies" in b for b in bits)


def test_no_shape_groups_only_cache():
    assert _shape_caption([_g(None, None)]) == ["cache: warm"]


def test_empty_groups():
    assert _shape_caption([]) == ["cache: unknown"]


# --- shape_label_problems() (the render-layer detector _shape_caption consults) ---


def test_shape_label_problems_empty_when_one_shape():
    assert shape_label_problems([_g(1024, 512), _g(1024, 512)]) == []


def test_shape_label_problems_empty_when_all_none():
    assert shape_label_problems([_g(None, None), _g(None, None)]) == []
    assert shape_label_problems([]) == []


def test_shape_label_problems_flags_heterogeneous_shapes():
    problems = shape_label_problems([_g(1024, 256), _g(4096, 512)])
    assert len(problems) == 1
    msg = problems[0]
    assert "2 distinct ISL/OSL shapes" in msg
    assert "1024" in msg and "256" in msg
    assert "4096" in msg and "512" in msg
    assert "label per-cell" in msg


def test_shape_label_problems_partial_shape_is_one_shape():
    # one group has no shape -- must not count as a second distinct shape.
    assert shape_label_problems([_g(1024, 512), _g(None, None)]) == []
