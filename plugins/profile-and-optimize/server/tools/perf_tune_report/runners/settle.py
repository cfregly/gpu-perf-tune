"""Settle discipline for the aa backend: shape-matched prewarm + burn-in.

Productizes the measured fix from a 2026-06-11 GB300 settle audit: the first trials after a fresh deploy run 6-37% low because
tiny-prompt warmup never exercises the workload's input shapes, and prewarm
alone is insufficient (shape-prewarmed burn-ins still ran 2-16% low). The
discipline is therefore BOTH:

1. **Shape-matched prewarm** -- one completion per workload shape at the
   shape's input/output dims, so long-context shape capture and allocator
   growth happen before any timed request.
2. **Burn-in pass** -- one run-and-discard execution of the first
   concurrency point before the recorded sweep.
3. **Inter-point settle** -- a pause between recorded points so each starts
   from a quiesced engine (queues drained, KV ~0).

All best-effort and opt-in (``cell.aa.prewarm_shapes`` / ``cell.aa.burn_in``
/ ``cell.aa.settle_s``); failures never fail the bench cell.
"""

from __future__ import annotations

import subprocess

from tools.perf_tune_report.runners.aa_workload import AA_SHAPES


def build_prewarm_command(endpoint_url: str, model: str, shapes: list[str]) -> list[str]:
    """In-pod python one-liner sending one shape-matched completion per shape.

    Wrapped in ``kubectl exec`` by the caller (the same wrapper as the bench
    command). The prompt is sized at roughly the shape's input tokens (one
    word ~= one token for the repeated filler) and the completion is forced
    to the shape's output budget via ``ignore_eos``.
    """
    base = endpoint_url.rstrip("/")
    dims = []
    for name in shapes:
        if name not in AA_SHAPES:
            raise ValueError(f"unknown prewarm shape {name!r}; expected one of {sorted(AA_SHAPES)}")
        s = AA_SHAPES[name]
        dims.append((s.input_tokens, s.output_tokens))
    code = (
        "import urllib.request, json\n"
        f"dims = {dims!r}\n"
        "for pt, ot in [(50, 256)] + dims:\n"
        "    p = 'warmup token ' * pt\n"
        f"    body = {{'model': {model!r}, 'prompt': p, 'max_tokens': ot, 'temperature': 0, 'ignore_eos': True}}\n"
        f"    r = urllib.request.Request('{base}/v1/completions', data=json.dumps(body).encode(), headers={{'Content-Type': 'application/json'}})\n"
        "    urllib.request.urlopen(r, timeout=600).read()\n"
        "print('SHAPE_PREWARM_OK')"
    )
    return ["python", "-c", code]


def prewarm(
    endpoint_url: str,
    model: str,
    shapes: list[str],
    *,
    kube_wrap,
    subprocess_runner=subprocess.run,
) -> bool:
    """Run the shape-matched prewarm via the bench pod. Best-effort:
    returns False on failure instead of raising."""
    cmd = kube_wrap(build_prewarm_command(endpoint_url, model, shapes))
    try:
        proc = subprocess_runner(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return False
    return proc.returncode == 0 and "SHAPE_PREWARM_OK" in (proc.stdout or "")
