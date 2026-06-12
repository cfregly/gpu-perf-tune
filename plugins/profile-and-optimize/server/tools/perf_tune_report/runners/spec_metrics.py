"""Spec-decode acceptance-length capture from vLLM ``/metrics`` counter windows.

Formalizes the ad-hoc shell scrape that motivated first-class support: bracket
each bench window with a pre/post scrape of the endpoint's
``vllm:spec_decode_*`` counters, then compute over the window delta

    AL          = 1 + accepted_tokens / drafts
    accept_rate =     accepted_tokens / draft_tokens

Counter-based on purpose: client-side AL tools that drive their own chat
requests and read ``choices[0].message.content`` break on reasoning-parser
deploys (e.g. ``kimi_k2``, where output lands in ``reasoning_content``).
Server-side counters never touch response bodies, so this path is immune to
that class of breakage and composes with any load generator (here: AIPerf).

Scrapes are best-effort and NEVER fail the bench cell; every attempt is logged
to ``spec_metrics/scrape.log`` so a lost scrape is visible (the ad-hoc driver
this replaces once silently lost its scrapes to a missing parent dir).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from tools.perf_tune_report.schema import AtlasCell

# Prefixes persisted raw (superset of what AL needs, so generation/iteration-
# token cross-checks stay possible).
SCRAPE_PREFIXES = (
    "vllm:spec_decode",
    "vllm:generation_tokens",
    "vllm:iteration_tokens",
)

_METRIC_RE = re.compile(
    r"^(?P<name>vllm:spec_decode_[^\s{]+)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')

SPEC_METRICS_DIRNAME = "spec_metrics"


def build_scrape_command(endpoint_url: str) -> list[str]:
    """The in-pod python one-liner that prints the spec-relevant /metrics lines.

    Wrapped in ``kubectl exec`` by the caller (same wrapper as the bench
    command) so the scrape runs from the bench pod.
    """
    base = endpoint_url.rstrip("/")
    prefixes = ", ".join(repr(p) for p in SCRAPE_PREFIXES)
    code = (
        "import urllib.request\n"
        f"m = urllib.request.urlopen('{base}/metrics', timeout=30).read().decode()\n"
        "print('\\n'.join(l for l in m.splitlines() "
        f"if l.startswith(({prefixes}))))"
    )
    return ["python", "-c", code]


def parse_spec_totals(text: str) -> dict:
    """Parse ``vllm:spec_decode_*_total`` counters out of a scrape.

    Sums across label sets (multi-engine deploys export one series per
    ``engine`` label). ``*_created`` timestamp series are skipped. Returns
    drafts / draft_tokens / accepted_tokens plus the per-position accepted
    counts (position -> count).
    """
    totals = {"drafts": 0.0, "draft_tokens": 0.0, "accepted_tokens": 0.0}
    accepted_per_pos: dict[int, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        if name.endswith("_created"):
            continue
        value = float(match.group("value"))
        if name == "vllm:spec_decode_num_drafts_total":
            totals["drafts"] += value
        elif name == "vllm:spec_decode_num_draft_tokens_total":
            totals["draft_tokens"] += value
        elif name == "vllm:spec_decode_num_accepted_tokens_total":
            totals["accepted_tokens"] += value
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            labels = dict(_LABEL_RE.findall(match.group("labels") or ""))
            pos = labels.get("position")
            if pos is not None:
                accepted_per_pos[int(pos)] = (
                    accepted_per_pos.get(int(pos), 0.0) + value
                )
    return {**totals, "accepted_per_pos": accepted_per_pos}


def compute_spec_window(pre_text: str, post_text: str) -> dict | None:
    """Window delta -> AL + accept rate. None when the window saw no drafts
    (spec decode off, or the counters are absent from the scrape)."""
    pre = parse_spec_totals(pre_text)
    post = parse_spec_totals(post_text)
    drafts = post["drafts"] - pre["drafts"]
    draft_tokens = post["draft_tokens"] - pre["draft_tokens"]
    accepted = post["accepted_tokens"] - pre["accepted_tokens"]
    if drafts <= 0:
        return None
    positions = sorted(set(pre["accepted_per_pos"]) | set(post["accepted_per_pos"]))
    per_pos_accept_rate = {
        str(pos): (
            post["accepted_per_pos"].get(pos, 0.0)
            - pre["accepted_per_pos"].get(pos, 0.0)
        )
        / drafts
        for pos in positions
    }
    return {
        "num_drafts": drafts,
        "num_draft_tokens": draft_tokens,
        "num_accepted_tokens": accepted,
        "al": 1.0 + accepted / drafts,
        "accept_rate": (accepted / draft_tokens) if draft_tokens > 0 else None,
        "per_pos_accept_rate": per_pos_accept_rate,
    }


def attach_windows_to_rows(
    rows: list[AtlasCell], windows: dict[int, dict]
) -> list[AtlasCell]:
    """Set ``acceptance_length`` / ``spec_accept_rate`` on each row from its
    concurrency's window; raw counter deltas go to ``extra`` for provenance.
    Rows whose concurrency has no window pass through unchanged."""
    out: list[AtlasCell] = []
    for row in rows:
        win = windows.get(row.concurrency)
        if win is None:
            out.append(row)
            continue
        d = row.to_dict()
        d["acceptance_length"] = win["al"]
        d["spec_accept_rate"] = win["accept_rate"]
        d["extra"] = {
            **d.get("extra", {}),
            "spec_num_drafts": win["num_drafts"],
            "spec_num_draft_tokens": win["num_draft_tokens"],
            "spec_num_accepted_tokens": win["num_accepted_tokens"],
        }
        out.append(AtlasCell(**d))
    return out


class SpecMetricsCapture:
    """Per-cell scrape lifecycle: pre/post .prom persistence + window compute.

    One instance per cell_run; ``scrape(tag)`` is called pre/post each
    per-concurrency bench run, ``window(c)`` computes that concurrency's AL
    from the bracketing pair, ``finalize()`` writes ``spec_window.json``.
    """

    def __init__(
        self,
        cell_dir: Path,
        endpoint_url: str,
        *,
        kube_wrap,
        subprocess_runner=subprocess.run,
    ) -> None:
        self.dir = cell_dir / SPEC_METRICS_DIRNAME
        self._command = kube_wrap(build_scrape_command(endpoint_url))
        self._runner = subprocess_runner
        self._texts: dict[str, str] = {}
        self._log: list[str] = []
        self.windows: dict[int, dict] = {}

    def scrape(self, tag: str) -> bool:
        """Scrape /metrics into ``spec_metrics/metrics-<tag>.prom``. Best-effort:
        returns False (and logs) on failure instead of raising."""
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            proc = self._runner(
                self._command, capture_output=True, text=True, check=False
            )
        except OSError as exc:
            self._log.append(f"{tag} FAILED exec: {exc}")
            return False
        if proc.returncode != 0 or not (proc.stdout or "").strip():
            err = (proc.stderr or "").strip().splitlines()
            self._log.append(
                f"{tag} FAILED exit={proc.returncode}"
                + (f" stderr={err[-1]}" if err else "")
            )
            return False
        (self.dir / f"metrics-{tag}.prom").write_text(proc.stdout)
        self._texts[tag] = proc.stdout
        self._log.append(f"{tag} ok ({len(proc.stdout.splitlines())} lines)")
        return True

    def window(self, concurrency: int) -> dict | None:
        """Compute the AL window for one concurrency from its pre/post pair."""
        pre = self._texts.get(f"pre-c{concurrency}")
        post = self._texts.get(f"post-c{concurrency}")
        if pre is None or post is None:
            return None
        win = compute_spec_window(pre, post)
        if win is not None:
            self.windows[concurrency] = win
        return win

    def finalize(self) -> None:
        """Persist the computed windows + the scrape log (even when empty, so
        a fully-lost capture is an explicit artifact, not an absent dir)."""
        if not self._log:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "scrape.log").write_text("\n".join(self._log) + "\n")
        (self.dir / "spec_window.json").write_text(
            json.dumps(
                {str(c): w for c, w in sorted(self.windows.items())},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
