"""experiment_provenance_v1: structured source-code attribution for experiments.

This is the machine-readable layer that links an experiment (its immutable
``run_id``) to the ACTUAL source code under test -- the vLLM commit/branch, the
container image, and the delivery method (stock image vs ConfigMap overlay vs
``patchedVllm`` subPath vs an ``infr`` patch) -- plus human-queryable identity
(title / hypothesis / tags / status) and the verdict.

The provenance block is a fenced ````provenance```` YAML region embedded in a
bundle's ``SOURCE.md`` (one file, no drift, already the universal artifact the
coverage audit + the lake key off). ``SOURCE.md`` keeps its human prose; this
block is the source of truth a tool can parse.

Canonical shape (``experiment_provenance_v1``)::

    ```provenance
    schema: experiment_provenance_v1
    identity:
      run_id: glm51-nvfp4-vs-fp8-ab-20260604T103817Z   # == bundle dir, immutable
      id_slug: glm51-nvfp4-vs-fp8-ab-20260604t103817z
      title: "NVFP4 vs FP8 decode/throughput A/B on GB300"
      hypothesis: "NVFP4 weights beat block-FP8 on decode TPOT at parity accuracy"
      family: quant-format-ab
      tags: [glm51, nvfp4, fp8, gb300, quant]
      status: verified          # active|verified|refuted|incomplete|superseded
      supersedes: ""
      superseded_by: ""
    source:                     # one entry per code-under-test (vLLM + the harness)
      - repo: example/vllm
        branch: feature/nvfp4-kv
        commit: b5743e12e...           # full or short SHA
        dirty: false
        delivery: overlay              # image|overlay|patchedVllm|infr-patch
        overlay_mode: pythonpath-sitecustomize  # subpath|patchset-initcontainer|pythonpath-sitecustomize (when delivery=overlay)
        image: registry.example.com/infr/vllm:v2.12.3
        image_digest: sha256:d486637f9f5bb12c016a033770f57af99428493f6d92575043b3af22940a5844  # immutable content pin (the tag is mutable)
        image_pip_version: 0.21.1.dev0+gad7125a43
        overlay_configmap: glm51-nvfp4kv-gb300-overlay
        patch_files: [vllm/v1/attention/...]
      - repo: example/perf-tune-glm51
        commit: eafb4b4...
    lineage:
      cluster_objects: [glm51-nvfp4kv-gb300]
      perf_lake_campaign: campaign=glm51-nvfp4-vs-fp8-ab-20260604T103817Z
      s3_paths: ["s3://perf-lake/perflake/perf-report/atlas_v1/dt=.../campaign=.../part-0.parquet"]
    verdict:
      tier: verdict             # draft|verdict
      claim: "ship NVFP4 on GB300"
      baseline: "block-FP8, same vLLM + FlashInfer-TRTLLM"
      metric: tpot_median_ms
    ```

The flat ``vllm_*`` / ``code_*`` / ``title`` / ``experiment_status`` keys this
module derives (``flatten_for_lake``) become first-class ``campaign_v1`` columns
so the whole chain ``run_id -> source commit/branch -> verdict -> metrics`` is a
single query. Added for the durable-lineage workstream.
"""

from __future__ import annotations

import re
from typing import Any
from pathlib import Path

SCHEMA = "experiment_provenance_v1"

#: delivery methods = how the source code actually reached the cluster.
DELIVERY_KINDS = ("image", "overlay", "patchedVllm", "infr-patch")
#: overlay sub-mode (only when delivery == "overlay") = HOW the runtime overlay is
#: applied. The delivery ladder (AGENTS.md "Experiment delivery ladder"):
#:   subpath                  = ConfigMap files mounted over dist-packages via subPath
#:   patchset-initcontainer   = initContainer applies a .patch set into a shared
#:                              emptyDir, main subPath-remounts (overlay-patchset.sh)
#:   pythonpath-sitecustomize = PYTHONPATH=/overlay + sitecustomize monkeypatch (custom-ops only)
OVERLAY_MODES = ("subpath", "patchset-initcontainer", "pythonpath-sitecustomize")
#: experiment lifecycle status (queryable; first-class for negatives + partials).
STATUS_VALUES = ("active", "verified", "refuted", "incomplete", "superseded")
#: verdict rigor tier (mirrors lake_writer.VerdictSummary.tier).
VERDICT_TIERS = ("draft", "verdict")

#: The fenced ```provenance ...``` region inside a SOURCE.md. The closing fence
#: must start a line; DOTALL lets the body span lines.
_BLOCK_RE = re.compile(r"^```provenance[ \t]*\n(.*?)\n```[ \t]*$", re.DOTALL | re.MULTILINE)

#: Flat lake-column keys this module emits (also the campaign SOURCE.md bullet
#: keys parse_source_md reads). Kept here so lake_writer + the audit agree.
LAKE_KEYS = (
    "title",
    "experiment_status",
    "tags",
    "hypothesis",
    "vllm_repo",
    "vllm_branch",
    "vllm_commit",
    "vllm_image",
    "vllm_image_digest",
    "vllm_pip_version",
    "delivery",
    "overlay_mode",
    "code_repo",
    "code_sha",
)


def extract_block(text: str) -> str | None:
    """Return the raw YAML body of the ```provenance``` fenced block, or None."""
    m = _BLOCK_RE.search(text or "")
    return m.group(1) if m else None


def parse_text(text: str) -> dict[str, Any] | None:
    """Parse the provenance block out of SOURCE.md text. None if no block.

    Uses PyYAML (a perf_tune_report core dep). A malformed block raises so a typo is
    loud rather than silently dropping provenance.
    """
    body = extract_block(text)
    if body is None:
        return None
    import yaml  # core dep (helpers.load_yaml relies on it too)

    data = yaml.safe_load(body)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("provenance block must be a YAML mapping")
    return data


def parse_file(source_md: Path) -> dict[str, Any] | None:
    """Parse the provenance block from a SOURCE.md file. None if file/block absent."""
    p = Path(source_md)
    if not p.is_file():
        return None
    return parse_text(p.read_text(encoding="utf-8", errors="ignore"))


def _sources(prov: dict[str, Any]) -> list[dict[str, Any]]:
    src = prov.get("source") or []
    return [s for s in src if isinstance(s, dict)]


def vllm_source(prov: dict[str, Any]) -> dict[str, Any]:
    """The source entry that is vLLM (repo endswith /vllm), else the first entry."""
    srcs = _sources(prov)
    for s in srcs:
        if str(s.get("repo", "")).rstrip("/").endswith("/vllm") or s.get("delivery"):
            return s
    return srcs[0] if srcs else {}


def harness_source(prov: dict[str, Any]) -> dict[str, Any]:
    """The deploy-harness source entry (repo contains 'deploy' or is a perf-tune-* bundle), else {}."""
    for s in _sources(prov):
        repo = str(s.get("repo", "")).lower()
        if "deploy" in repo or "/perf-tune-" in repo:
            return s
    return {}


def flatten_for_lake(prov: dict[str, Any]) -> dict[str, str]:
    """Project the nested block to the flat string keys the lake stores.

    Every value is a string (campaign SOURCE.md bullets + Parquet columns are
    strings); empty string when absent so the schema is always populated.
    """
    ident = prov.get("identity") or {}
    verd = prov.get("verdict") or {}
    v = vllm_source(prov)
    h = harness_source(prov)
    tags = ident.get("tags") or []
    if isinstance(tags, (list, tuple)):
        tags_s = ",".join(str(t) for t in tags)
    else:
        tags_s = str(tags)
    out = {
        "title": str(ident.get("title", "") or ""),
        "experiment_status": str(ident.get("status", "") or ""),
        "tags": tags_s,
        "hypothesis": str(ident.get("hypothesis", "") or ""),
        "vllm_repo": str(v.get("repo", "") or ""),
        "vllm_branch": str(v.get("branch", "") or ""),
        "vllm_commit": str(v.get("commit", "") or ""),
        "vllm_image": str(v.get("image", "") or ""),
        "vllm_image_digest": str(v.get("image_digest", "") or ""),
        "vllm_pip_version": str(v.get("image_pip_version", "") or ""),
        "delivery": str(v.get("delivery", "") or ""),
        "overlay_mode": str(v.get("overlay_mode", "") or ""),
        "code_repo": str(h.get("repo", "") or ""),
        "code_sha": str(h.get("commit", "") or ""),
    }
    # Verdict tier is recorded separately (verdict.json -> verdict_tier); keep
    # the claim/baseline reachable via the block, not duplicated as a column.
    _ = verd
    return out


def validate(prov: dict[str, Any]) -> list[str]:
    """Structural problems with a provenance block (empty list = OK).

    Schema-shape only (no git / no cluster). The git "commit exists + is on a
    pushed branch" check lives in the inference-side provenance audit, which runs
    where git is available.
    """
    problems: list[str] = []
    if not isinstance(prov, dict):
        return ["provenance is not a mapping"]
    schema = prov.get("schema")
    # Absent schema defaults to the current (single) version: a block with no
    # ``schema:`` line is unambiguously ``experiment_provenance_v1``, so its
    # absence is NOT a problem. Only a PRESENT-but-wrong value is a genuine
    # version mismatch worth flagging. (Removes the systemic false-positive on
    # hand-authored roofline/ncu fleet bundles that omit the version marker.)
    if schema is not None and schema != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r} (got {schema!r})")
    ident = prov.get("identity") or {}
    if not ident.get("run_id"):
        problems.append("identity.run_id is required (the immutable join key)")
    status = ident.get("status")
    if status and status not in STATUS_VALUES:
        problems.append(
            f"identity.status {status!r} not in {STATUS_VALUES}"
        )
    srcs = _sources(prov)
    if not srcs:
        problems.append("source: must list >=1 code-under-test entry (repo + commit)")
    for i, s in enumerate(srcs):
        if not s.get("repo"):
            problems.append(f"source[{i}].repo is required")
        if not s.get("commit"):
            problems.append(f"source[{i}].commit is required (the actual source SHA)")
        deliv = s.get("delivery")
        if deliv and deliv not in DELIVERY_KINDS:
            problems.append(
                f"source[{i}].delivery {deliv!r} not in {DELIVERY_KINDS}"
            )
        omode = s.get("overlay_mode")
        if omode and omode not in OVERLAY_MODES:
            problems.append(
                f"source[{i}].overlay_mode {omode!r} not in {OVERLAY_MODES}"
            )
    verd = prov.get("verdict") or {}
    tier = verd.get("tier")
    if tier and tier not in VERDICT_TIERS:
        problems.append(f"verdict.tier {tier!r} not in {VERDICT_TIERS}")
    return problems


def source_provenance_problems(prov: dict[str, Any] | None, verdict_tier: str) -> list[str]:
    """VERDICT-tier source-pinning gate (empty list when draft / OK).

    A ship/no-ship VERDICT must be attributable to a clean, pinned source
    commit. Drafts publish freely (return []); this is the source-code analog of
    ``verdict_problems`` (same-node / >=3 trials / named baseline).
    """
    if verdict_tier != "verdict":
        return []
    if not prov:
        return [
            "verdict_tier=verdict needs a provenance block in SOURCE.md "
            "(```provenance ... ```) pinning the source commit -- run "
            "capture-provenance.sh on the bundle"
        ]
    problems: list[str] = []
    v = vllm_source(prov)
    if not v.get("commit"):
        problems.append(
            "verdict_tier=verdict needs source[*].commit pinned (the exact vLLM "
            "SHA under test) -- a verdict with no source commit is not attributable"
        )
    if v.get("dirty") is True:
        problems.append(
            "verdict_tier=verdict needs a clean source tree (source[*].dirty=true) "
            "-- commit + push the patch to example/vllm before promoting"
        )
    return problems


def _commit_prefix_match(a: str, b: str) -> bool:
    """SHA equality tolerant of short-vs-full (one a prefix of the other)."""
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return True
    return a.startswith(b) or b.startswith(a)


def provenance_match_problems(
    expected_refs: list[dict[str, Any]] | None,
    campaign_prov: dict[str, Any] | None,
    *,
    label: str = "",
) -> list[str]:
    """Flag CROSS-TIER citation: a finding/PR cites a campaign whose code-under-test
    provenance (delivery + commit) does NOT match the code the finding claims to promote.

    ``expected_refs`` -- the finding's declared source(s) = the code being promoted, e.g.
        value-findings.yaml ``source_refs: [{delivery: infr-patch, commit: ...}]`` (or a PR's
        stated delivery/commit).
    ``campaign_prov`` -- the CITED campaign's provenance, accepted as EITHER a nested
        ``experiment_provenance_v1`` block (has ``source``/``identity``) OR the flattened
        lake/``provenance.json`` form (has ``delivery``/``vllm_commit``).

    Mechanical backstop for AGENTS.md "code-under-test provenance match" (rigor principle ``p``):
    an ``overlay`` / offline-prepped campaign cited as the benefit of an ``infr-patch`` is a DRAFT
    defect even when the kernels match. Empty list = OK (matches, or not enough info to assert).
    Commit comparison is prefix-tolerant (short vs full SHA).
    """
    if not expected_refs or not campaign_prov:
        return []
    if "source" in campaign_prov or "identity" in campaign_prov:
        cv = vllm_source(campaign_prov)
        cam_delivery = str(cv.get("delivery", "") or "")
        cam_commit = str(cv.get("commit", "") or "")
    else:  # flattened lake / provenance.json form
        cam_delivery = str(campaign_prov.get("delivery", "") or "")
        cam_commit = str(
            campaign_prov.get("vllm_commit") or campaign_prov.get("code_sha") or ""
        )
    if not cam_delivery and not cam_commit:
        return []  # campaign provenance unknown -> nothing to assert (ungrounded check covers it)
    pfx = f"{label}: " if label else ""
    problems: list[str] = []
    for ref in expected_refs:
        if not isinstance(ref, dict):
            continue
        exp_delivery = str(ref.get("delivery", "") or "")
        exp_commit = str(ref.get("commit", "") or "")
        if exp_delivery and cam_delivery and exp_delivery != cam_delivery:
            problems.append(
                f"{pfx}code-under-test mismatch -- cited campaign delivery={cam_delivery!r} but the "
                f"finding declares delivery={exp_delivery!r}; a {cam_delivery} measurement is not "
                f"evidence for {exp_delivery} code (cite the patch's own run -- rigor principle p)"
            )
        if exp_commit and cam_commit and not _commit_prefix_match(exp_commit, cam_commit):
            problems.append(
                f"{pfx}code-under-test mismatch -- cited campaign commit {cam_commit[:12]} != the "
                f"finding's commit {exp_commit[:12]} (principle p)"
            )
    return problems


def render_block(prov: dict[str, Any]) -> str:
    """Serialize a provenance dict to a fenced ```provenance``` block string.

    Used by capture/backfill tooling on the profile_and_optimize side; the bash
    capture-provenance.sh builds the equivalent text without PyYAML.
    """
    import yaml

    body = yaml.safe_dump(prov, sort_keys=False, default_flow_style=False).rstrip("\n")
    return f"```provenance\n{body}\n```\n"


def flat_bullets(prov: dict[str, Any]) -> str:
    """The flat ``- key: value`` lines (lake keys) for a campaign SOURCE.md, so
    ``lake_writer.parse_source_md`` lifts them into ``campaign_v1`` columns."""
    flat = flatten_for_lake(prov)
    return "".join(f"- {k}: {v}\n" for k, v in flat.items() if v)


def github_url(repo: str, branch: str = "", commit: str = "") -> str:
    """Best-effort GitHub URL for a source entry. Prefers the exact commit
    (``/commit/<sha>``), else the branch (``/tree/<branch>``), else the repo."""
    repo = (repo or "").rstrip("/")
    if not repo:
        return ""
    base = f"https://github.com/{repo}"
    if commit:
        return f"{base}/commit/{commit}"
    if branch:
        return f"{base}/tree/{branch}"
    return base


def _registry_lookup(registry: dict[str, Any] | None, branch: str) -> dict[str, Any]:
    """Find a branch's entry in a parsed source-registry.yaml (the
    ``branches:[{branch, purpose, status, landed_as, ...}]`` shape)."""
    if not registry or not branch:
        return {}
    for b in registry.get("branches", []) or []:
        if isinstance(b, dict) and b.get("branch") == branch:
            return b
    return {}


def source_links(prov: dict[str, Any] | None, registry: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Render each code-under-test entry as a link descriptor for the PDF.

    One dict per ``source[]`` entry: ``{repo, branch, commit, delivery, image,
    url, purpose, patch}``. ``purpose`` comes from the source-registry (when the
    branch is registered); ``patch`` lists any ``patch_files`` (e.g. the
    ``infr/images/{vllm,sglang}/patches/*.patch`` for an ``infr-patch`` delivery).
    Empty list when there is no provenance block.
    """
    if not prov:
        return []
    out: list[dict[str, str]] = []
    for s in _sources(prov):
        repo = str(s.get("repo", "") or "")
        branch = str(s.get("branch", "") or "")
        commit = str(s.get("commit", "") or "")
        reg = _registry_lookup(registry, branch)
        patch_files = s.get("patch_files") or []
        if isinstance(patch_files, (list, tuple)):
            patch = ", ".join(str(p) for p in patch_files)
        else:
            patch = str(patch_files)
        out.append({
            "repo": repo,
            "branch": branch,
            "commit": commit,
            "delivery": str(s.get("delivery", "") or ""),
            "overlay_mode": str(s.get("overlay_mode", "") or ""),
            "image": str(s.get("image", "") or ""),
            "image_digest": str(s.get("image_digest", "") or ""),
            "url": github_url(repo, branch, commit),
            "purpose": str(reg.get("purpose", "") or ""),
            "patch": patch,
        })
    return out
