"""Safety constants for the LLM-assisted MLPerf tuner.

Per the Reviewability Overhaul plan the patterns that block dangerous
LLM-proposed shell snippets live in their own small module so reviewers
can audit them without paging through the 3,000-line tuner CLI.

The tuner ([`ai_tuning.py`](ai_tuning.py)) re-exports the same names for
stable public API compatibility; new code should import from this module
directly. Per [`../../CLAUDE.md`](../../CLAUDE.md) "AI-Assisted Tuning
Safety", any change to ``FORBIDDEN_PATCH_PATTERNS`` requires a
corresponding test that proves the pattern still rejects the canonical
malicious snippet.
"""

from __future__ import annotations

#: Patterns that any LLM-proposed config / template patch must NOT match.
#: Format: ``(error_code, regex)``. The regex flavor is Python's stdlib
#: ``re`` syntax. Patterns are evaluated in order; the tuner aborts on
#: the first match. New patterns should target a *behavioral* class
#: (mutates Slurm state, mutates Kubernetes state, deletes data,
#: restarts a service, etc.), not a specific binary name.
#:
#: When adding a new pattern:
#:
#: 1. Add a row here with a ``snake_case`` error code unique within the
#:    table.
#: 2. Add a unit test that proves the pattern fires on the canonical
#:    malicious snippet AND that a benign snippet does not false-fire.
#: 3. Update the relevant section of ``docs/ai-assisted-tuning.md``.
FORBIDDEN_PATCH_PATTERNS: tuple[tuple[str, str], ...] = (
    ("submit_slurm_job", r"(^|[;&|]\s*)\s*sbatch\b"),
    ("cancel_slurm_job", r"\bscancel\b"),
    (
        "mutate_slurm_state",
        r"\bscontrol\s+(update|requeue|hold|release|reboot|down|resume)\b",
    ),
    (
        "mutate_kubernetes",
        r"\bkubectl\s+(delete|drain|cordon|uncordon|scale|patch|apply|replace|exec)\b",
    ),
    ("destructive_remove", r"\brm\s+-[^\n]*r[^\n]*f|\brm\s+-[^\n]*f[^\n]*r"),
    ("restart_service", r"\bsystemctl\s+(restart|stop|disable)\b"),
    ("power_action", r"\b(reboot|shutdown|poweroff)\b"),
)


#: Allowed lifecycle states for a tuner experiment row in
#: ``experiments/ledger.jsonl``. The set is intentionally small; new
#: states require a same-diff update to every consumer that reads
#: ``status`` (campaign DAG, dashboard, ledger summarizer).
EXPERIMENT_STATUSES: frozenset[str] = frozenset(
    {
        "planned",
        "staged",
        "submitted",
        "running",
        "succeeded",
        "failed",
        "blocked",
        "cancelled",
    }
)


#: Schema version for ledger / proposal / template-patch / experiment
#: ledger documents. Bump only on **strict-superset** changes; every
#: consumer must accept the previous version in the same diff that bumps
#: the constant. Per CLAUDE.md "Vendor With Intent (DRY / YAGNI)".
REPORT_SCHEMA_VERSION: int = 1
PROPOSAL_SCHEMA_VERSION: int = 1
TEMPLATE_PATCH_SCHEMA_VERSION: int = 1
EXPERIMENT_LEDGER_SCHEMA_VERSION: int = 1


__all__ = (
    "EXPERIMENT_LEDGER_SCHEMA_VERSION",
    "EXPERIMENT_STATUSES",
    "FORBIDDEN_PATCH_PATTERNS",
    "PROPOSAL_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "TEMPLATE_PATCH_SCHEMA_VERSION",
)
