#!/usr/bin/env python3
"""MLPerf Training v6.0 rule constraint validator.

Loads ``tuning/mlperf_rules_v6_0.json`` and checks a candidate parameter
dict against benchmark-specific rules: max LR, max global batch, warmup
ratio, ruleset string, hyperparameter borrowing, expert parallel values,
LoRA rank, etc.

Per ``mlperf-6.0-training/CLAUDE.md`` "Fail Fast, No Silent Fallbacks", a
violation produces a structured rejection with a stable error code; never
a silent clamp to legal range.

Usage from Python::

    from mlperf_rules import load_rules, validate_candidate
    rules = load_rules(Path("tuning/mlperf_rules_v6_0.json"))
    violations = validate_candidate(
        rules,
        benchmark="llama31_405b",
        parameters={"learning_rate": "0.001", "global_batch_size": "9216"},
        nexp=5,
    )

Usage from CLI::

    python3 tools/ai_tuning/mlperf_rules.py validate \\
        --proposal experiments/proposals/foo.json \\
        --benchmark llama31_405b \\
        --rules tuning/mlperf_rules_v6_0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "tuning" / "mlperf_rules_v6_0.json"


@dataclass
class RuleViolation:
    code: str
    message: str
    severity: str = "blocker"
    parameter: str | None = None
    observed: Any = None
    legal: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "parameter": self.parameter,
            "observed": self.observed,
            "legal": self.legal,
        }


def load_rules(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_candidate(
    rules: dict[str, Any],
    *,
    benchmark: str,
    parameters: dict[str, Any],
    nexp: int = 1,
    use_synthetic_data: bool | None = None,
) -> list[RuleViolation]:
    """Return all violations the candidate triggers, in deterministic order."""
    out: list[RuleViolation] = []
    glob = rules.get("global", {}) or {}
    bench_rules = (rules.get("benchmarks") or {}).get(benchmark)
    if bench_rules is None:
        # Unknown benchmark -> blocker (no partial validation that pretends).
        out.append(
            RuleViolation(
                code="unknown_benchmark",
                message=f"benchmark {benchmark!r} not found in mlperf_rules_v6_0.json",
                parameter="benchmark",
                observed=benchmark,
            )
        )
        return out

    ruleset = parameters.get("MLPERF_RULESET")
    expected_ruleset = glob.get("ruleset_string")
    if expected_ruleset and ruleset and str(ruleset) != str(expected_ruleset):
        out.append(
            RuleViolation(
                code="ruleset_mismatch",
                message=f"MLPERF_RULESET={ruleset!r} not equal to {expected_ruleset!r}",
                parameter="MLPERF_RULESET",
                observed=ruleset,
                legal=expected_ruleset,
            )
        )

    if use_synthetic_data and nexp >= int(glob.get("submission_runs_required", 5)):
        if glob.get("synthetic_data_forbidden_in_submission"):
            out.append(
                RuleViolation(
                    code="synthetic_data_in_submission",
                    message="USE_SYNTHETIC_DATA=1 in submission run (NEXP>=5)",
                    parameter="USE_SYNTHETIC_DATA",
                    observed=True,
                    legal=False,
                )
            )

    lr = _coerce_float(parameters.get("learning_rate"))
    if lr is None:
        # Some tuning spaces only carry TRAINER_LR_SCALE; pair with rcp base
        # LR if available. We only flag explicit out-of-range learning_rate.
        pass
    else:
        max_lr = _coerce_float(bench_rules.get("max_lr"))
        min_lr = _coerce_float(bench_rules.get("min_lr"))
        if max_lr is not None and lr > max_lr:
            out.append(
                RuleViolation(
                    code="lr_above_max",
                    message=f"learning_rate {lr} exceeds max_lr {max_lr}",
                    parameter="learning_rate",
                    observed=lr,
                    legal={"max": max_lr},
                )
            )
        if min_lr is not None and lr < min_lr:
            out.append(
                RuleViolation(
                    code="lr_below_min",
                    message=f"learning_rate {lr} below min_lr {min_lr}",
                    parameter="learning_rate",
                    observed=lr,
                    legal={"min": min_lr},
                )
            )

    gbs = _coerce_int(parameters.get("global_batch_size"))
    if gbs is not None:
        gbs_rules = bench_rules.get("global_batch_size") or {}
        gbs_min = _coerce_int(gbs_rules.get("min"))
        gbs_max = _coerce_int(gbs_rules.get("max"))
        if (gbs_min is not None and gbs < gbs_min) or (
            gbs_max is not None and gbs > gbs_max
        ):
            out.append(
                RuleViolation(
                    code="global_batch_outside_legal",
                    message=(
                        f"global_batch_size {gbs} outside legal range "
                        f"[{gbs_min},{gbs_max}]"
                    ),
                    parameter="global_batch_size",
                    observed=gbs,
                    legal={"min": gbs_min, "max": gbs_max},
                )
            )

    warmup = _coerce_int(parameters.get("warmup_steps"))
    max_steps = _coerce_int(parameters.get("TRAINER_TRAIN_STEPS")) or _coerce_int(
        parameters.get("max_steps")
    )
    warmup_rules = bench_rules.get("warmup_steps") or {}
    if warmup is not None:
        wmin = _coerce_int(warmup_rules.get("min"))
        wmax = _coerce_int(warmup_rules.get("max"))
        if (wmin is not None and warmup < wmin) or (
            wmax is not None and warmup > wmax
        ):
            out.append(
                RuleViolation(
                    code="warmup_outside_legal",
                    message=f"warmup_steps {warmup} outside legal range [{wmin},{wmax}]",
                    parameter="warmup_steps",
                    observed=warmup,
                    legal={"min": wmin, "max": wmax},
                )
            )
    proportion_max = _coerce_float(bench_rules.get("warmup_proportion_of_max_steps_max"))
    if warmup is not None and max_steps is not None and proportion_max is not None and max_steps > 0:
        ratio = warmup / max_steps
        if ratio > proportion_max + 1e-9:
            out.append(
                RuleViolation(
                    code="warmup_above_proportion",
                    message=(
                        f"warmup_steps/max_steps={ratio:.4f} > limit "
                        f"{proportion_max}"
                    ),
                    parameter="warmup_steps",
                    observed=ratio,
                    legal={"max_ratio": proportion_max},
                )
            )

    lora_rank = _coerce_int(parameters.get("LORA_RANK") or parameters.get("lora_rank"))
    legal_lora = bench_rules.get("lora_rank_legal_values")
    if lora_rank is not None and legal_lora and lora_rank not in legal_lora:
        out.append(
            RuleViolation(
                code="lora_rank_illegal",
                message=f"LORA_RANK={lora_rank} not in {legal_lora}",
                parameter="LORA_RANK",
                observed=lora_rank,
                legal=legal_lora,
            )
        )

    ep = _coerce_int(parameters.get("EXPERT_PARALLEL") or parameters.get("expert_parallel"))
    legal_ep = bench_rules.get("expert_parallel_legal_values")
    if ep is not None and legal_ep and ep not in legal_ep:
        out.append(
            RuleViolation(
                code="expert_parallel_illegal",
                message=f"EXPERT_PARALLEL={ep} not in {legal_ep}",
                parameter="EXPERT_PARALLEL",
                observed=ep,
                legal=legal_ep,
            )
        )

    aux_coef = _coerce_float(parameters.get("AUX_LOSS_BALANCE_COEF"))
    aux_range = bench_rules.get("aux_loss_balance_coef_legal_range")
    if aux_coef is not None and aux_range:
        amin = _coerce_float(aux_range.get("min"))
        amax = _coerce_float(aux_range.get("max"))
        if (amin is not None and aux_coef < amin) or (
            amax is not None and aux_coef > amax
        ):
            out.append(
                RuleViolation(
                    code="aux_loss_coef_outside_legal",
                    message=(
                        f"AUX_LOSS_BALANCE_COEF={aux_coef} outside legal range "
                        f"[{amin},{amax}]"
                    ),
                    parameter="AUX_LOSS_BALANCE_COEF",
                    observed=aux_coef,
                    legal={"min": amin, "max": amax},
                )
            )

    non_borrow = bench_rules.get("non_borrowable_hyperparameters") or []
    for parameter in non_borrow:
        if parameter in parameters:
            out.append(
                RuleViolation(
                    code="non_borrowable_hyperparameter_changed",
                    message=(
                        f"hyperparameter {parameter!r} is not borrowable for "
                        f"benchmark {benchmark!r}"
                    ),
                    parameter=parameter,
                    observed=parameters[parameter],
                )
            )

    return out


def validate_proposal(
    rules: dict[str, Any], proposal: dict[str, Any], *, benchmark: str
) -> list[dict[str, Any]]:
    candidates = proposal.get("candidates") or []
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        params = candidate.get("parameters") if isinstance(candidate, dict) else None
        if not isinstance(params, dict):
            results.append(
                {
                    "index": index,
                    "valid": False,
                    "violations": [
                        RuleViolation(
                            code="invalid_candidate",
                            message="candidate.parameters missing or not an object",
                        ).to_dict()
                    ],
                }
            )
            continue
        nexp = _coerce_int(params.get("NEXP")) or 1
        synthetic = (
            str(params.get("USE_SYNTHETIC_DATA", "0")).lower() in {"1", "true", "yes"}
        )
        violations = validate_candidate(
            rules,
            benchmark=benchmark,
            parameters=params,
            nexp=nexp,
            use_synthetic_data=synthetic,
        )
        results.append(
            {
                "index": index,
                "valid": not violations,
                "violations": [v.to_dict() for v in violations],
            }
        )
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate a proposal against the rules")
    validate.add_argument("--proposal", type=Path, required=True)
    validate.add_argument("--benchmark", required=True)
    validate.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    validate.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rules = load_rules(args.rules)
    proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
    results = validate_proposal(rules, proposal, benchmark=args.benchmark)
    payload = {
        "schema_version": 1,
        "rules_id": rules.get("id"),
        "benchmark": args.benchmark,
        "valid_count": sum(1 for r in results if r["valid"]),
        "invalid_count": sum(1 for r in results if not r["valid"]),
        "results": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if payload["invalid_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
