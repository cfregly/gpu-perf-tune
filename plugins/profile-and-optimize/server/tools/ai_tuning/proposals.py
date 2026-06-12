"""Proposal and template-patch helpers for the AI tuner."""

from __future__ import annotations

try:  # Package import, e.g. import tools.ai_tuning.proposals.
    from .ai_tuning import (
        command_proposal_diff,
        command_proposal_validate,
        command_template_patch_validate,
    )
    from .helpers import (
        build_remaining_candidates,
        candidate_key,
        candidate_seen,
        evaluate_template_patch_request,
        finite_parameter_domains,
        iter_candidate_parameters,
        materialize_record_config_patches,
        normalize_candidate_parameters,
        normalize_config_patches,
        parameter_index,
        validate_patch_safety,
        validate_patched_template_structure,
    )
except ImportError:  # Direct import from tools/ai_tuning.
    from ai_tuning import (
        command_proposal_diff,
        command_proposal_validate,
        command_template_patch_validate,
    )
    from helpers import (
        build_remaining_candidates,
        candidate_key,
        candidate_seen,
        evaluate_template_patch_request,
        finite_parameter_domains,
        iter_candidate_parameters,
        materialize_record_config_patches,
        normalize_candidate_parameters,
        normalize_config_patches,
        parameter_index,
        validate_patch_safety,
        validate_patched_template_structure,
    )

__all__ = [
    "build_remaining_candidates",
    "candidate_key",
    "candidate_seen",
    "command_proposal_diff",
    "command_proposal_validate",
    "command_template_patch_validate",
    "evaluate_template_patch_request",
    "finite_parameter_domains",
    "iter_candidate_parameters",
    "materialize_record_config_patches",
    "normalize_candidate_parameters",
    "normalize_config_patches",
    "parameter_index",
    "validate_patch_safety",
    "validate_patched_template_structure",
]
