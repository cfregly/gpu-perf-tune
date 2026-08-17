"""Proposal and template-patch helpers for the AI tuner."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # Package import, e.g. import tools.ai_tuning.proposals.
    from .cmd_proposal import (
        command_proposal_diff,
        command_proposal_validate,
    )
    from .cmd_template_patch import (
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
    from cmd_proposal import (
        command_proposal_diff,
        command_proposal_validate,
    )
    from cmd_template_patch import (
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
