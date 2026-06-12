"""perf_tune_report_kernel_profile verb (v1.21.0).

Capture per-kernel CUDA profile data from a live vLLM inference pod via an
nsys-enabled debug sidecar (no production image rebuild). Implements the
recipe documented in the ``inference-kernel-profile`` skill.

Always ack-gated: the verb attaches an ephemeral container to a running
pod, which is a write to the cluster.

See ``kernel_profile.py`` for the step-function architecture and
``test_kernel_profile.py`` for the test surface.
"""

from tools.perf_tune_report.kernel_profile.kernel_profile import (
    KernelProfileResult,
    KernelProfileStepFns,
    capture_kernel_profile,
)

__all__ = [
    "KernelProfileResult",
    "KernelProfileStepFns",
    "capture_kernel_profile",
]
