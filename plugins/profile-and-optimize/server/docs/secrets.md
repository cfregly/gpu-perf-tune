# Secrets And Credentials

Status: Active
Audience: operators and engineers running dataset, image, or cluster workflows.

Secrets are local operator prerequisites. They are never durable repo evidence
and must not be copied into `experiments/artifacts/` or submission packages.

Keep a non-secret `.env.example`-style template for local variable names, and
keep the real `.env` untracked and scoped to the current workstation or
cluster login environment.

| Credential | Used for | Storage expectation |
| --- | --- | --- |
| `HF_TOKEN` | Hugging Face license-gated assets. | Local shell, password manager, or approved secret store. |
| `NGC_API_KEY` / `NGC_CLI_API_KEY` | NVIDIA NGC private images and artifacts. | Local shell or approved secret store. Never in docs or logs. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Dataset staging helpers that use gcloud. | Local path to a user-approved credential file. Do not copy into artifacts. |
| `PROFILE_AND_OPTIMIZE_LOGIN_HOST` | Optional SSH target override for read-only cluster helper commands. | Local shell only. Not a secret, but environment-specific. |

If a workflow writes a run context or provenance file, record that a credential
class was required, not the credential value.
