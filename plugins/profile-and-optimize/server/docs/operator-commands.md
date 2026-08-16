Status: Active
Audience: operators who need direct shell commands.

# Operator commands

Run `make help` at the repository root for the supported project commands. The direct commands below expose the eight parser-backed libraries.

| Task | Command |
| --- | --- |
| List the MCP surface | `python3 plugins/profile-and-optimize/server/mcp_surface.py list` |
| Inspect tuning commands | `python3 -m ai_tuning --help` |
| Compare profiler captures | `python3 -m profile profile-diff --help` |
| Record or compare a baseline | `python3 -m perf_baseline --help` |
| Create an evidence bundle | `python3 -m evidence --help` |
| Triage or guard Slurm work | `python3 -m slurm --help` |
| Record and render findings | `python3 -m findings --help` |
| Build an inference report | `perftunereport --help` |
| Check a known configuration | `python3 -m known_good_config --help` |

Run these commands from `plugins/profile-and-optimize/server` after installing the development environment. The full verb matrix, safety classes, and acknowledgement flags live in [`cli-contract.md`](cli-contract.md).
