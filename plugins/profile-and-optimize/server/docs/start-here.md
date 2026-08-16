Status: Active
Audience: anyone new to the bundled MCP server.

# Start here

Benchmark results depend on the hardware, software, workload, and measurement method. Do not quote local results as universal claims.

Choose the shortest path for your task:

| Task | Start here |
| --- | --- |
| Evaluate the project | [`README.md`](../../../../README.md) |
| Set up a development checkout | [`first-hour.md`](first-hour.md) |
| Install the MCP server | [`INSTALL.md`](../tools/profile_and_optimize_mcp/INSTALL.md) |
| Review the CLI and MCP contract | [`cli-contract.md`](cli-contract.md) |
| Review tool input and output rules | [`mcp-tool-io-contract.md`](mcp-tool-io-contract.md) |
| Run a cluster workflow | [`runbooks/`](../runbooks) |
| Contribute a change | [`CONTRIBUTING.md`](../../../../CONTRIBUTING.md) |

## Mental model

The server packages GPU performance workflows behind eight CLI libraries. [`mcp_surface.py`](../mcp_surface.py) derives 51 MCP tools from those CLIs and adds two read-only search tools. The same parsers and safety labels serve shell users, Python callers, and MCP clients.

The toolkit supports inference profiling, benchmarking, tuning, evidence capture, findings, and guarded Slurm operations. Some lower-level utilities originated in MLPerf-oriented work, but MLPerf is not the product boundary.

## Directory map

| Path | Contents |
| --- | --- |
| [`../README.md`](../README.md) | Server overview and verification |
| [`../mcp_surface.py`](../mcp_surface.py) | Canonical library and tool counts |
| [`../tools/`](../tools) | Tool implementations and tests |
| [`../runbooks/`](../runbooks) | Operator procedures |
| [`../experiments/artifacts/`](../experiments/artifacts) | Synthetic examples and local evidence layouts |
| [`../tuning/`](../tuning) | Tuning rules and known configurations |
| [`AGENTS.md`](../../../../AGENTS.md) | Project policy for contributors and coding agents |

Mutating tools use explicit acknowledgement flags. Read the tool help and the relevant runbook before submitting jobs, changing Slurm node state, pulling restricted data, or writing external systems.
