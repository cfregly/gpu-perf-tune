# tools/

Tool libraries backing the bundled MCP server. Each library ships a CLI
(`<lib>_cli.py` or `cli.py`) whose `CONTRACT` defines the MCP verbs exposed
through `server/mcp_surface.py`.

See `server/docs/cli-contract.md` for the envelope/exit-code contract and
`server/docs/mcp-tool-io-contract.md` for the MCP surface contract.
