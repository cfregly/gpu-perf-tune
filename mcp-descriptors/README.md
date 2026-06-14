# mcp-descriptors

Offline snapshots of MCP tool schemas, used by `scripts/lint-skill-mcp-args.py`
to validate the `with:` blocks in each `SKILL.md` without a live MCP connection.

One directory per server, one JSON file per tool. The shipped skill set
references three servers - the bundled `profile_and_optimize` (validated
directly against `mcp_surface.py`, no snapshot needed) plus the optional
external `prometheus_mcp` and `zymtrace` servers. No snapshots are bundled
for the external servers. The lint treats a missing directory as "schema
validation skipped for that server".

To refresh a snapshot, connect the server in your client, dump its tool list,
and re-serialize each tool as `<dir>/tools/<tool_name>.json`. Keep only the
directories for servers your skills actually reference - the lint treats a
missing directory as "server not in use".
