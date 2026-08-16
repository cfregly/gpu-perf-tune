# Agent Rationale

Status: Reference

The concise operational rules live in
[`../AGENTS.md`](../AGENTS.md). In short,
agents treat tools as read-only unless the contract says otherwise. Tools with
an ack field require that explicit acknowledgement in the current turn. Local
artifact writers use explicit output paths without an extra ack. Results count
only when they are backed by citable evidence. The per-skill `SKILL.md` files
restate the rules where they apply.
