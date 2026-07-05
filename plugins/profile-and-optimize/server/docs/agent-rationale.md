# Agent Rationale

Status: Reference

The concise operational rules live in [`../CLAUDE.md`](/plugins/profile-and-optimize/server/CLAUDE.md). In short: agents get a read-only surface by default, every mutating tool requires an explicit `i_understand_this_*` acknowledgement in the current turn, and results only count when they are backed by citable evidence. The per-skill SKILL.md files restate the rules where they apply.
