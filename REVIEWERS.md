# Reviewing `gpu-perf-tune` changes

Review the project surface that changed. Claude Code packaging is one adapter,
not the default review boundary.

## Fast review path

1. Read the changed skills, contracts, runtime code, and public docs.
2. Run `make all`.
3. Check the version and changelog against the change scope.
4. Confirm safety gates fail closed and require the documented acknowledgement.
5. Confirm claims point to reproducible evidence.
6. Run `make validate-claude-plugin` when Claude packaging changed.

## Sources of truth

| Path | Review focus |
| --- | --- |
| `plugins/profile-and-optimize/skills/*/SKILL.md` | Task scope, official frontmatter, workflow, safety, and evidence |
| `plugins/profile-and-optimize/templates/skill/SKILL.md` | New skill defaults and current specification |
| `plugins/profile-and-optimize/server/mcp_surface.py` | Public tool registry and canonical counts |
| `plugins/profile-and-optimize/server/` | Contracts, implementations, tests, and runtime behavior |
| `VERSION` | Release version source of truth |
| `plugins/profile-and-optimize/.claude-plugin/plugin.json` | Claude adapter metadata and mirrored version |
| `plugins/profile-and-optimize/.mcp.json` | Bundled and optional MCP declarations |
| `AGENTS.md` | Shared agent policy |
| `CHANGELOG.md` | User-visible release history |

## Skill review

The official Agent Skills header may contain `name`, `description`, `license`,
`compatibility`, `metadata`, and `allowed-tools`.

Check that:

- `name` matches the directory and uses the supported character set.
- `description` names the task and concrete trigger phrases.
- `license` is correct, including for adapted third-party work.
- `metadata.last-validated` is a quoted ISO date.
- `allowed-tools` is a space-delimited string with the narrowest required set.
- Prerequisites fail closed.
- Each phase has a clear checkpoint.
- External writes and cluster changes require explicit approval.
- The output records enough context to reproduce and challenge the result.
- Shared rules are linked, not copied into the skill.

`make validate-agent-skills` is the specification gate. Do not replace it with
an ad hoc YAML parser.

## Runtime and MCP review

The bundled server has 8 libraries, 51 contract tools, and 2 auxiliary tools.
The exact surface comes from `mcp_surface.py`.

For a runtime change, check:

- Valid and invalid inputs have tests.
- Return envelopes keep the documented shape.
- Read-only tools do not mutate external state.
- Mutating tools expose the correct safety class and acknowledgement field.
- Tool names and arguments still match the skill references.
- The live MCP handshake lists the canonical tool count and completes a call.

Run:

```bash
make pytest
make smoke-mcp-runtime
make lint-skill-mcp-args
```

## Version review

| Change | Expected bump |
| --- | --- |
| Documentation or behavior-preserving fix | PATCH |
| New skill, tool, or supported workflow | MINOR |
| Breaking public skill, tool, or contract change before 1.0.0 | MINOR plus migration notes |
| Breaking public skill, tool, or contract change from 1.0.0 onward | MAJOR |

The root `VERSION`, adapter manifest, Python packages, package README, and
changelog must agree. A release commit must receive the matching annotated tag.

## Blocking findings

Request changes when:

- Any required `make all` gate fails.
- Official Agent Skills validation fails.
- A public tool or skill changes without the matching version treatment.
- A safety check can be bypassed without explicit acknowledgement.
- A performance claim omits its workload, baseline, context, or evidence.
- A public artifact contains credentials, private infrastructure details, or
  customer data.
- Third-party material lacks a verified source and license.
- Documentation points to missing files or unsupported client behavior.

## Review comment

A useful approval records the checks that actually ran:

```text
Validated:
- make all: PASS
- version and changelog: PASS
- safety and acknowledgement paths: PASS
- public docs and redaction: PASS
- Claude adapter validation: PASS or not applicable
```

For questions about the review process, open an operator question with the
public issue template. Send security or conduct reports through their private
channels.
