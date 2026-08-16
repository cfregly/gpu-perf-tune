# profile-and-optimize operator commands.
#
# Run `make` (or `make help`) for a list of targets. POSIX-make-friendly;
# no GNU-make-only constructs.

.DEFAULT_GOAL := help

PLUGIN_DIR := plugins/profile-and-optimize
SERVER_DIR := $(PLUGIN_DIR)/server
SCRIPTS_DIR := scripts
SERVER_PY := $(SERVER_DIR)/.venv/bin/python
PYTHON ?= $(if $(wildcard $(SERVER_PY)),$(SERVER_PY),python3)

# Default VERSION for `make release-notes`. Override on the command line:
#   make release-notes VERSION=v0.3.0
VERSION ?=

.PHONY: help demo check validate validate-agent-skills validate-claude-plugin validate-claude-plugin-uncached validate-uncached smoke-test smoke-mcp-runtime check-doc-links workload-proof-check lint-skill-mcp-args lint-skill-counts lint-tool-counts lint-versions check-version-transition test-release-gates pytest pytest-mcp pytest-xdist all freshness bootstrap print-mcp-snippet doctor install-mcp install-skills install-into-codex install-into-cursor refresh-symlinks release release-notes mcp-surface clean-pycache

help:
	@printf 'profile-and-optimize operator commands\n\n'
	@printf 'Common targets:\n'
	@printf '  make demo                     List all skills and a sample of the MCP surface. No GPU required\n'
	@printf '  make check                    Run the public static gates, including Agent Skills and all Markdown links\n'
	@printf '  make all                      Run the full client-neutral local test suite. Use `make -j4 all` for parallel execution\n'
	@printf '  make validate-agent-skills    Validate every skill with the official Agent Skills reference validator\n'
	@printf '  make validate-claude-plugin   Run the optional Claude Code package validator\n'
	@printf '  make smoke-test               Validate skills, canonical counts, versions, and MCP argument references\n'
	@printf '  make smoke-mcp-runtime       End-to-end: spawn the bundled MCP server over stdio + verify the canonical MCP tool count (<2s)\n'
	@printf '  make check-doc-links         Verify every Markdown link in the repository. HTTP checks run in parallel\n'
	@printf '  make lint-skill-mcp-args     Cross-check SKILL.md `with:` arg blocks against MCP tool descriptors\n'
	@printf '  make lint-skill-counts       Assert every doc that names the skill count agrees with the on-disk plugins/profile-and-optimize/skills/ tree\n'
	@printf '  make lint-tool-counts        Assert every doc that names a tool / library / aux-tool count agrees with the canonical constants in mcp_surface.py\n'
	@printf '  make lint-versions           Assert public version surfaces match the root VERSION file\n'
	@printf '  make check-version-transition Reject release version regressions against the current base\n'
	@printf '  make workload-proof-check    Validate every checked-in workload-proof-packet.json against the neocloud packet and workflow handoff gates\n'
	@printf '  make release                 Recover a missing release from the current, already-green main version bump\n'
	@printf '  make pytest                  Run the bundled tool tests. Requires `bash plugins/profile-and-optimize/server/install.sh --with-dev` first\n'
	@printf '  make pytest-mcp              Rerun only the MCP server and client-configuration tests\n'
	@printf '  make pytest-xdist            Run the full test set with `-n auto`. Use only when you need xdist semantics\n'
	@printf '  make freshness               Per-skill freshness report based on metadata.last-validated\n'
	@printf '  make install-mcp CLIENT=...  Install MCP for first-class claude or codex. Other client helpers are best effort\n'
	@printf '  make install-skills CLIENT=codex|cursor  Link the Agent Skills into the selected client\n'
	@printf '  make install-into-codex      Link every skill into ~/.agents/skills/\n'
	@printf '  make mcp-surface             Print the canonical MCP tool surface derived by mcp_surface.py (counts subcommand verifies live derivation matches the constants)\n'
	@printf '\n'
	@printf 'Less common:\n'
	@printf '  make install-into-cursor     Link every skill into ~/.cursor/skills/ (best-effort adapter)\n'
	@printf '  make refresh-symlinks        Compatibility alias for install-into-cursor\n'
	@printf '  make bootstrap               Set up the best-effort Cursor adapter from a clone\n'
	@printf '  make print-mcp-snippet       Print a Cursor MCP config block without writing it\n'
	@printf '  make doctor                  Diagnose a Cursor MCP entry. Read-only unless FIX=1\n'
	@printf '  make release-notes VERSION=v0.3.0   Extract the CHANGELOG section for v0.3.0\n'
	@printf '  make clean-pycache           Remove __pycache__ + *.pyc under server/\n'
	@printf '\n'
	@printf 'Variables (override on command line):\n'
	@printf '  VERSION                      Version tag for release-notes (e.g. v0.3.0)\n'

validate: validate-agent-skills

validate-agent-skills:
	@if command -v skills-ref >/dev/null 2>&1; then \
		validator=$$(command -v skills-ref); \
	elif [ -x "$(SERVER_DIR)/.venv/bin/skills-ref" ]; then \
		validator="$(SERVER_DIR)/.venv/bin/skills-ref"; \
	else \
		echo '[FAIL] skills-ref is not installed. Run: bash $(SERVER_DIR)/install.sh --with-dev'; \
		exit 2; \
	fi; \
	for skill in $(PLUGIN_DIR)/skills/*; do "$$validator" validate "$$skill" || exit; done

validate-claude-plugin:
	bash $(SCRIPTS_DIR)/validate-cached.sh

validate-claude-plugin-uncached:
	claude plugin validate $(PLUGIN_DIR)

validate-uncached: validate-claude-plugin-uncached

smoke-test: validate-agent-skills
	@echo '--- mcp_surface canonical counts ---'
	@$(PYTHON) $(SERVER_DIR)/mcp_surface.py counts
	@echo '--- skill / tool count lints ---'
	@$(PYTHON) $(SCRIPTS_DIR)/lint-skill-counts.py
	@$(PYTHON) $(SCRIPTS_DIR)/lint-tool-counts.py
	@echo '--- version-header lint ---'
	@$(PYTHON) $(SCRIPTS_DIR)/lint-versions.py
	@echo '--- skill MCP argument lint ---'
	@$(PYTHON) $(SCRIPTS_DIR)/lint-skill-mcp-args.py

check: smoke-test
	@$(PYTHON) scripts/check-version-transition.py
	@$(PYTHON) scripts/test_release_gates.py
	@$(PYTHON) scripts/check_docs.py
	@$(PYTHON) scripts/check_workload_proof_packets.py --self-test --require-workflow-handoff
	@bash scripts/check-doc-links.sh --no-network --quiet

demo:
	@printf '== gpu-perf-tune: 32 Agent Skills ==\n'
	@find $(PLUGIN_DIR)/skills -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort | sed 's/^/  /'
	@printf '\n== MCP surface ==\n'
	@$(PYTHON) $(SERVER_DIR)/mcp_surface.py counts
	@printf '\nRepresentative tools:\n'
	@$(PYTHON) $(SERVER_DIR)/mcp_surface.py list | sed -n '3,14p'
	@printf '\nRun `make mcp-surface` for all tools. A performance run needs the target workload and suitable GPU hardware.\n'

mcp-surface:
	$(PYTHON) $(SERVER_DIR)/mcp_surface.py list

install-skills:
	@if [ -z "$(CLIENT)" ]; then \
		echo 'usage: make install-skills CLIENT=codex|cursor'; \
		exit 2; \
	fi
	bash $(SCRIPTS_DIR)/install-agent-skills.sh --client "$(CLIENT)"

install-into-codex:
	bash $(SCRIPTS_DIR)/install-agent-skills.sh --client codex

install-into-cursor:
	bash $(SCRIPTS_DIR)/install-agent-skills.sh --client cursor

install-mcp:
	@if [ -z "$(CLIENT)" ]; then \
		echo 'usage: make install-mcp CLIENT=claude|cursor|codex|gemini|antigravity'; \
		exit 2; \
	fi
	bash $(SERVER_DIR)/tools/profile_and_optimize_mcp/scripts/install_profile_and_optimize_mcp.sh --client "$(CLIENT)"

refresh-symlinks: install-into-cursor

# One-shot Cursor/dev-clone setup. Forward --full / --with-dev via FULL=1 / DEV=1:
#   make bootstrap FULL=1 DEV=1
bootstrap:
	bash $(SCRIPTS_DIR)/bootstrap.sh $(if $(FULL),--full,) $(if $(DEV),--with-dev,)

print-mcp-snippet:
	bash $(SCRIPTS_DIR)/print-cursor-mcp-snippet.sh

# Diagnose a stale ~/.cursor/mcp.json profile_and_optimize entry. Read-only by default;
# FIX=1 repoints it (with a timestamped backup):
#   make doctor FIX=1
doctor:
	bash $(SCRIPTS_DIR)/cursor-mcp-doctor.sh $(if $(FIX),--fix,)

smoke-mcp-runtime:
	bash $(SCRIPTS_DIR)/smoke-mcp-runtime.sh

check-doc-links:
	bash $(SCRIPTS_DIR)/check-doc-links.sh

workload-proof-check:
	@$(PYTHON) $(SCRIPTS_DIR)/check_workload_proof_packets.py --self-test --require-workflow-handoff


pytest:
	@if [ ! -x "$(SERVER_DIR)/.venv/bin/pytest" ]; then \
	  echo '[FAIL] pytest not installed; run: bash $(SERVER_DIR)/install.sh --with-dev'; \
	  exit 2; \
	fi
	# Keep the default run deterministic. Use pytest-xdist for an explicit
	# parallel run when its worker startup cost is worthwhile.
	cd $(SERVER_DIR) && .venv/bin/python -m pytest -q
	bash $(SERVER_DIR)/tools/shared/test_capture_cmd.sh

pytest-mcp:
	@if [ ! -x "$(SERVER_DIR)/.venv/bin/pytest" ]; then \
		echo '[FAIL] pytest not installed. Run: bash $(SERVER_DIR)/install.sh --with-dev'; \
		exit 2; \
	fi
	cd $(SERVER_DIR) && .venv/bin/python -m pytest -q tools/profile_and_optimize_mcp/tests

pytest-xdist:
	@if [ ! -x "$(SERVER_DIR)/.venv/bin/pytest" ]; then \
	  echo '[FAIL] pytest not installed; run: bash $(SERVER_DIR)/install.sh --with-dev'; \
	  exit 2; \
	fi
	cd $(SERVER_DIR) && .venv/bin/python -m pytest -n auto -q




freshness:
	@$(PYTHON) $(SCRIPTS_DIR)/freshness-report.py

# These targets are independent and can run in parallel with `make -j4 all`.
# The static check includes offline Markdown links. Scheduled CI checks live
# external URLs so local validation does not depend on network availability.
all: check smoke-mcp-runtime pytest
	@echo '[ok] all client-neutral checks passed'

lint-skill-mcp-args:
	@$(PYTHON) $(SCRIPTS_DIR)/lint-skill-mcp-args.py

lint-skill-counts:
	@$(PYTHON) $(SCRIPTS_DIR)/lint-skill-counts.py

lint-tool-counts:
	@$(PYTHON) $(SCRIPTS_DIR)/lint-tool-counts.py

lint-versions:
	@$(PYTHON) $(SCRIPTS_DIR)/lint-versions.py

check-version-transition:
	@$(PYTHON) $(SCRIPTS_DIR)/check-version-transition.py

test-release-gates:
	@$(PYTHON) $(SCRIPTS_DIR)/test_release_gates.py

release:
	@bash $(SCRIPTS_DIR)/release.sh

release-notes:
	@if [ -z "$(VERSION)" ]; then \
	  echo 'usage: make release-notes VERSION=v0.X.Y'; \
	  exit 2; \
	fi
	@$(PYTHON) $(SCRIPTS_DIR)/extract-release-notes.py "$(VERSION)" CHANGELOG.md

clean-pycache:
	find $(SERVER_DIR) -type d -name __pycache__ -prune -exec rm -rf {} +
	find $(SERVER_DIR) -type f -name '*.pyc' -delete
