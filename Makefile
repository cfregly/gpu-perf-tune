# profile-and-optimize operator commands.
#
# Run `make` (or `make help`) for a list of targets. POSIX-make-friendly;
# no GNU-make-only constructs.

.DEFAULT_GOAL := help

PLUGIN_DIR := plugins/profile-and-optimize
SERVER_DIR := $(PLUGIN_DIR)/server
SCRIPTS_DIR := scripts

# Default VERSION for `make release-notes`. Override on the command line:
#   make release-notes VERSION=v0.4.0
VERSION ?=

.PHONY: help validate validate-uncached smoke-test smoke-mcp-runtime check-doc-links lint-skill-mcp-args lint-skill-counts lint-tool-counts lint-versions pytest pytest-xdist all freshness bootstrap print-mcp-snippet doctor install-into-cursor refresh-symlinks release-notes mcp-surface clean-pycache

help:
	@printf 'profile-and-optimize operator commands\n\n'
	@printf 'Common targets:\n'
	@printf '  make all                     Run smoke-test + smoke-mcp-runtime + check-doc-links + lint-skill-mcp-args + pytest in series; use `make -j5 all` to run in parallel (target ~3s wall-clock)\n'
	@printf '  make validate                Run claude plugin validate on the plugin manifest (cached by manifest SHA; ~0.05s on a cache hit)\n'
	@printf '  make validate-uncached       Bypass the cache and re-run claude plugin validate unconditionally\n'
	@printf '  make smoke-test              Validate + frontmatter lint + canonical-counts verify (libraries / contract tools) + skill-count lint + tool-count lint (<2s)\n'
	@printf '  make smoke-mcp-runtime       End-to-end: spawn the bundled MCP server over stdio + verify the canonical MCP tool count (<2s)\n'
	@printf '  make check-doc-links         Verify every [text](path|url) link in profile-and-optimize-authored docs resolves; HTTP checks run in parallel (<2s for 55 URLs)\n'
	@printf '  make lint-skill-mcp-args     Cross-check SKILL.md `with:` arg blocks against MCP tool descriptors (catches the v0.8.1 filter/regex class of bug)\n'
	@printf '  make lint-skill-counts       Assert every doc that names the skill count agrees with the on-disk plugins/profile-and-optimize/skills/ tree\n'
	@printf '  make lint-tool-counts        Assert every doc that names a tool / library / aux-tool count agrees with the canonical constants in mcp_surface.py\n'
	@printf '  make lint-versions           Assert README + plugin-README version headers match plugin.json version\n'
	@printf '  make release                 Tag the current release commit vX.Y.Z (read from plugin.json) + push main + tag atomically (tagging rigidity)\n'
	@printf '  make pytest                  Run the bundled pytest suite sequentially (~700 profile-and-optimize-native tests in <1s; requires `bash server/install.sh --with-dev` first)\n'
	@printf '  make pytest-xdist            Same as `make pytest` but with `-n auto` (pytest-xdist parallel); slower for the ~700-test set due to worker-startup overhead; use only if you specifically want xdist semantics\n'
	@printf '  make freshness               Per-skill freshness report based on last_validated frontmatter; YELLOW > 90d, RED > 180d\n'
	@printf '  make bootstrap               One-shot Cursor/dev-clone setup: server venv + skill symlinks + ~/.cursor/mcp.json snippet (pass FULL=1 / DEV=1 to forward --full / --with-dev)\n'
	@printf '  make print-mcp-snippet       Print the ~/.cursor/mcp.json `profile_and_optimize` block with the venv path resolved to this checkout (read-only)\n'
	@printf '  make doctor                  Diagnose a stale/broken ~/.cursor/mcp.json profile_and_optimize entry after a version bump (read-only; FIX=1 repoints it with a backup)\n'
	@printf '  make install-into-cursor     Symlink every skill into ~/.cursor/skills/ (alias of refresh-symlinks for back-compat with v0.7.x docs)\n'
	@printf '  make refresh-symlinks        Re-symlink every plugins/profile-and-optimize/skills/<skill>/ into ~/.cursor/skills/<skill>/ — run this after every release\n'
	@printf '  make mcp-surface             Print the canonical MCP tool surface derived by mcp_surface.py (counts subcommand verifies live derivation matches the constants)\n'
	@printf '\n'
	@printf 'Less common:\n'
	@printf '  make release-notes VERSION=v0.4.0   Extract the CHANGELOG section for v0.4.0\n'
	@printf '  make clean-pycache           Remove __pycache__ + *.pyc under server/\n'
	@printf '\n'
	@printf 'Variables (override on command line):\n'
	@printf '  VERSION                      Version tag for release-notes (e.g. v0.4.0)\n'

validate:
	bash $(SCRIPTS_DIR)/validate-cached.sh

validate-uncached:
	claude plugin validate $(PLUGIN_DIR)

smoke-test: validate
	@echo '--- frontmatter lint ---'
	@python3 -c "$$FRONTMATTER_LINT_PY"
	@echo '--- mcp_surface canonical counts ---'
	@python3 $(SERVER_DIR)/mcp_surface.py counts
	@echo '--- skill / tool count lints ---'
	@python3 $(SCRIPTS_DIR)/lint-skill-counts.py
	@python3 $(SCRIPTS_DIR)/lint-tool-counts.py
	@echo '--- version-header lint ---'
	@python3 $(SCRIPTS_DIR)/lint-versions.py

mcp-surface:
	python3 $(SERVER_DIR)/mcp_surface.py list

install-into-cursor: refresh-symlinks

refresh-symlinks:
	bash $(SCRIPTS_DIR)/install-skills-into-cursor.sh

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


pytest:
	@if [ ! -x "$(SERVER_DIR)/.venv/bin/pytest" ]; then \
	  echo '[FAIL] pytest not installed; run: bash $(SERVER_DIR)/install.sh --with-dev'; \
	  exit 2; \
	fi
	# Sequential pytest. The 700 profile-and-optimize-native tests average ~2ms each;
	# pytest-xdist worker-process startup makes `-n auto` ~3.5x slower than
	# sequential for this test set (1.58s xdist vs 0.45s sequential). The
	# xdist dep stays in the `dev` extras for ad-hoc parallel runs against
	# larger selections via `make pytest-xdist`.
	cd $(SERVER_DIR) && .venv/bin/python -m pytest -q

pytest-xdist:
	@if [ ! -x "$(SERVER_DIR)/.venv/bin/pytest" ]; then \
	  echo '[FAIL] pytest not installed; run: bash $(SERVER_DIR)/install.sh --with-dev'; \
	  exit 2; \
	fi
	cd $(SERVER_DIR) && .venv/bin/python -m pytest -n auto -q




freshness:
	@python3 $(SCRIPTS_DIR)/freshness-report.py

# `make all` runs the four operator-facing gate checks. The four targets are
# mutually independent (no shared mutable repo state) so `make -j4 all` runs
# them concurrently for ~3s wall-clock instead of ~8s sequential.
all: smoke-test smoke-mcp-runtime check-doc-links lint-skill-mcp-args pytest
	@echo '[ok] all checks passed (smoke-test + smoke-mcp-runtime + check-doc-links + lint-skill-mcp-args + pytest)'

lint-skill-mcp-args:
	@python3 $(SCRIPTS_DIR)/lint-skill-mcp-args.py

lint-skill-counts:
	@python3 $(SCRIPTS_DIR)/lint-skill-counts.py

lint-tool-counts:
	@python3 $(SCRIPTS_DIR)/lint-tool-counts.py

lint-versions:
	@python3 $(SCRIPTS_DIR)/lint-versions.py

release:
	@bash $(SCRIPTS_DIR)/release.sh

release-notes:
	@if [ -z "$(VERSION)" ]; then \
	  echo 'usage: make release-notes VERSION=v0.X.Y'; \
	  exit 2; \
	fi
	@awk '/^## \[$(VERSION:v%=%)\]/{flag=1; print; next} /^## \[/ && flag{exit} flag' CHANGELOG.md

clean-pycache:
	find $(SERVER_DIR) -type d -name __pycache__ -prune -exec rm -rf {} +
	find $(SERVER_DIR) -type f -name '*.pyc' -delete

# Frontmatter-lint program. Embedded here so `make smoke-test` is single-command.
define FRONTMATTER_LINT_PY
import yaml, glob, sys
ok = True
for f in sorted(glob.glob('$(PLUGIN_DIR)/skills/*/SKILL.md')):
    body = open(f).read()
    if not body.startswith('---\n'):
        print(f, '-> NO FRONTMATTER'); ok = False; continue
    end = body.find('\n---\n', 4)
    if end < 0:
        print(f, '-> NO FRONTMATTER END'); ok = False; continue
    try:
        fm = yaml.safe_load(body[4:end])
    except Exception as e:
        print(f, '-> YAML ERROR:', e); ok = False; continue
    dir_name = f.split('/')[-2]
    name = fm.get('name')
    desc = fm.get('description', '')
    at = fm.get('allowed-tools', [])
    if dir_name == '_template':
        continue
    if name != dir_name:
        print(f, '-> NAME MISMATCH', name, 'vs', dir_name); ok = False; continue
    if len(desc) > 1024:
        print(f, '-> DESC TOO LONG', len(desc)); ok = False; continue
    if not isinstance(at, list):
        print(f, '-> allowed-tools not list'); ok = False; continue
print('[ok] frontmatter lint clean' if ok else '[FAIL] frontmatter lint failed')
sys.exit(0 if ok else 1)
endef
export FRONTMATTER_LINT_PY
