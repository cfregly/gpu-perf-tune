from __future__ import annotations

import json
import unittest
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[4]


class PluginManifestTests(unittest.TestCase):
    def test_manifest_declares_only_the_bundled_server(self) -> None:
        manifest = json.loads((PLUGIN_ROOT / ".mcp.json").read_text(encoding="utf-8"))

        self.assertEqual(set(manifest["mcpServers"]), {"profile_and_optimize"})
        server = manifest["mcpServers"]["profile_and_optimize"]
        self.assertEqual(server["command"], "${CLAUDE_PLUGIN_ROOT}/server/.venv/bin/python")
        self.assertEqual(
            server["env"],
            {"PROFILE_AND_OPTIMIZE_REPO_ROOT": "${CLAUDE_PLUGIN_ROOT}/server"},
        )


if __name__ == "__main__":
    unittest.main()
