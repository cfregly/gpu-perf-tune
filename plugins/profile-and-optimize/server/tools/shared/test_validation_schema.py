"""Tests for validation_schema.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.shared import validation_schema  # noqa: E402


def _load_validate_artifacts():
    path = REPO_ROOT / "validate_artifacts.py"
    spec = importlib.util.spec_from_file_location("validate_artifacts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SchemaModuleTests(unittest.TestCase):
    def test_benchmark_columns_is_immutable_tuple(self) -> None:
        self.assertIsInstance(validation_schema.BENCHMARK_COLUMNS, tuple)
        self.assertGreater(len(validation_schema.BENCHMARK_COLUMNS), 0)
        self.assertEqual(
            len(validation_schema.BENCHMARK_COLUMNS),
            len(set(validation_schema.BENCHMARK_COLUMNS)),
        )

    def test_required_summary_fields_is_immutable_tuple(self) -> None:
        self.assertIsInstance(validation_schema.REQUIRED_SUMMARY_FIELDS, tuple)
        self.assertGreater(len(validation_schema.REQUIRED_SUMMARY_FIELDS), 0)
        self.assertEqual(
            len(validation_schema.REQUIRED_SUMMARY_FIELDS),
            len(set(validation_schema.REQUIRED_SUMMARY_FIELDS)),
        )

    def test_required_summary_fields_are_lowercase_snake(self) -> None:
        for field in validation_schema.REQUIRED_SUMMARY_FIELDS:
            with self.subTest(field=field):
                self.assertEqual(field, field.lower())
                self.assertNotIn(" ", field)


class ValidateArtifactsReExportTests(unittest.TestCase):
    def test_validate_artifacts_re_exports_match_schema_module(self) -> None:
        validator = _load_validate_artifacts()
        self.assertEqual(
            tuple(validator.BENCHMARK_COLUMNS),
            validation_schema.BENCHMARK_COLUMNS,
        )
        self.assertEqual(
            tuple(validator.REQUIRED_SUMMARY_FIELDS),
            validation_schema.REQUIRED_SUMMARY_FIELDS,
        )

    def test_validate_artifacts_all_names_resolve(self) -> None:
        validator = _load_validate_artifacts()
        for name in validator.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(validator, name))


if __name__ == "__main__":
    unittest.main()
