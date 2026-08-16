"""Findings library.

Structured findings YAML + record / render / diff verbs.
Schema: docs/findings-schema.md.
"""

from .findings_cli import CONTRACT, build_parser, main

__all__ = ["CONTRACT", "build_parser", "main"]
